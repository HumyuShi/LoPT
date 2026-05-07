"""Model wrapper for localized post-training.

LoPT keeps the forward function value identical to a standard decoder-only
causal LM, but stops task-loss gradients at block boundaries. Non-final blocks
receive a local reconstruction objective, while the final block receives the
task objective. This file intentionally avoids project-specific paths and
supports common Llama/Qwen-style decoder-only Hugging Face models.
"""

from __future__ import annotations

import inspect
import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


def build_block_ranges(num_layers: int, num_blocks: int) -> List[Tuple[int, int]]:
    if num_blocks < 2:
        raise ValueError(f"num_blocks must be >= 2, got {num_blocks}")
    if num_blocks > num_layers:
        raise ValueError(f"num_blocks={num_blocks} exceeds num_layers={num_layers}")
    base = num_layers // num_blocks
    extra = num_layers % num_blocks
    ranges = []
    start = 0
    for block_idx in range(num_blocks):
        width = base + (1 if block_idx < extra else 0)
        end = start + width
        ranges.append((start, end))
        start = end
    return ranges


def _get_inner_decoder(model: PreTrainedModel) -> nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        raise NotImplementedError("GPT-style transformer.h models are not supported yet.")
    raise AttributeError(
        "LoPT expects a decoder-only HF model with `model.layers`, e.g. Llama/Qwen/Mistral."
    )


def _init_aux_decoder(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight, gain=0.1)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def _module_parameters(modules: Iterable[nn.Module]) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            params.append(param)
    return params


class LoPTModelForCausalLM(PreTrainedModel, GenerationMixin):
    """Gradient-isolated wrapper around a decoder-only causal LM.

    The wrapped model is saved as a normal Hugging Face causal LM for inference.
    Auxiliary reconstruction heads are training-only and are not needed after
    post-training.
    """

    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def __init__(
        self,
        base_model: PreTrainedModel,
        num_blocks: int = 2,
        aux_loss: str = "recon",
        detach_boundaries: bool = True,
    ) -> None:
        super().__init__(base_model.config)
        if aux_loss != "recon":
            raise ValueError("The open-source LoPT wrapper currently supports aux_loss='recon'.")
        # Do not call this attribute `base_model`: PreTrainedModel already has
        # a `base_model` property whose semantics vary across architectures.
        self.wrapped_model = base_model
        self.num_blocks = int(num_blocks)
        self.aux_loss_type = aux_loss
        self.detach_boundaries = detach_boundaries
        self.gradient_checkpointing = False
        self._last_aux_loss: Optional[torch.Tensor] = None

        self.inner = _get_inner_decoder(self.wrapped_model)
        self.num_layers = len(self.inner.layers)
        self.block_ranges = build_block_ranges(self.num_layers, self.num_blocks)

        self._untie_output_embeddings_if_needed()

        hidden_size = self.config.hidden_size
        bottleneck = max(hidden_size // 4, 1)
        self.aux_decoders = nn.ModuleList()
        for _ in range(self.num_blocks - 1):
            decoder = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, hidden_size),
            )
            decoder.to(dtype=next(self.wrapped_model.parameters()).dtype)
            _init_aux_decoder(decoder)
            self.aux_decoders.append(decoder)

        logger.info(
            "LoPT wrapper: %d layers split into %d blocks: %s",
            self.num_layers,
            self.num_blocks,
            self.block_ranges,
        )

    @classmethod
    def from_pretrained_lopt(
        cls,
        model_name_or_path: str,
        *,
        num_blocks: int = 2,
        aux_loss: str = "recon",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "LoPTModelForCausalLM":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(base_model, num_blocks=num_blocks, aux_loss=aux_loss)

    def _untie_output_embeddings_if_needed(self) -> None:
        if not getattr(self.config, "tie_word_embeddings", False):
            return
        input_embeddings = self.wrapped_model.get_input_embeddings()
        output_embeddings = self.wrapped_model.get_output_embeddings()
        if output_embeddings is None or input_embeddings is None:
            return
        if output_embeddings.weight is input_embeddings.weight:
            output_embeddings.weight = nn.Parameter(output_embeddings.weight.detach().clone())
            self.config.tie_word_embeddings = False
            logger.info("Untied input/output embeddings for localized optimization.")

    def get_input_embeddings(self):
        return self.wrapped_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.wrapped_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.wrapped_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.wrapped_model.set_output_embeddings(new_embeddings)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.wrapped_model.prepare_inputs_for_generation(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.wrapped_model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, *args, **kwargs):
        state_dict = kwargs.get("state_dict")
        if state_dict is not None:
            cleaned = {}
            for key, value in state_dict.items():
                if key.startswith("wrapped_model."):
                    cleaned[key[len("wrapped_model."):]] = value
                elif key.startswith("module.wrapped_model."):
                    cleaned[key[len("module.wrapped_model."):]] = value
            kwargs["state_dict"] = cleaned or None
        return self.wrapped_model.save_pretrained(save_directory, *args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True
        if hasattr(self.wrapped_model, "gradient_checkpointing_enable"):
            self.wrapped_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        if hasattr(self.wrapped_model, "gradient_checkpointing_disable"):
            self.wrapped_model.gradient_checkpointing_disable()

    def add_model_tags(self, *args, **kwargs):
        if hasattr(self.wrapped_model, "add_model_tags"):
            return self.wrapped_model.add_model_tags(*args, **kwargs)
        return None

    def get_aux_loss(self) -> Optional[torch.Tensor]:
        return self._last_aux_loss

    def clear_aux_loss(self) -> None:
        self._last_aux_loss = None

    def _position_ids(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if attention_mask is None:
            return torch.arange(
                input_ids.size(1), dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(input_ids.size(0), -1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 0)

    def _rotary_embeddings(self, hidden: torch.Tensor, position_ids: torch.Tensor):
        rotary = getattr(self.inner, "rotary_emb", None)
        if rotary is None:
            return None
        try:
            return rotary(hidden, position_ids)
        except TypeError:
            return None

    def _fallback_causal_mask(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        seq_len = hidden.size(1)
        dtype = hidden.dtype if hidden.dtype.is_floating_point else torch.float32
        min_value = torch.finfo(dtype).min
        causal = torch.full(
            (seq_len, seq_len), min_value, dtype=dtype, device=hidden.device
        )
        causal = torch.triu(causal, diagonal=1)
        causal = causal[None, None, :, :]
        if attention_mask is None:
            return causal
        padding = (1.0 - attention_mask[:, None, None, :].to(dtype=dtype)) * min_value
        return causal + padding.to(device=hidden.device)

    def _attention_mask_mapping(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ):
        try:
            from transformers.masking_utils import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )

            cache_position = position_ids[0]
            kwargs = {
                "config": self.config,
                "input_embeds": hidden,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            mapping: Dict[str, object] = {"full_attention": create_causal_mask(**kwargs)}
            if getattr(self.config, "use_sliding_window", False):
                mapping["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
            return mapping
        except Exception:
            return self._fallback_causal_mask(hidden, attention_mask)

    def _mask_for_layer(self, layer: nn.Module, mask_mapping):
        if isinstance(mask_mapping, dict):
            attention_type = getattr(layer, "attention_type", "full_attention")
            return mask_mapping.get(attention_type, mask_mapping.get("full_attention"))
        return mask_mapping

    def _run_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        attention_mask,
        position_ids: torch.Tensor,
        position_embeddings,
    ) -> torch.Tensor:
        layer_mask = self._mask_for_layer(layer, attention_mask)
        kwargs = {"attention_mask": layer_mask, "position_ids": position_ids}
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings

        try:
            output = layer(hidden, **kwargs)
        except TypeError:
            kwargs.pop("position_embeddings", None)
            try:
                output = layer(hidden, **kwargs)
            except TypeError:
                kwargs.pop("attention_mask", None)
                output = layer(hidden, **kwargs)
        return output[0] if isinstance(output, tuple) else output

    def _maybe_checkpoint(self, layer, hidden, attention_mask, position_ids, position_embeddings):
        if not (self.training and self.gradient_checkpointing):
            return self._run_layer(layer, hidden, attention_mask, position_ids, position_embeddings)

        def custom_forward(h):
            return self._run_layer(layer, h, attention_mask, position_ids, position_embeddings)

        return checkpoint(custom_forward, hidden, use_reentrant=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")
        if inputs_embeds is not None:
            hidden = inputs_embeds
            if input_ids is None:
                raise ValueError("LoPT currently requires input_ids when inputs_embeds is used.")
        else:
            hidden = self.inner.embed_tokens(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden.device)

        position_ids = self._position_ids(input_ids, attention_mask).to(hidden.device)
        position_embeddings = self._rotary_embeddings(hidden, position_ids)
        mask_mapping = self._attention_mask_mapping(hidden, attention_mask, position_ids)

        aux_losses: List[torch.Tensor] = []
        for block_idx, (start, end) in enumerate(self.block_ranges):
            block_input = hidden.detach()
            for layer in self.inner.layers[start:end]:
                hidden = self._maybe_checkpoint(
                    layer, hidden, mask_mapping, position_ids, position_embeddings
                )
            is_final = block_idx == len(self.block_ranges) - 1
            if not is_final:
                decoder = self.aux_decoders[block_idx]
                recon = decoder(hidden.to(dtype=next(decoder.parameters()).dtype))
                aux_losses.append(F.mse_loss(recon.float(), block_input.float()))
                if self.detach_boundaries:
                    hidden = hidden.detach()

        hidden = self.inner.norm(hidden)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_for_logits = hidden[:, -logits_to_keep:, :]
        elif torch.is_tensor(logits_to_keep):
            hidden_for_logits = hidden[:, logits_to_keep, :]
        else:
            hidden_for_logits = hidden
        logits = self.wrapped_model.lm_head(hidden_for_logits)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if aux_losses:
            self._last_aux_loss = torch.stack(aux_losses).mean()
        else:
            self._last_aux_loss = None

        return CausalLMOutputWithPast(loss=loss, logits=logits)


def build_lopt_optimizer_groups(
    model: nn.Module,
    learning_rate: float,
    lr_k1: Optional[float] = None,
    weight_decay: float = 0.0,
):
    """Return AdamW parameter groups for E2E or LoPT models.

    For LoPT, non-final blocks and auxiliary heads use `lr_k1`, while the final
    block uses `learning_rate`. For E2E models this returns one parameter group.
    """

    lr_k1 = learning_rate if lr_k1 is None else lr_k1
    if not isinstance(model, LoPTModelForCausalLM):
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]

    final_start = model.block_ranges[-1][0]
    inner = model.inner
    early_modules: List[nn.Module] = [inner.embed_tokens, model.aux_decoders]
    early_modules.extend(inner.layers[:final_start])
    final_modules: List[nn.Module] = list(inner.layers[final_start:])
    final_modules.extend([inner.norm, model.wrapped_model.lm_head])

    early_params = _module_parameters(early_modules)
    early_ids = {id(p) for p in early_params}
    final_params = [p for p in _module_parameters(final_modules) if id(p) not in early_ids]

    return [
        {"params": [p for p in early_params if p.requires_grad], "lr": lr_k1, "weight_decay": weight_decay},
        {"params": [p for p in final_params if p.requires_grad], "lr": learning_rate, "weight_decay": weight_decay},
    ]
