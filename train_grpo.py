#!/usr/bin/env python3
"""TRL on-policy GRPO for E2E and LoPT local learning."""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from lopt.data import load_grpo_dataset, resolve_model_name
from lopt.modeling_lopt import LoPTModelForCausalLM, build_lopt_optimizer_groups
from lopt.rewards import boxed_math_reward, gsm8k_reward

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = logging.getLogger(__name__)


def config_kwargs_for_current_trl(kwargs):
    signature = inspect.signature(GRPOConfig.__init__)
    return {k: v for k, v in kwargs.items() if k in signature.parameters}


def import_trl_grpo():
    try:
        from trl import GRPOConfig as _GRPOConfig
        from trl import GRPOTrainer as _GRPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "Failed to import TRL GRPO. Install a compatible stack, for example "
            "`pip install -r requirements.txt`. If you use a newer TRL release, "
            "make sure your PyTorch version provides the FSDP APIs required by TRL."
        ) from exc
    return _GRPOConfig, _GRPOTrainer


def build_lopt_grpo_trainer_class(base_trainer_cls):
    class LoPTGRPOTrainer(base_trainer_cls):
        """GRPOTrainer that adds LoPT's local auxiliary loss when present."""

        def __init__(self, *args, lopt_lambda_aux: float = 10.0, **kwargs):
            self.lopt_lambda_aux = lopt_lambda_aux
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            kwargs = {"return_outputs": return_outputs}
            signature = inspect.signature(super().compute_loss)
            if "num_items_in_batch" in signature.parameters:
                kwargs["num_items_in_batch"] = num_items_in_batch
            result = super().compute_loss(model, inputs, **kwargs)

            if return_outputs:
                loss, outputs = result
            else:
                loss, outputs = result, None

            raw = model.module if hasattr(model, "module") else model
            aux_loss = raw.get_aux_loss() if isinstance(raw, LoPTModelForCausalLM) else None
            if aux_loss is not None:
                loss = loss + self.lopt_lambda_aux * aux_loss

            return (loss, outputs) if return_outputs else loss

    return LoPTGRPOTrainer


def build_model(args):
    model_name = resolve_model_name(args.model_name_or_path)
    dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32
    if args.method == "e2e":
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        )
    return LoPTModelForCausalLM.from_pretrained_lopt(
        model_name,
        num_blocks=args.num_blocks,
        torch_dtype=dtype,
        trust_remote_code=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="TRL GRPO with E2E or LoPT local learning.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--method", choices=["e2e", "ll"], default="ll")
    parser.add_argument("--num_blocks", type=int, default=2, help="LoPT block count. Use 4 for k=4.")
    parser.add_argument("--task", choices=["gsm8k", "numinamath"], default="gsm8k")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="outputs/grpo-lopt")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_k1", type=float, default=None)
    parser.add_argument("--lambda_aux", type=float, default=10.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--loss_type", default="grpo")
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--steps_per_generation", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--use_vllm", action="store_true", help="Use TRL/vLLM rollout if installed and configured.")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    model_name = resolve_model_name(args.model_name_or_path)
    global GRPOConfig, GRPOTrainer
    GRPOConfig, GRPOTrainer = import_trl_grpo()
    LoPTGRPOTrainer = build_lopt_grpo_trainer_class(GRPOTrainer)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_dataset = load_grpo_dataset(
        task=args.task,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_file=args.train_file,
        split=args.split,
        max_samples=args.max_samples,
    )
    model = build_model(args)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    reward_func = gsm8k_reward if args.task == "gsm8k" else boxed_math_reward
    training_args = GRPOConfig(
        **config_kwargs_for_current_trl(
            {
                "output_dir": args.output_dir,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
                "num_generations": args.num_generations,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": args.max_completion_length,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "beta": args.beta,
                "loss_type": args.loss_type,
                "num_iterations": args.num_iterations,
                "steps_per_generation": args.steps_per_generation,
                "bf16": args.bf16,
                "fp16": args.fp16,
                "gradient_checkpointing": args.gradient_checkpointing,
                "logging_steps": args.logging_steps,
                "save_steps": args.save_steps,
                "save_strategy": "steps",
                "report_to": "none",
                "remove_unused_columns": False,
                "ddp_find_unused_parameters": False,
                "use_vllm": args.use_vllm,
            }
        )
    )

    optimizer = torch.optim.AdamW(
        build_lopt_optimizer_groups(
            model,
            learning_rate=args.learning_rate,
            lr_k1=args.lr_k1,
            weight_decay=args.weight_decay,
        )
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "reward_funcs": reward_func,
        "optimizers": (optimizer, None),
        "lopt_lambda_aux": args.lambda_aux,
    }
    trainer_signature = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = LoPTGRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        logger.info("Saved model to %s", Path(args.output_dir).resolve())

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
