#!/usr/bin/env python3
"""Supervised fine-tuning for E2E and LoPT local learning."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from lopt.data import SFTDataset, load_sft_dataset, resolve_model_name
from lopt.modeling_lopt import LoPTModelForCausalLM, build_lopt_optimizer_groups

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = logging.getLogger(__name__)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_main():
    return not is_dist() or dist.get_rank() == 0


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def build_model(args, device):
    model_name = resolve_model_name(args.model_name_or_path)
    dtype = torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32
    if args.method == "e2e":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
        )
    else:
        model = LoPTModelForCausalLM.from_pretrained_lopt(
            model_name,
            num_blocks=args.num_blocks,
            aux_loss=args.aux_loss,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model.to(device)


def save_model(model, tokenizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    if isinstance(raw, LoPTModelForCausalLM):
        raw.save_pretrained(str(output_dir))
    else:
        raw.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def parse_args():
    parser = argparse.ArgumentParser(description="SFT with E2E or LoPT localized training.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--method", choices=["e2e", "ll"], default="ll")
    parser.add_argument("--num_blocks", type=int, default=2, help="LoPT block count. Use 4 for k=4.")
    parser.add_argument("--aux_loss", choices=["recon"], default="recon")
    parser.add_argument("--lambda_aux", type=float, default=10.0)
    parser.add_argument("--dataset_preset", choices=["alpaca", "metamathqa", "tulu3"], default="metamathqa")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_k1", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=0, help="0 saves only at the end.")
    parser.add_argument("--output_dir", default="outputs/sft-lopt")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    local_rank, world_size, device = setup_distributed()

    model_name = resolve_model_name(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_dataset = load_sft_dataset(
        dataset_preset=args.dataset_preset,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_file=args.train_file,
        split=args.split,
        max_samples=args.max_samples,
    )
    train_dataset = SFTDataset(raw_dataset, tokenizer, max_length=args.max_seq_length)
    sampler = DistributedSampler(train_dataset, shuffle=True) if is_dist() else None
    loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if len(loader) == 0:
        raise RuntimeError(
            "The training dataloader is empty. Increase --max_samples or lower "
            "--per_device_train_batch_size."
        )

    model = build_model(args, device)
    optimizer = torch.optim.AdamW(
        build_lopt_optimizer_groups(
            model,
            learning_rate=args.learning_rate,
            lr_k1=args.lr_k1,
            weight_decay=args.weight_decay,
        )
    )

    if is_dist():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    updates_per_epoch = math.ceil(len(loader) / max(args.gradient_accumulation_steps, 1))
    total_updates = args.max_steps if args.max_steps > 0 else updates_per_epoch * args.num_train_epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    model.train()
    step = 0
    micro_step = 0
    start = time.time()
    pbar = tqdm(total=total_updates, disable=not is_main(), desc=f"SFT-{args.method}")

    for epoch in range(args.num_train_epochs if args.max_steps <= 0 else 10**9):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available() and (args.bf16 or args.fp16)):
                outputs = model(**batch)
                loss = outputs.loss
                raw = model.module if hasattr(model, "module") else model
                aux_loss = raw.get_aux_loss() if isinstance(raw, LoPTModelForCausalLM) else None
                if aux_loss is not None:
                    loss = loss + args.lambda_aux * aux_loss
            loss = loss / max(args.gradient_accumulation_steps, 1)
            loss.backward()
            micro_step += 1

            if micro_step % max(args.gradient_accumulation_steps, 1) != 0:
                continue

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main() and (step == 1 or step % args.logging_steps == 0):
                peak = torch.cuda.max_memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0.0
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item() * args.gradient_accumulation_steps:.4f}", peak=f"{peak:.1f}GB")
            elif is_main():
                pbar.update(1)

            if is_main() and args.save_steps > 0 and step % args.save_steps == 0:
                save_model(model, tokenizer, Path(args.output_dir) / f"checkpoint-{step}")
            if args.max_steps > 0 and step >= args.max_steps:
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break

    pbar.close()
    if is_main():
        elapsed = time.time() - start
        save_model(model, tokenizer, Path(args.output_dir))
        stats = {
            "method": args.method,
            "num_blocks": args.num_blocks if args.method == "ll" else None,
            "dataset_preset": args.dataset_preset,
            "model_name_or_path": model_name,
            "steps": step,
            "world_size": world_size,
            "elapsed_sec": round(elapsed, 2),
        }
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "training_args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args) | stats, f, indent=2)
        logger.info("Saved model to %s", args.output_dir)

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
