#!/usr/bin/env python3
"""Download/cache training datasets and optionally export local parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    "alpaca": ("tatsu-lab/alpaca", None, "train"),
    "metamathqa": ("meta-math/MetaMathQA", None, "train"),
    "tulu3": ("allenai/tulu-3-sft-mixture", None, "train"),
    "gsm8k": ("openai/gsm8k", "main", "train"),
    "numinamath": ("AI-MO/NuminaMath-CoT", None, "train"),
}


def load_one(name: str, max_samples: int | None):
    path, config, split = DATASETS[name]
    kwargs = {"path": path, "split": split}
    if config:
        kwargs["name"] = config
    data = load_dataset(**kwargs)
    if max_samples and max_samples > 0:
        data = data.shuffle(seed=42).select(range(min(max_samples, len(data))))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--write_parquet", action="store_true")
    args = parser.parse_args()

    selected = list(DATASETS) if args.dataset == "all" else [args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in selected:
        data = load_one(name, args.max_samples)
        print(f"{name}: {len(data)} rows")
        if args.write_parquet:
            out = output_dir / f"{name}.parquet"
            data.to_parquet(str(out))
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
