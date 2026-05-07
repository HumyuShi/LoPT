"""Dataset formatting utilities for SFT and GRPO."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from datasets import load_dataset


SFT_PRESETS: Dict[str, Dict[str, Optional[str]]] = {
    "alpaca": {"path": "tatsu-lab/alpaca", "name": None, "split": "train"},
    "metamathqa": {"path": "meta-math/MetaMathQA", "name": None, "split": "train"},
    "tulu3": {"path": "allenai/tulu-3-sft-mixture", "name": None, "split": "train"},
}

GRPO_PRESETS: Dict[str, Dict[str, Optional[str]]] = {
    "gsm8k": {"path": "openai/gsm8k", "name": "main", "split": "train"},
    "numinamath": {"path": "AI-MO/NuminaMath-CoT", "name": None, "split": "train"},
}

MODEL_ALIASES = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def resolve_model_name(name_or_path: str) -> str:
    return MODEL_ALIASES.get(name_or_path, name_or_path)


def _maybe_parse_messages(messages):
    if isinstance(messages, str):
        try:
            return json.loads(messages)
        except json.JSONDecodeError:
            return None
    return messages


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if "qwen3" in getattr(tokenizer, "name_or_path", "").lower():
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            parts.append(f"{role}: {msg.get('content', '')}")
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


def format_sft_prompt_response(item: Dict, tokenizer=None):
    messages = _maybe_parse_messages(item.get("messages"))
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "assistant":
            prompt_messages = messages[:-1]
            response = last.get("content", "")
            if tokenizer is not None:
                prompt = apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
            else:
                prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt_messages)
                prompt += "\nassistant:"
            return prompt, response

    if "instruction" in item and "output" in item:
        instruction = item.get("instruction") or ""
        extra_input = item.get("input") or ""
        prompt = f"Instruction: {instruction}\n"
        if extra_input:
            prompt += f"Input: {extra_input}\n"
        prompt += "Response:"
        return prompt, item.get("output") or ""

    if "query" in item and "response" in item:
        return f"Question: {item.get('query')}\nAnswer:", item.get("response") or ""

    if "question" in item and "answer" in item:
        return f"Question: {item.get('question')}\nAnswer:", item.get("answer") or ""

    if "problem" in item and ("solution" in item or "answer" in item):
        return f"Problem: {item.get('problem')}\nSolution:", item.get("solution") or item.get("answer") or ""

    if "prompt" in item and ("completion" in item or "response" in item):
        return str(item.get("prompt")), str(item.get("completion") or item.get("response") or "")

    raise ValueError(f"Cannot infer SFT fields from columns: {list(item.keys())}")


def load_sft_dataset(
    dataset_preset: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    train_file: Optional[str] = None,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> HFDataset:
    if train_file:
        data = load_dataset("parquet" if train_file.endswith(".parquet") else "json", data_files=train_file)["train"]
    else:
        if dataset_preset:
            if dataset_preset not in SFT_PRESETS:
                raise ValueError(f"Unknown SFT preset: {dataset_preset}")
            spec = SFT_PRESETS[dataset_preset]
            dataset_name = spec["path"]
            dataset_config = spec["name"]
            split = split or spec["split"]
        if not dataset_name:
            raise ValueError("Set --dataset_preset, --dataset_name, or --train_file.")
        kwargs = {"path": dataset_name, "split": split or "train"}
        if dataset_config:
            kwargs["name"] = dataset_config
        data = load_dataset(**kwargs)
    if max_samples and max_samples > 0:
        data = data.shuffle(seed=42).select(range(min(max_samples, len(data))))
    return data


class SFTDataset(Dataset):
    def __init__(self, dataset: HFDataset, tokenizer, max_length: int = 1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt, response = format_sft_prompt_response(item, self.tokenizer)
        eos = self.tokenizer.eos_token or ""
        prompt_text = prompt.rstrip() + "\n"
        full_text = prompt_text + str(response).strip() + eos

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        prompt_ids = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        labels[: min(prompt_ids.numel(), labels.numel())] = -100
        if torch.all(labels == -100):
            valid = torch.nonzero(attention_mask, as_tuple=False)
            if valid.numel() > 0:
                labels[valid[-1].item()] = input_ids[valid[-1].item()]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _grpo_prompt(task: str, prompt: str) -> str:
    if task == "gsm8k":
        system = "Solve the math problem step by step. Put the final answer after ####."
        return f"{system}\n\nProblem: {prompt}\n\nSolution:"
    if task == "numinamath":
        system = "Solve the math problem step by step. Put the final answer in \\boxed{}."
        return f"{system}\n\nProblem: {prompt}\n\nSolution:"
    return str(prompt)


def load_grpo_dataset(
    task: str,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    train_file: Optional[str] = None,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> HFDataset:
    if train_file:
        raw = load_dataset("parquet" if train_file.endswith(".parquet") else "json", data_files=train_file)["train"]
    else:
        if task in GRPO_PRESETS and not dataset_name:
            spec = GRPO_PRESETS[task]
            dataset_name = spec["path"]
            dataset_config = spec["name"]
            split = split or spec["split"]
        if not dataset_name:
            raise ValueError("Set --task preset, --dataset_name, or --train_file.")
        kwargs = {"path": dataset_name, "split": split or "train"}
        if dataset_config:
            kwargs["name"] = dataset_config
        raw = load_dataset(**kwargs)

    def convert(item):
        if task == "gsm8k":
            prompt = item.get("question") or item.get("prompt") or item.get("problem")
            answer = item.get("answer") or item.get("solution")
        elif task == "numinamath":
            prompt = item.get("problem") or item.get("question") or item.get("prompt")
            answer = item.get("solution") or item.get("answer")
        else:
            prompt = item.get("prompt") or item.get("question") or item.get("problem")
            answer = item.get("answer") or item.get("solution")
        return {"prompt": _grpo_prompt(task, prompt), "answer": str(answer or "")}

    data = raw.map(convert, remove_columns=raw.column_names)
    if max_samples and max_samples > 0:
        data = data.shuffle(seed=42).select(range(min(max_samples, len(data))))
    return data
