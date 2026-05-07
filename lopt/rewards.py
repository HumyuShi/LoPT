"""Verifiable reward functions for TRL GRPO."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(completion)


def extract_gsm8k_answer(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def extract_boxed_answer(text: str) -> Optional[str]:
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    start = idx + len(marker)
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            if depth == 0:
                return text[start:pos].strip()
            depth -= 1
    return None


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    value = str(answer).strip().rstrip(".").replace("$", "").replace(",", "").replace(" ", "")
    try:
        return str(float(value))
    except ValueError:
        return value.lower()


def gsm8k_reward(completions, answer=None, **kwargs) -> List[float]:
    refs = answer or kwargs.get("answers") or kwargs.get("solution") or []
    rewards = []
    for completion, ref in zip(completions, refs):
        pred = normalize_answer(extract_gsm8k_answer(_completion_text(completion)))
        gold = normalize_answer(extract_gsm8k_answer(str(ref)))
        rewards.append(1.0 if pred is not None and pred == gold else 0.0)
    return rewards


def boxed_math_reward(completions, answer=None, **kwargs) -> List[float]:
    refs = answer or kwargs.get("answers") or kwargs.get("solution") or []
    rewards = []
    for completion, ref in zip(completions, refs):
        text = _completion_text(completion)
        pred = extract_boxed_answer(text) or extract_gsm8k_answer(text)
        gold = extract_boxed_answer(str(ref)) or extract_gsm8k_answer(str(ref)) or str(ref)
        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(gold)
        rewards.append(1.0 if pred_norm is not None and pred_norm == gold_norm else 0.0)
    return rewards
