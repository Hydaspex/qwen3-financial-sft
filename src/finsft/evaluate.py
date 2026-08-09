"""Offline evaluation harness: generation + numeric-tolerance scoring."""

from __future__ import annotations

import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finsft.config import ExperimentConfig

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text: str) -> float | None:
    """First numeric token in the text, normalised (commas stripped)."""
    match = _NUM_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def numeric_match(prediction: str, gold: str, rel_tol: float = 1e-3) -> bool:
    """True if both parse to numbers within relative tolerance."""
    pred_num, gold_num = extract_number(prediction), extract_number(gold)
    if pred_num is None or gold_num is None:
        return False
    if gold_num == 0:
        return abs(pred_num) < rel_tol
    return abs(pred_num - gold_num) / abs(gold_num) <= rel_tol


def span_match(prediction: str, gold: str) -> bool:
    """Case-insensitive containment, for text/span answers."""
    return gold.strip().lower() in prediction.strip().lower()


def score(predictions: list[str], golds: list[str]) -> dict[str, float]:
    """Aggregate metrics: numeric EM (tolerant) and span match."""
    assert len(predictions) == len(golds)
    num_hits = sum(numeric_match(p, g) for p, g in zip(predictions, golds))
    span_hits = sum(span_match(p, g) for p, g in zip(predictions, golds))
    n = len(golds)
    return {
        "numeric_em": num_hits / n,
        "span_match": span_hits / n,
        "combined": (num_hits + span_hits) / (2 * n),
        "n": float(n),
    }


@torch.inference_mode()
def generate_answers(
    cfg: ExperimentConfig,
    records: list[dict],
    adapter_path: str | None = None,
    max_new_tokens: int = 64,
) -> list[str]:
    """Generate answers for chat-format records with base model + optional adapter."""
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    predictions: list[str] = []
    for rec in records:
        prompt = tokenizer.apply_chat_template(
            rec["messages"][:-1], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        predictions.append(
            tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        )
    return predictions
