"""Dataset builders: TAT-QA -> chat-format examples.

Formatting functions are pure and unit-tested; I/O lives in scripts/prepare_data.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are a financial analyst. Answer the question using only the provided "
    "report context. If the answer is numeric, respond with the number only."
)


def linearize_table(table: list[list[str]]) -> str:
    """Linearize a TAT-QA table (list of rows) into pipe-delimited text."""
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def build_context(example: dict[str, Any], max_chars: int) -> str:
    """Concatenate paragraphs + linearized table into a single context string."""
    parts: list[str] = []
    for p in example.get("paragraphs", []):
        text = p.get("text", "") if isinstance(p, dict) else str(p)
        if text:
            parts.append(text)
    table = example.get("table", {})
    rows = table.get("table", []) if isinstance(table, dict) else table
    if rows:
        parts.append(linearize_table(rows))
    context = "\n\n".join(parts)
    return context[:max_chars]


def normalise_answer(answer: Any) -> str:
    """TAT-QA answers can be a scalar or a list (multi-span). Join lists."""
    if isinstance(answer, list):
        return ", ".join(str(a) for a in answer)
    return str(answer)


def format_example(example: dict[str, Any], question: dict[str, Any], max_chars: int) -> dict | None:
    """Convert one TAT-QA question into a chat-format example.

    Returns None for unanswerable/scale-only questions (e.g. percent-sign slots).
    """
    answer = question.get("answer")
    if answer is None or answer == "":
        return None
    context = build_context(example, max_chars)
    if not context:
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nQuestion: {question['question']}"},
            {"role": "assistant", "content": normalise_answer(answer)},
        ]
    }


def flatten_tatqa(dataset, max_context_chars: int, max_samples: int | None = None) -> list[dict]:
    """Flatten TAT-QA (one example = many questions) into chat-format records."""
    records: list[dict] = []
    for example in dataset:
        for question in example.get("questions", []):
            rec = format_example(example, question, max_context_chars)
            if rec is not None:
                records.append(rec)
            if max_samples is not None and len(records) >= max_samples:
                return records
    return records


def train_val_split(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle split."""
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val = max(1, int(len(records) * val_fraction))
    val_idx = set(idx[:n_val])
    train = [r for i, r in enumerate(records) if i not in val_idx]
    val = [r for i, r in enumerate(records) if i in val_idx]
    return train, val


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in records)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
