"""Build chat-format train/val JSONL from TAT-QA.

Usage:
    python scripts/prepare_data.py --config configs/sft_lora_qwen3_4b.yaml
"""

from __future__ import annotations

import argparse

from datasets import load_dataset

from finsft.config import load_config
from finsft.data import flatten_tatqa, train_val_split, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    dataset = load_dataset(cfg.data.dataset_name, split="train")
    records = flatten_tatqa(dataset, cfg.data.max_context_chars, cfg.data.max_samples)
    train, val = train_val_split(records, cfg.data.val_fraction, cfg.seed)

    write_jsonl(train, cfg.data.train_path)
    write_jsonl(val, cfg.data.val_path)
    print(f"Wrote {len(train)} train / {len(val)} val examples")


if __name__ == "__main__":
    main()
