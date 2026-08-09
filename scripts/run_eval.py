"""Run the offline eval harness and log metrics to MLflow.

Usage:
    python scripts/run_eval.py --config configs/sft_lora_qwen3_4b.yaml \
        --adapter outputs/qwen3-4b-tatqa-lora [--limit 200]
Omit --adapter to score the zero-shot base-model baseline.
"""

from __future__ import annotations

import argparse

import mlflow

from finsft.config import load_config
from finsft.data import read_jsonl
from finsft.evaluate import generate_answers, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    records = read_jsonl(cfg.data.val_path)
    if args.limit:
        records = records[: args.limit]

    predictions = generate_answers(cfg, records, adapter_path=args.adapter)
    golds = [rec["messages"][-1]["content"] for rec in records]
    metrics = score(predictions, golds)

    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    run_name = f"eval-{cfg.experiment_name}" + ("" if args.adapter else "-base")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"adapter": args.adapter or "base", "n_eval": len(records)})
        mlflow.log_metrics(metrics)

    print(metrics)


if __name__ == "__main__":
    main()
