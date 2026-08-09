# qwen3-financial-sft

Portfolio-grade **supervised fine-tuning (SFT)** of an open-weight LLM (**Qwen3-4B-Instruct**) on **financial reasoning** tasks (TAT-QA / FinQA-style numeric QA over financial reports), using **TRL + PEFT (LoRA/QLoRA)** with **MLflow** experiment tracking and an **offline evaluation harness**.

## Why this project

Base instruct models are weak at grounded numeric reasoning over hybrid table+text financial contexts. This repo demonstrates an end-to-end, reproducible workflow:

1. **Data prep** — TAT-QA → chat-format JSONL with deterministic train/val splits.
2. **Training** — LoRA/QLoRA SFT via TRL's `SFTTrainer`, fully config-driven (YAML + pydantic validation).
3. **Evaluation** — numeric-tolerance exact match + span match on a held-out split, logged to MLflow alongside training curves.
4. **Export** — merge adapter → standalone model artifact ready for serving.

## Architecture

```
configs/                  # Experiment configs (validated by pydantic)
scripts/
  prepare_data.py         # TAT-QA -> data/{train,val}.jsonl (chat format)
  run_eval.py             # Offline eval harness -> MLflow metrics
src/finsft/
  config.py               # pydantic config models + YAML loader
  data.py                 # dataset builders, example formatting (pure functions)
  train.py                # SFTTrainer entrypoint (LoRA/QLoRA, MLflow autolog)
  evaluate.py             # generation + numeric/span scoring
tests/                    # unit tests (no GPU required)
```

## Quickstart (local GPU, ~16 GB VRAM with QLoRA)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

python scripts/prepare_data.py --config configs/sft_lora_qwen3_4b.yaml
python -m finsft.train --config configs/sft_lora_qwen3_4b.yaml
python scripts/run_eval.py --config configs/sft_lora_qwen3_4b.yaml \
    --adapter outputs/qwen3-4b-tatqa-lora
```

## Databricks / MLflow

Set `MLFLOW_TRACKING_URI=databricks` (and `DATABRICKS_HOST` / `DATABRICKS_TOKEN`) to log runs to a Databricks workspace experiment instead of local `./mlruns`. The pipeline is cluster-agnostic: run `train.py` as a job task on a single-node GPU cluster (A10G/A100), artifacts land in the MLflow run.

## Results

| Model | EM (numeric, tol=1e-3) | Span match | Notes |
|---|---|---|---|
| Qwen3-4B-Instruct (base) | _run eval_ | _run eval_ | zero-shot baseline |
| + LoRA SFT (r=16, 3 epochs) | _run eval_ | _run eval_ | config: `sft_lora_qwen3_4b.yaml` |

## Key design choices

- **LoRA over full FT**: 4B model trains on a single 16–24 GB GPU; adapter weights (~40 MB) are cheap to version in MLflow.
- **Completion-only loss** (TRL): loss is computed on the assistant turn only — prevents the model from learning to generate questions.
- **Numeric-tolerance EM**: financial answers are floats; exact string match under-credits correct arithmetic. We parse the first numeric token and compare with relative tolerance, falling back to span match for text answers.
- **Config-driven**: every hyperparameter lives in one YAML, validated at load time — same pattern as my production Databricks pipelines.

## License

MIT. TAT-QA dataset is CC-BY-4.0 (next-tat/TAT-QA on Hugging Face).
