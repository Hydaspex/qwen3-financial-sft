"""Training entrypoint: LoRA/QLoRA SFT with TRL + MLflow tracking.

Usage:
    python -m finsft.train --config configs/sft_lora_qwen3_4b.yaml
"""

from __future__ import annotations

import argparse

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from finsft.config import ExperimentConfig, load_config
from finsft.data import read_jsonl


def build_model_and_tokenizer(cfg: ExperimentConfig):
    quant = None
    if cfg.model.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if cfg.trainer.bf16 else None,
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto",
    )
    model.config.use_cache = cfg.model.use_cache
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_trainer(cfg: ExperimentConfig) -> SFTTrainer:
    model, tokenizer = build_model_and_tokenizer(cfg)

    train_records = read_jsonl(cfg.data.train_path)
    val_records = read_jsonl(cfg.data.val_path)

    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(cfg.trainer.output_dir),
        num_train_epochs=cfg.trainer.num_train_epochs,
        per_device_train_batch_size=cfg.trainer.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        learning_rate=cfg.trainer.learning_rate,
        lr_scheduler_type=cfg.trainer.lr_scheduler_type,
        warmup_ratio=cfg.trainer.warmup_ratio,
        max_length=cfg.trainer.max_length,
        logging_steps=cfg.trainer.logging_steps,
        save_strategy=cfg.trainer.save_strategy,
        bf16=cfg.trainer.bf16,
        gradient_checkpointing=cfg.trainer.gradient_checkpointing,
        completion_only_loss=True,  # loss on assistant turn only
        seed=cfg.seed,
        report_to=[],
    )

    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(val_records),
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    with mlflow.start_run(run_name=cfg.experiment_name):
        mlflow.log_params(cfg.model_dump(mode="json", exclude={"mlflow"}))
        trainer = build_trainer(cfg)
        metrics = trainer.train().metrics
        mlflow.log_metrics({f"train_{k}": v for k, v in metrics.items()})
        trainer.save_model(str(cfg.trainer.output_dir))
        mlflow.log_artifacts(str(cfg.trainer.output_dir), artifact_path="adapter")
        print(f"Adapter saved to {cfg.trainer.output_dir}")


if __name__ == "__main__":
    main()
