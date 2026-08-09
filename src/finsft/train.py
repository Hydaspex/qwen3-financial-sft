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


def resolve_attn_implementation(requested: str) -> str:
    """Fall back to sdpa when FlashAttention-2 is unavailable (e.g. Colab T4)."""
    if requested == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("[finsft] flash-attn not installed; falling back to sdpa")
            return "sdpa"
    return requested


def resolve_precision(bf16_requested: bool) -> tuple[torch.dtype, bool, bool]:
    """Pick (dtype, bf16, fp16) based on hardware support.

    T4-class GPUs lack bf16; training there runs fp16 instead.
    """
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_bf16 = bf16_requested and bf16_ok
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    return dtype, use_bf16, use_fp16


def build_model_and_tokenizer(cfg: ExperimentConfig):
    dtype, _, _ = resolve_precision(cfg.trainer.bf16)
    quant = None
    if cfg.model.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        quantization_config=quant,
        torch_dtype=dtype,
        attn_implementation=resolve_attn_implementation(cfg.model.attn_implementation),
        device_map="auto",
    )
    model.config.use_cache = cfg.model.use_cache
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_trainer(cfg: ExperimentConfig) -> SFTTrainer:
    model, tokenizer = build_model_and_tokenizer(cfg)
    _, use_bf16, use_fp16 = resolve_precision(cfg.trainer.bf16)

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
        bf16=use_bf16,
        fp16=use_fp16,
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
