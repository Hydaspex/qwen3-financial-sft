"""Experiment configuration: pydantic-validated YAML loading."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    load_in_4bit: bool = True
    attn_implementation: str = "flash_attention_2"
    use_cache: bool = False


class DataConfig(BaseModel):
    dataset_name: str = "next-tat/TAT-QA"
    train_path: Path = Path("data/train.jsonl")
    val_path: Path = Path("data/val.jsonl")
    val_fraction: float = Field(default=0.05, gt=0.0, lt=0.5)
    max_context_chars: int = 6000
    max_samples: int | None = None


class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


class TrainerConfig(BaseModel):
    output_dir: Path = Path("outputs/qwen3-4b-tatqa-lora")
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_length: int = 2048
    logging_steps: int = 10
    save_strategy: str = "epoch"
    bf16: bool = True
    gradient_checkpointing: bool = True


class MLflowConfig(BaseModel):
    tracking_uri: str | None = None
    experiment: str = "/Shared/qwen3-financial-sft"


class ExperimentConfig(BaseModel):
    experiment_name: str = "qwen3-4b-tatqa-lora"
    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)
