from pathlib import Path

import pytest
from pydantic import ValidationError

from finsft.config import ExperimentConfig, load_config


def test_load_repo_config():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "sft_lora_qwen3_4b.yaml")
    assert cfg.model.name_or_path.startswith("Qwen/")
    assert cfg.lora.r == 16
    assert cfg.trainer.gradient_accumulation_steps == 8
    assert "q_proj" in cfg.lora.target_modules


def test_defaults_form_valid_config():
    cfg = ExperimentConfig()
    assert 0.0 < cfg.data.val_fraction < 0.5


def test_invalid_val_fraction_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig(data={"val_fraction": 0.9})
