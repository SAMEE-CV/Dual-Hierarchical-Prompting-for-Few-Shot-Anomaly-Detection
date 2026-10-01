from pathlib import Path

import pytest

from dual_prompt.config import load_config


def test_smoke_config_is_valid() -> None:
    config = load_config(Path("configs/trainers/DualPrompt/smoke.yaml"))
    assert config.training.epochs == 1
    assert config.model.prompt_mode == "dual"
    assert config.model.prompt_layers == [0, 1, 2]
    assert config.model.text_prompt_layers == [0, 1, 2]
    assert sum(config.inference.layer_weights) == pytest.approx(1.0)


def test_layer_weight_count_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model:\n  prompt_mode: visual_only\n  prompt_layers: [0, 1]\n"
        "inference:\n  layer_weights: [1.0]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="layer_weight"):
        load_config(path)


def test_dual_depths_must_be_paired(tmp_path: Path) -> None:
    path = tmp_path / "bad_dual.yaml"
    path.write_text(
        "model:\n  prompt_layers: [0, 1]\n  text_prompt_layers: [0]\n"
        "inference:\n  layer_weights: [0.5, 0.5]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="one text prompt layer"):
        load_config(path)
