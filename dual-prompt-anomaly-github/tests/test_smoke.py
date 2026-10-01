from pathlib import Path

from dual_prompt.config import load_config
from dual_prompt.engine import run_training
from dual_prompt.synthetic import create_tiny_mvtec


def test_complete_one_epoch_smoke_run(tmp_path: Path) -> None:
    config = load_config("configs/trainers/DualPrompt/smoke.yaml")
    config.data.root = str(create_tiny_mvtec(tmp_path / "data"))
    config.output_dir = str(tmp_path / "outputs")
    artifacts = run_training(config)
    assert artifacts.checkpoint.is_file()
    assert len(artifacts.history) == 1
    assert set(artifacts.metrics) == {
        "image_auroc",
        "image_aupr",
        "pixel_auroc",
        "pixel_aupr",
        "pixel_dice",
    }
    assert (tmp_path / "outputs" / "bottle" / "predictions.csv").is_file()
