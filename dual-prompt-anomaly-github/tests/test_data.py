from pathlib import Path

from dual_prompt.data import MVTecDataset, validate_mvtec_root
from dual_prompt.synthetic import create_tiny_mvtec


def test_tiny_mvtec_layout_and_masks(tmp_path: Path) -> None:
    root = create_tiny_mvtec(tmp_path / "mvtec")
    train = MVTecDataset(root, "bottle", "train", 32, shots=2, seed=3)
    test = MVTecDataset(root, "bottle", "test", 32)
    assert len(train) == 2
    assert len(test) == 4
    assert test[0]["image"].shape == (3, 32, 32)
    anomalous = [sample for sample in test if sample["label"].item() == 1]
    assert anomalous
    assert anomalous[0]["mask"].sum() > 0
    assert validate_mvtec_root(root)["bottle"] == 8


def test_few_shot_selection_is_deterministic(tmp_path: Path) -> None:
    root = create_tiny_mvtec(tmp_path / "mvtec")
    first = MVTecDataset(root, "bottle", "train", 32, shots=2, seed=11)
    second = MVTecDataset(root, "bottle", "train", 32, shots=2, seed=11)
    assert [sample[0] for sample in first.samples] == [sample[0] for sample in second.samples]
