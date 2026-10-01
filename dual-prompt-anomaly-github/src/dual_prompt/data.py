"""MVTec AD dataset loading and deterministic few-shot sampling."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _image_files(directory: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions)


class MVTecDataset(Dataset[dict[str, Any]]):
    """Canonical MVTec AD directory reader.

    Expected layout: ``root/category/{train,test,ground_truth}``.
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str,
        image_size: int,
        shots: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size
        category_root = self.root / category
        if not category_root.is_dir():
            raise FileNotFoundError(
                f"MVTec category directory not found: {category_root}. "
                "See README.md for the expected dataset layout."
            )
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        samples: list[tuple[Path, Path | None, int, str]] = []
        split_root = category_root / split
        if split == "train":
            good_root = split_root / "good"
            if not good_root.is_dir():
                raise FileNotFoundError(f"Missing normal training directory: {good_root}")
            paths = _image_files(good_root)
            if shots is not None:
                if shots > len(paths):
                    raise ValueError(
                        f"Requested {shots} shots, but {category} has only {len(paths)} normal images"
                    )
                paths = sorted(random.Random(seed).sample(paths, shots))
            samples.extend((path, None, 0, "good") for path in paths)
        else:
            if not split_root.is_dir():
                raise FileNotFoundError(f"Missing test directory: {split_root}")
            for defect_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
                defect_type = defect_root.name
                for path in _image_files(defect_root):
                    if defect_type == "good":
                        mask_path, label = None, 0
                    else:
                        mask_path = (
                            category_root / "ground_truth" / defect_type / f"{path.stem}_mask.png"
                        )
                        if not mask_path.is_file():
                            raise FileNotFoundError(f"Missing ground-truth mask: {mask_path}")
                        label = 1
                    samples.append((path, mask_path, label, defect_type))
        if not samples:
            raise RuntimeError(f"No images found for {category}/{split}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, mask_path, label, defect_type = self.samples[index]
        with Image.open(path) as image_file:
            image = image_file.convert("RGB")
        original_size = (image.height, image.width)
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        image_tensor = TF.normalize(TF.to_tensor(image), CLIP_MEAN, CLIP_STD)

        if mask_path is None:
            mask = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        else:
            with Image.open(mask_path) as mask_file:
                mask_image = mask_file.convert("L")
            mask_image = TF.resize(
                mask_image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )
            mask = (TF.to_tensor(mask_image) > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
            "defect_type": defect_type,
            "original_size": torch.tensor(original_size, dtype=torch.long),
        }


def make_loaders(
    root: str | Path,
    category: str,
    image_size: int,
    shots: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]]]:
    train_dataset = MVTecDataset(root, category, "train", image_size, shots, seed)
    test_dataset = MVTecDataset(root, category, "test", image_size, None, seed)
    generator = torch.Generator().manual_seed(seed)
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        generator=generator,
        **common,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common)
    return train_loader, test_loader


def validate_mvtec_root(root: str | Path) -> dict[str, int]:
    """Return image counts for available categories, raising on malformed data."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")
    result: dict[str, int] = {}
    for category in MVTEC_CATEGORIES:
        category_root = root_path / category
        if not category_root.is_dir():
            continue
        train = MVTecDataset(root_path, category, "train", 32)
        test = MVTecDataset(root_path, category, "test", 32)
        result[category] = len(train) + len(test)
    if not result:
        raise RuntimeError(f"No valid MVTec AD categories found below {root_path}")
    return result
