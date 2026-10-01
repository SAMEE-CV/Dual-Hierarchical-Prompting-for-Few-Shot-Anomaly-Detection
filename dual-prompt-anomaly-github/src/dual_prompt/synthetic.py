"""Creation of a tiny MVTec-shaped fixture for smoke testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def create_tiny_mvtec(root: str | Path, size: int = 32, seed: int = 7) -> Path:
    """Create a deterministic bottle subset with normal and synthetic test images.

    This fixture validates the complete pipeline. It is not a benchmark and is
    never selected by production configurations.
    """
    root_path = Path(root)
    rng = np.random.default_rng(seed)
    category_root = root_path / "bottle"
    directories = (
        category_root / "train" / "good",
        category_root / "test" / "good",
        category_root / "test" / "crack",
        category_root / "ground_truth" / "crack",
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    yy, xx = np.mgrid[:size, :size]
    bottle = (
        (xx - size / 2) ** 2 / (size * 0.22) ** 2 + (yy - size / 2) ** 2 / (size * 0.42) ** 2
    ) <= 1

    def normal_image(index: int) -> np.ndarray:
        background = rng.normal(45, 3, (size, size, 3))
        color = np.array([80 + index * 2, 145, 190])
        background[bottle] = color + rng.normal(0, 4, (bottle.sum(), 3))
        return np.clip(background, 0, 255).astype(np.uint8)

    for index in range(4):
        Image.fromarray(normal_image(index)).save(
            category_root / "train" / "good" / f"{index:03d}.png"
        )
    for index in range(2):
        Image.fromarray(normal_image(index + 10)).save(
            category_root / "test" / "good" / f"{index:03d}.png"
        )
    for index in range(2):
        image = normal_image(index + 20)
        mask = np.zeros((size, size), dtype=np.uint8)
        start = size // 3 + index
        mask[start : start + 3, size // 3 : 2 * size // 3] = 255
        image[mask > 0] = np.array([235, 40, 35], dtype=np.uint8)
        Image.fromarray(image).save(category_root / "test" / "crack" / f"{index:03d}.png")
        Image.fromarray(mask).save(
            category_root / "ground_truth" / "crack" / f"{index:03d}_mask.png"
        )
    return root_path
