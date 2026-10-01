"""Typed experiment configuration and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    root: str = "data/mvtec_anomaly_detection"
    category: str = "bottle"
    shots: int = 4
    image_size: int = 224
    num_workers: int = 4


@dataclass(slots=True)
class ModelConfig:
    backbone: str = "ViT-B-16-quickgelu"
    pretrained: str = "openai"
    prompt_mode: str = "dual"
    prompt_layers: list[int] = field(default_factory=lambda: [0, 5, 11])
    prompt_length: int = 4
    text_prompt_layers: list[int] = field(default_factory=lambda: [0, 5, 11])
    text_prompt_length: int = 4
    conditioner: str = "linear"


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 3.5e-3
    momentum: float = 0.9
    weight_decay: float = 0.0
    regularization: float = 1e-4


@dataclass(slots=True)
class MemoryConfig:
    max_size: int = 1024
    sampling: str = "random"
    neighbors: int = 1
    chunk_size: int = 2048


@dataclass(slots=True)
class InferenceConfig:
    layer_weights: list[float] = field(default_factory=lambda: [0.333333, 0.333334, 0.333333])
    semantic_weight: float = 0.5
    top_pixels: int = 100
    threshold: float = 0.5


@dataclass(slots=True)
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs/mvtec"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def validate(self) -> None:
        if self.data.shots < 1:
            raise ValueError("data.shots must be positive")
        if self.data.image_size < 16:
            raise ValueError("data.image_size must be at least 16")
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.model.prompt_length < 1 or not self.model.prompt_layers:
            raise ValueError("prompt_length and prompt_layers must be positive/non-empty")
        if self.model.prompt_mode not in {"dual", "visual_only"}:
            raise ValueError("model.prompt_mode must be 'dual' or 'visual_only'")
        if len(set(self.model.prompt_layers)) != len(self.model.prompt_layers):
            raise ValueError("model.prompt_layers must not contain duplicates")
        if self.model.prompt_mode == "dual":
            if self.model.text_prompt_length < 1 or not self.model.text_prompt_layers:
                raise ValueError(
                    "text_prompt_length and text_prompt_layers must be positive/non-empty in dual mode"
                )
            if len(set(self.model.text_prompt_layers)) != len(self.model.text_prompt_layers):
                raise ValueError("model.text_prompt_layers must not contain duplicates")
            if len(self.model.text_prompt_layers) != len(self.model.prompt_layers):
                raise ValueError("dual mode requires one text prompt layer per visual prompt layer")
            if self.model.text_prompt_length != self.model.prompt_length:
                raise ValueError("dual mode requires equal text and visual prompt lengths")
        if self.memory.sampling not in {"random", "none"}:
            raise ValueError("memory.sampling must be 'random' or 'none'")
        if self.memory.max_size < 1 or self.memory.neighbors < 1:
            raise ValueError("memory sizes must be positive")
        if len(self.inference.layer_weights) != len(self.model.prompt_layers):
            raise ValueError("one inference.layer_weight is required per prompt layer")
        if any(weight < 0 for weight in self.inference.layer_weights):
            raise ValueError("inference.layer_weights must be non-negative")
        weight_sum = sum(self.inference.layer_weights)
        if abs(weight_sum - 1.0) > 1e-5:
            raise ValueError("inference.layer_weights must sum to one")
        if not 0.0 <= self.inference.semantic_weight <= 1.0:
            raise ValueError("inference.semantic_weight must be in [0, 1]")
        if not 0.0 <= self.inference.threshold <= 1.0:
            raise ValueError("inference.threshold must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_config(raw: dict[str, Any]) -> ExperimentConfig:
    known_top = {"seed", "device", "output_dir", "data", "model", "training", "memory", "inference"}
    unknown = set(raw) - known_top
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    config = ExperimentConfig(
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        output_dir=raw.get("output_dir", "outputs/mvtec"),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        memory=MemoryConfig(**raw.get("memory", {})),
        inference=InferenceConfig(**raw.get("inference", {})),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("The configuration root must be a mapping")
    return _construct_config(raw)
