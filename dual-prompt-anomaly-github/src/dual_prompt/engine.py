"""Training, evaluation, checkpointing, and experiment orchestration."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .backbones import create_backbone
from .config import ExperimentConfig
from .data import make_loaders
from .memory import FeatureMemory, anomaly_map, build_feature_memory, image_anomaly_score
from .metrics import compute_metrics
from .model import DualPromptModel
from .prompts import prompts_for_category
from .utils import resolve_device, seed_everything, trainable_parameter_count, write_json


@dataclass(slots=True)
class ExperimentArtifacts:
    checkpoint: Path
    metrics: dict[str, float]
    history: list[dict[str, float]]


def build_model(config: ExperimentConfig, category: str, device: torch.device) -> DualPromptModel:
    backbone = create_backbone(
        config.model.backbone, config.model.pretrained, config.data.image_size, device
    )
    model = DualPromptModel(
        backbone=backbone,
        prompt_set=prompts_for_category(category),
        prompt_layers=config.model.prompt_layers,
        prompt_length=config.model.prompt_length,
        conditioner=config.model.conditioner,
        prompt_mode=config.model.prompt_mode,
        text_prompt_layers=config.model.text_prompt_layers,
        text_prompt_length=config.model.text_prompt_length,
    )
    return model.to(device)


def train_prompts(
    model: DualPromptModel,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable prompt parameters were found")
    optimizer = SGD(
        parameters,
        lr=config.training.learning_rate,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    history: list[dict[str, float]] = []
    for epoch in range(config.training.epochs):
        model.train()
        total_loss = 0.0
        total_text_loss = 0.0
        sample_count = 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config.training.epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            target = torch.zeros(images.shape[0], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images)
            text_loss = nn.functional.cross_entropy(logits, target)
            regularization = config.training.regularization * model.prompt_regularization()
            loss = text_loss + regularization
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss encountered: {loss.item()}")
            loss.backward()
            optimizer.step()
            batch_size = images.shape[0]
            sample_count += batch_size
            total_loss += float(loss.detach()) * batch_size
            total_text_loss += float(text_loss.detach()) * batch_size
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": total_loss / sample_count,
                "text_loss": total_text_loss / sample_count,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    return history


@torch.no_grad()
def evaluate(
    model: DualPromptModel,
    memory: FeatureMemory,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    maps: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    map_directory = output_dir / "anomaly_maps" if output_dir is not None else None
    if map_directory is not None:
        map_directory.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="evaluate", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        logits, patch_features = model(images)
        pixel_map = anomaly_map(
            patch_features,
            memory,
            (config.data.image_size, config.data.image_size),
            config.inference.layer_weights,
            config.memory.neighbors,
            config.memory.chunk_size,
        )
        combined, semantic, spatial = image_anomaly_score(
            logits,
            pixel_map,
            config.inference.semantic_weight,
            config.inference.top_pixels,
        )
        batch_labels = batch["label"].cpu().numpy()
        batch_masks = batch["mask"].cpu().numpy()[:, 0]
        batch_scores = combined.cpu().numpy()
        batch_maps = pixel_map.cpu().numpy()[:, 0]
        labels.append(batch_labels)
        masks.append(batch_masks)
        scores.append(batch_scores)
        maps.append(batch_maps)
        for index, path in enumerate(batch["path"]):
            record = {
                "path": path,
                "defect_type": batch["defect_type"][index],
                "label": int(batch_labels[index]),
                "anomaly_score": float(batch_scores[index]),
                "semantic_score": float(semantic[index].cpu()),
                "spatial_score": float(spatial[index].cpu()),
            }
            records.append(record)
            if map_directory is not None:
                source = Path(path)
                defect = str(batch["defect_type"][index])
                target = map_directory / f"{defect}_{source.stem}.png"
                Image.fromarray(np.uint8(np.clip(batch_maps[index], 0, 1) * 255), mode="L").save(
                    target
                )

    image_labels = np.concatenate(labels)
    image_scores = np.concatenate(scores)
    mask_array = np.concatenate(masks)
    map_array = np.concatenate(maps)
    metrics = compute_metrics(
        image_labels, image_scores, mask_array, map_array, config.inference.threshold
    )
    return metrics, records


def _save_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save_checkpoint(
    path: Path,
    model: DualPromptModel,
    memory: FeatureMemory,
    config: ExperimentConfig,
    category: str,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt_set = prompts_for_category(category)
    torch.save(
        {
            "format_version": 1,
            "category": category,
            "config": config.to_dict(),
            "model": model.trainable_state_dict(),
            "memory": memory.state_dict(),
            "history": history,
            "prompts": {
                "normal": list(prompt_set.normal),
                "defective": list(prompt_set.defective),
            },
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.4 compatibility
        return torch.load(path, map_location=device)


def run_training(config: ExperimentConfig, category: str | None = None) -> ExperimentArtifacts:
    category = category or config.data.category
    seed_everything(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir) / category
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(
        config.data.root,
        category,
        config.data.image_size,
        config.data.shots,
        config.training.batch_size,
        config.data.num_workers,
        config.seed,
    )
    model = build_model(config, category, device)
    run_metadata = {
        "category": category,
        "device": str(device),
        "trainable_parameters": trainable_parameter_count(model),
        "support_images": len(train_loader.dataset),
        "test_images": len(test_loader.dataset),
        "implementation_choices": {
            "prompt_mode": config.model.prompt_mode,
            "prompt_layers_are_zero_indexed": True,
            "text_prompt_layers_are_zero_indexed": True,
            "text_visual_coupling": "token-wise linear projection",
            "memory_subsampling": config.memory.sampling,
            "memory_neighbor_count": config.memory.neighbors,
            "top_pixel_count": config.inference.top_pixels,
            "semantic_fusion_weight": config.inference.semantic_weight,
        },
    }
    write_json(output_dir / "run_metadata.json", run_metadata)
    history = train_prompts(model, train_loader, config, device)
    memory = build_feature_memory(
        model,
        train_loader,
        device,
        config.memory.max_size,
        config.memory.sampling,
        config.seed,
    )
    metrics, records = evaluate(model, memory, test_loader, config, device, output_dir)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Evaluation produced undefined metrics: {metrics}")
    checkpoint = output_dir / "checkpoint.pt"
    save_checkpoint(checkpoint, model, memory, config, category, history)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "history.json", {"epochs": history})
    _save_predictions(output_dir / "predictions.csv", records)
    return ExperimentArtifacts(checkpoint=checkpoint, metrics=metrics, history=history)


def run_evaluation(
    config: ExperimentConfig, checkpoint_path: str | Path
) -> tuple[str, dict[str, float], list[dict[str, Any]]]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    category = str(checkpoint["category"])
    _, test_loader = make_loaders(
        config.data.root,
        category,
        config.data.image_size,
        config.data.shots,
        config.training.batch_size,
        config.data.num_workers,
        config.seed,
    )
    model = build_model(config, category, device)
    model.load_trainable_state_dict(checkpoint["model"])
    memory = FeatureMemory.from_state_dict(checkpoint["memory"])
    output_dir = Path(config.output_dir) / category / "evaluation"
    metrics, records = evaluate(model, memory, test_loader, config, device, output_dir)
    write_json(output_dir / "metrics.json", metrics)
    _save_predictions(output_dir / "predictions.csv", records)
    return category, metrics, records
