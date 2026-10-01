"""Layer-specific normal-feature memories and anomaly maps."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import DualPromptModel


@dataclass(slots=True)
class FeatureMemory:
    banks: dict[int, Tensor]
    grid_size: tuple[int, int]

    def state_dict(self) -> dict[str, object]:
        return {
            "banks": {str(layer): bank.cpu() for layer, bank in self.banks.items()},
            "grid_size": list(self.grid_size),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> FeatureMemory:
        raw_banks = state["banks"]
        if not isinstance(raw_banks, dict):
            raise TypeError("Invalid feature-memory checkpoint")
        banks = {int(layer): tensor for layer, tensor in raw_banks.items()}
        grid = state["grid_size"]
        if not isinstance(grid, (list, tuple)) or len(grid) != 2:
            raise TypeError("Invalid feature-memory grid")
        return cls(banks=banks, grid_size=(int(grid[0]), int(grid[1])))


@torch.no_grad()
def build_feature_memory(
    model: DualPromptModel,
    loader: DataLoader,
    device: torch.device,
    max_size: int,
    sampling: str,
    seed: int,
) -> FeatureMemory:
    model.eval()
    collected: dict[int, list[Tensor]] = {layer: [] for layer in model.prompt_layers}
    for batch in loader:
        _, features = model(batch["image"].to(device))
        for layer, tensor in features.items():
            collected[layer].append(F.normalize(tensor, dim=-1).reshape(-1, tensor.shape[-1]).cpu())

    generator = torch.Generator().manual_seed(seed)
    banks: dict[int, Tensor] = {}
    for layer, chunks in collected.items():
        if not chunks:
            raise RuntimeError(f"No features were collected for prompt layer {layer}")
        bank = torch.cat(chunks, dim=0)
        if sampling == "random" and bank.shape[0] > max_size:
            indices = torch.randperm(bank.shape[0], generator=generator)[:max_size]
            bank = bank[indices]
        banks[layer] = F.normalize(bank, dim=-1)
    return FeatureMemory(banks=banks, grid_size=model.backbone.grid_size)


def _nearest_neighbor_response(
    queries: Tensor, bank: Tensor, neighbors: int, chunk_size: int
) -> Tensor:
    """Compute Eq. (10), chunking over memory to bound peak allocation."""
    if neighbors > bank.shape[0]:
        raise ValueError(f"neighbors={neighbors} exceeds memory size {bank.shape[0]}")
    best = torch.full(
        (*queries.shape[:-1], neighbors),
        -torch.inf,
        device=queries.device,
        dtype=queries.dtype,
    )
    for start in range(0, bank.shape[0], chunk_size):
        memory_chunk = bank[start : start + chunk_size].to(queries.device, queries.dtype)
        similarities = queries @ memory_chunk.T
        candidates = torch.cat((best, similarities), dim=-1)
        best = candidates.topk(neighbors, dim=-1).values
    return 0.5 * (1.0 - best.mean(dim=-1))


def anomaly_map(
    patch_features: dict[int, Tensor],
    memory: FeatureMemory,
    output_size: tuple[int, int],
    layer_weights: list[float],
    neighbors: int,
    chunk_size: int,
) -> Tensor:
    layers = list(patch_features)
    if len(layers) != len(layer_weights):
        raise ValueError("Layer weights do not match extracted features")
    height, width = memory.grid_size
    maps = []
    for layer in layers:
        query = F.normalize(patch_features[layer], dim=-1)
        response = _nearest_neighbor_response(query, memory.banks[layer], neighbors, chunk_size)
        response = response.view(query.shape[0], 1, height, width)
        maps.append(F.interpolate(response, output_size, mode="bilinear", align_corners=False))
    fused = sum(weight * layer_map for weight, layer_map in zip(layer_weights, maps, strict=True))
    return fused.clamp(0.0, 1.0)


def image_anomaly_score(
    logits: Tensor,
    pixel_map: Tensor,
    semantic_weight: float,
    top_pixels: int,
) -> tuple[Tensor, Tensor, Tensor]:
    semantic = logits.softmax(dim=-1)[:, 1]
    flat_map = pixel_map.flatten(1)
    count = min(top_pixels, flat_map.shape[1])
    spatial = flat_map.topk(count, dim=-1).values.mean(dim=-1)
    combined = semantic_weight * semantic + (1.0 - semantic_weight) * spatial
    return combined, semantic, spatial
