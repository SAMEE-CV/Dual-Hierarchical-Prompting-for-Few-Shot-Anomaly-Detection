"""Detection and localization metrics."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_roc_auc(targets: np.ndarray, scores: np.ndarray, name: str) -> float:
    if np.unique(targets).size < 2:
        warnings.warn(f"{name} AUROC is undefined because only one class is present", stacklevel=2)
        return float("nan")
    return float(roc_auc_score(targets, scores))


def _safe_average_precision(targets: np.ndarray, scores: np.ndarray, name: str) -> float:
    if np.unique(targets).size < 2:
        warnings.warn(f"{name} AUPR is undefined because only one class is present", stacklevel=2)
        return float("nan")
    return float(average_precision_score(targets, scores))


def binary_dice(targets: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    predicted = scores >= threshold
    actual = targets.astype(bool)
    denominator = predicted.sum() + actual.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(predicted, actual).sum() / denominator)


def compute_metrics(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    pixel_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    flat_masks = masks.astype(np.uint8).reshape(-1)
    flat_pixel_scores = pixel_scores.reshape(-1)
    return {
        "image_auroc": _safe_roc_auc(image_labels, image_scores, "image"),
        "image_aupr": _safe_average_precision(image_labels, image_scores, "image"),
        "pixel_auroc": _safe_roc_auc(flat_masks, flat_pixel_scores, "pixel"),
        "pixel_aupr": _safe_average_precision(flat_masks, flat_pixel_scores, "pixel"),
        "pixel_dice": binary_dice(flat_masks, flat_pixel_scores, threshold),
    }
