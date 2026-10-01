"""Dual-Prompt few-shot industrial anomaly detection."""

from .config import ExperimentConfig, load_config
from .model import DualPromptModel

__all__ = ["DualPromptModel", "ExperimentConfig", "load_config"]
__version__ = "0.1.0"
