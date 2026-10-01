"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig, load_config
from .data import MVTEC_CATEGORIES, validate_mvtec_root
from .engine import run_evaluation, run_training
from .synthetic import create_tiny_mvtec
from .utils import write_json


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if getattr(args, "epochs", None) is not None:
        config.training.epochs = args.epochs
    if getattr(args, "category", None) not in {None, "all"}:
        config.data.category = args.category
    if getattr(args, "output_dir", None) is not None:
        config.output_dir = args.output_dir
    if getattr(args, "device", None) is not None:
        config.device = args.device
    if getattr(args, "data_root", None) is not None:
        config.data.root = args.data_root
    if getattr(args, "shots", None) is not None:
        config.data.shots = args.shots
    if getattr(args, "batch_size", None) is not None:
        config.training.batch_size = args.batch_size
    if getattr(args, "num_workers", None) is not None:
        config.data.num_workers = args.num_workers
    config.validate()


def _print_metrics(category: str, metrics: dict[str, float]) -> None:
    print(json.dumps({"category": category, **metrics}, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dual-prompt",
        description="Few-shot Dual-Prompt anomaly detection on MVTec AD",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="adapt prompts, build memory, and evaluate")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--category", choices=(*MVTEC_CATEGORIES, "all"))
    train.add_argument("--epochs", type=int)
    train.add_argument("--output-dir", type=str)
    train.add_argument("--device", type=str)
    train.add_argument("--data-root", type=str)
    train.add_argument("--shots", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--num-workers", type=int)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved category checkpoint")
    evaluate.add_argument("--config", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--output-dir", type=str)
    evaluate.add_argument("--device", type=str)
    evaluate.add_argument("--data-root", type=str)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--num-workers", type=int)

    inspect = subparsers.add_parser("inspect-data", help="validate an MVTec AD directory")
    inspect.add_argument("--root", required=True, type=Path)

    smoke = subparsers.add_parser("smoke", help="run one epoch on a tiny MVTec-shaped fixture")
    smoke.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    smoke.add_argument("--output-dir", type=str)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-data":
        print(json.dumps(validate_mvtec_root(args.root), indent=2, sort_keys=True))
        return 0

    config = load_config(args.config)
    _apply_overrides(config, args)

    if args.command == "smoke":
        create_tiny_mvtec(config.data.root, config.data.image_size, config.seed)
        artifacts = run_training(config, "bottle")
        _print_metrics("bottle", artifacts.metrics)
        print(f"checkpoint: {artifacts.checkpoint}")
        return 0

    if args.command == "evaluate":
        category, metrics, _ = run_evaluation(config, args.checkpoint)
        _print_metrics(category, metrics)
        return 0

    categories = MVTEC_CATEGORIES if args.category == "all" else (config.data.category,)
    all_metrics: dict[str, dict[str, float]] = {}
    for category in categories:
        artifacts = run_training(config, category)
        all_metrics[category] = artifacts.metrics
        _print_metrics(category, artifacts.metrics)
    if len(all_metrics) > 1:
        metric_names = next(iter(all_metrics.values()))
        mean = {
            name: sum(metrics[name] for metrics in all_metrics.values()) / len(all_metrics)
            for name in metric_names
        }
        write_json(
            Path(config.output_dir) / "summary.json",
            {"categories": all_metrics, "mean": mean},
        )
        _print_metrics("mean", mean)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
