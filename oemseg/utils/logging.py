"""Experiment logging helpers."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def logger_for(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"oemseg.{run_dir.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def format_log_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}: {value:.6f}" for key, value in metrics.items())


def config_dict(args: argparse.Namespace, device: str) -> dict[str, object]:
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return values | {"device": str(device)}
