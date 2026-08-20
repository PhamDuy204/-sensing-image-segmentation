"""Optimizer factory."""

from __future__ import annotations

import torch

from oemseg.models.registry import normalize_name


def available_optimizers() -> tuple[str, ...]:
    return ("adam", "adamw")


def build_optimizer(name: str, parameter_groups, weight_decay: float) -> torch.optim.Optimizer:
    key = normalize_name(name)
    classes = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if key not in classes:
        raise ValueError(f"Unknown optimizer '{name}'. Valid optimizers: {', '.join(available_optimizers())}")
    return classes[key](parameter_groups, weight_decay=weight_decay)
