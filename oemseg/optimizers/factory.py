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

    decay_groups = []
    no_decay_groups = []
    for group in parameter_groups:
        params = list(group["params"])
        options = {key: value for key, value in group.items() if key != "params"}
        decay = [parameter for parameter in params if not getattr(parameter, "_no_weight_decay", False)]
        no_decay = [parameter for parameter in params if getattr(parameter, "_no_weight_decay", False)]
        if decay:
            decay_groups.append({**options, "params": decay})
        if no_decay:
            no_decay_groups.append({**options, "params": no_decay, "weight_decay": 0.0})

    return classes[key]([*decay_groups, *no_decay_groups], weight_decay=weight_decay)
