"""Learning-rate scheduler factory."""

from __future__ import annotations

import math

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    power: float,
):
    def multiplier(epoch: int) -> float:
        current = epoch + 1
        if current <= warmup_epochs:
            return current / max(1, warmup_epochs)
        progress = (current - warmup_epochs) / max(1, epochs - warmup_epochs)
        return max(0.0, 1.0 - progress) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def should_evaluate(epoch: int, total_epochs: int, start_fraction: float, every: int) -> bool:
    start_epoch = max(1, math.ceil(total_epochs * start_fraction))
    return epoch == total_epochs or (epoch >= start_epoch and (epoch - start_epoch) % every == 0)
