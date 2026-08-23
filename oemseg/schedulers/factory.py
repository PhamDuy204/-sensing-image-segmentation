"""Learning-rate and evaluation scheduling helpers."""

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
    """Legacy fraction-based schedule: validation and test run together."""
    start_epoch = max(1, math.ceil(total_epochs * start_fraction))
    return epoch == total_epochs or (epoch >= start_epoch and (epoch - start_epoch) % every == 0)


def evaluation_schedule(
    epoch: int,
    total_epochs: int,
    train_only_epochs: int,
    validation_every: int,
    test_every_validations: int,
) -> tuple[bool, bool]:
    """Return ``(run_validation, run_test)`` for the epoch.

    ``train_only_epochs`` is the count of initial epochs with no evaluation. The
    final epoch always runs both validation and test.
    """
    if epoch == total_epochs:
        return True, True
    first_validation = train_only_epochs + 1
    if epoch < first_validation or (epoch - first_validation) % validation_every:
        return False, False
    validation_index = (epoch - first_validation) // validation_every + 1
    return True, validation_index % test_every_validations == 0
