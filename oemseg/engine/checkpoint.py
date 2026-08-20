"""Checkpoint serialization with legacy-compatible keys."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    args,
    metadata: dict[str, object] | None = None,
    model_state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model_state_dict if model_state_dict is not None else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
    }
    if metadata:
        payload.update(metadata)
    torch.save(payload, path)
