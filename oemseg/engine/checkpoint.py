"""Checkpoint serialization with legacy-compatible keys."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


RESUME_RECIPE_KEYS = (
    "model",
    "model_variant",
    "decoder",
    "decoder_channels",
    "pretrained",
    "epochs",
    "size",
    "batch_size",
    "grad_accumulation",
    "optimizer",
    "lr",
    "encoder_lr",
    "weight_decay",
    "warmup_epochs",
    "poly_power",
    "loss",
    "max_grad_norm",
    "mixed_precision",
    "channels_last",
    "internal_val_fraction",
    "patience",
    "seed",
)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    args,
    metadata: dict[str, object] | None = None,
    model_state_dict: dict[str, torch.Tensor] | None = None,
    training_state: dict[str, object] | None = None,
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
    if training_state is not None:
        payload["training_state"] = training_state
    torch.save(payload, path)


def _validate_resume_recipe(checkpoint: dict[str, object], args, world_size: int) -> None:
    if checkpoint.get("world_size") != world_size:
        raise ValueError(
            f"resume checkpoint world_size={checkpoint.get('world_size')} does not match {world_size}"
        )
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        raise ValueError("resume checkpoint is missing args")
    for key in RESUME_RECIPE_KEYS:
        saved = saved_args.get(key)
        current = getattr(args, key, None)
        if saved != current:
            raise ValueError(f"resume checkpoint {key}={saved!r} does not match {current!r}")


def restore_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args,
    *,
    world_size: int,
    map_location,
) -> dict[str, object]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    _validate_resume_recipe(checkpoint, args, world_size)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
