"""Loss registry."""

from __future__ import annotations

from torch import nn

from oemseg.losses.segmentation import CrossEntropyDiceLoss, CrossEntropyLoss, DiceLoss
from oemseg.models.registry import normalize_name


def available_losses() -> tuple[str, ...]:
    return ("ce", "ce_dice", "dice")


def build_loss(name: str) -> nn.Module:
    key = normalize_name(name)
    if key == "cedice":
        key = "ce_dice"
    builders = {
        "ce": CrossEntropyLoss,
        "dice": DiceLoss,
        "ce_dice": CrossEntropyDiceLoss,
    }
    if key not in builders:
        raise ValueError(f"Unknown loss '{name}'. Valid losses: {', '.join(available_losses())}")
    return builders[key]()
