"""Mask2Former native training loss helpers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def build_mask2former_targets(targets: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    """Convert a dense semantic mask into the set targets expected by Mask2Former."""
    masks: list[Tensor] = []
    labels: list[Tensor] = []
    for target in targets:
        present = torch.unique(target).long()
        labels.append(present)
        masks.append(torch.stack([target.eq(label) for label in present]).float())
    return masks, labels


def mask2former_native_loss(model: nn.Module, images: Tensor, targets: Tensor) -> Tensor:
    """Delegate training loss to Hugging Face's native Mask2Former criterion."""
    mask_labels, class_labels = build_mask2former_targets(targets)
    output = model(
        pixel_values=images,
        mask_labels=mask_labels,
        class_labels=class_labels,
    )
    if output.loss is None:
        raise RuntimeError("Mask2Former did not return its native training loss")
    return output.loss


def build_mask2former_reporting_loss() -> nn.Module:
    """Dense proxy loss for validation/error-analysis; training stays native."""
    return nn.CrossEntropyLoss()
