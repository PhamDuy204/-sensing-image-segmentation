"""Shared multi-scale and flip test-time augmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def model_logits(model: nn.Module, images: Tensor, scales: list[float], flips: bool) -> Tensor:
    outputs: list[Tensor] = []
    original_size = images.shape[-2:]
    for scale in scales:
        scaled = (
            images
            if scale == 1.0
            else F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)
        )
        variants: list[tuple[Tensor, int | None]] = [(scaled, None)]
        if flips:
            variants.extend(
                [
                    (torch.flip(scaled, (-1,)), -1),
                    (torch.flip(scaled, (-2,)), -2),
                ]
            )
        for variant, flip_dimension in variants:
            logits = model(variant)
            if flip_dimension is not None:
                logits = torch.flip(logits, (flip_dimension,))
            if logits.shape[-2:] != original_size:
                logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
            outputs.append(logits)
    if not outputs:
        raise ValueError("At least one TTA scale is required")
    return torch.stack(outputs).mean(0)
