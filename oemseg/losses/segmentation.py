"""Reusable segmentation losses."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn


class CrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.loss(logits, target)


class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.loss(logits, target)


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce = CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.ce(logits, target) + self.dice(logits, target)
