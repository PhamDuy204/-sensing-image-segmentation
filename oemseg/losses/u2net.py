"""U2-Net deep-supervision loss adapted to multiclass segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn


class U2NetDeepSupervisionLoss(nn.Module):
    """Sum CE over the fused prediction and six side outputs.

    The U2-Net paper supervises all seven outputs with equal weights. The
    original binary BCE is replaced by multiclass CE for OpenEarthMap.
    """

    def forward(self, outputs: Sequence[Tensor], target: Tensor) -> Tensor:
        if len(outputs) != 7:
            raise ValueError(f"U2-Net expects 7 supervised outputs, got {len(outputs)}")
        return sum(F.cross_entropy(logits, target) for logits in outputs)
