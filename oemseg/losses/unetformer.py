"""Official GeoSeg losses used by UNetFormer variants."""

from __future__ import annotations

import importlib

from torch import nn


def _losses_module():
    try:
        return importlib.import_module("geoseg.losses")
    except ImportError as error:
        raise ImportError(
            "UNetFormer loss requires the pinned GeoSeg source; run bash scripts/setup_unetformer.sh"
        ) from error


def build_unetformer_loss(variant: str, num_classes: int) -> nn.Module:
    """Return the exact loss recipe used by the pinned GeoSeg implementation."""
    losses = _losses_module()
    variant_key = variant.lower().replace("_", "-")
    if variant_key in {"swin-b", "swin-base", "swinb"}:
        return losses.JointLoss(
            losses.SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=num_classes),
            losses.DiceLoss(smooth=0.05, ignore_index=num_classes),
            1.0,
            1.0,
        )
    return losses.UnetFormerLoss(ignore_index=num_classes)


def build_unetformer_reporting_loss(num_classes: int) -> nn.Module:
    """Main-head loss used only for validation/error-analysis reporting."""
    losses = _losses_module()
    return losses.JointLoss(
        losses.SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=num_classes),
        losses.DiceLoss(smooth=0.05, ignore_index=num_classes),
        1.0,
        1.0,
    )
