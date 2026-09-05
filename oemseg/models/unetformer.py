"""UNetFormer adapter using the pinned upstream GeoSeg implementation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch.nn.functional as F
from torch import Tensor, nn

from oemseg.constants import NUM_CLASSES
from oemseg.models.base import SegmentationModelAdapter
from oemseg.models.registry import register_model

GEOSEG_DIR = Path(__file__).resolve().parents[2] / ".vendor" / "GeoSeg"
SWIN_BASE_WEIGHT = GEOSEG_DIR / "pretrain_weights" / "stseg_base.pth"


def _compatible_pad(self, x, patch_size):
    # GeoSeg's older two-value reflect padding call is rejected by modern PyTorch for 4-D tensors.
    _, _, height, width = x.shape
    pad_width = (-width) % patch_size
    pad_height = (-height) % patch_size
    if pad_width:
        x = F.pad(
            x,
            (0, pad_width, 0, 0),
            mode="reflect" if pad_width < width else "replicate",
        )
    if pad_height:
        x = F.pad(
            x,
            (0, 0, 0, pad_height),
            mode="reflect" if pad_height < height else "replicate",
        )
    return x


def _load_upstream():
    if not (GEOSEG_DIR / "geoseg" / "models" / "UNetFormer.py").exists():
        raise ImportError("UNetFormer source is missing; run bash scripts/setup_unetformer.sh")
    sys.path.insert(0, str(GEOSEG_DIR))
    upstream = importlib.import_module("geoseg.models.UNetFormer")
    upstream.GlobalLocalAttention.pad = _compatible_pad
    return upstream


def _load_ft_upstream():
    if not (GEOSEG_DIR / "geoseg" / "models" / "FTUNetFormer.py").exists():
        raise ImportError("FTUNetFormer source is missing; run bash scripts/setup_unetformer.sh")
    sys.path.insert(0, str(GEOSEG_DIR))
    upstream = importlib.import_module("geoseg.models.FTUNetFormer")
    upstream.GlobalLocalAttention.pad = _compatible_pad
    return upstream


class UNetFormerAdapter(SegmentationModelAdapter):
    def __init__(
        self,
        variant: str = "resnet18",
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        decoder_channels: int = 64,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        loss = None
        variant_key = variant.lower().replace("_", "-")
        self.backbone_name = "swsl_resnet18" if variant_key == "resnet18" else variant
        if model is None and variant_key in {"swin-b", "swin-base", "swinb"}:
            if pretrained and not SWIN_BASE_WEIGHT.exists():
                raise ImportError("Swin-B weights are missing; run bash scripts/setup_unetformer.sh")
            upstream = _load_ft_upstream()
            model = upstream.ft_unetformer(
                pretrained=pretrained,
                num_classes=num_classes,
                decoder_channels=decoder_channels,
                weight_path=str(SWIN_BASE_WEIGHT),
            )
            losses = importlib.import_module("geoseg.losses")
            loss = losses.JointLoss(
                losses.SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=num_classes),
                losses.DiceLoss(smooth=0.05, ignore_index=num_classes),
                1.0,
                1.0,
            )
        elif model is None:
            upstream = _load_upstream()
            model = upstream.UNetFormer(
                decode_channels=decoder_channels,
                backbone_name=self.backbone_name,
                pretrained=pretrained,
                num_classes=num_classes,
            )
            loss = importlib.import_module("geoseg.losses").UnetFormerLoss(ignore_index=num_classes)
        self.model = model
        self.loss = loss
        self.uses_native_loss = loss is not None

    @property
    def backbone(self) -> nn.Module:
        return self.model.backbone

    def forward(self, images: Tensor, targets: Tensor | None = None) -> Tensor:
        output = self.model(images)
        if targets is not None:
            if self.loss is None:
                raise RuntimeError("UNetFormer native loss is unavailable for this injected model")
            return self.loss(output, targets)
        return output[0] if isinstance(output, tuple) else output


@register_model("unetformer", aliases=("unet-former", "unet_former"))
def build_unetformer(
    variant: str,
    pretrained: bool,
    decoder: str,
    decoder_channels: int = 512,
) -> UNetFormerAdapter:
    del decoder
    # GeoSeg uses 256 decoder channels for FTUNetFormer/Swin-B and 64 for its ResNet18 UNetFormer.
    if decoder_channels == 512:
        channels = 256 if variant.lower().replace("_", "-") in {"swin-b", "swin-base", "swinb"} else 64
    else:
        channels = decoder_channels
    return UNetFormerAdapter(
        variant=variant,
        pretrained=pretrained,
        decoder_channels=channels,
    )
