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


def _load_upstream():
    if not (GEOSEG_DIR / "geoseg" / "models" / "UNetFormer.py").exists():
        raise ImportError("UNetFormer source is missing; run bash scripts/setup_unetformer.sh")
    sys.path.insert(0, str(GEOSEG_DIR))
    upstream = importlib.import_module("geoseg.models.UNetFormer")

    # GeoSeg's older two-value reflect padding call is rejected by modern PyTorch for 4-D tensors.
    def compatible_pad(self, x, patch_size):
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

    upstream.GlobalLocalAttention.pad = compatible_pad
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
        if model is None:
            upstream = _load_upstream()
            model = upstream.UNetFormer(
                decode_channels=decoder_channels,
                backbone_name=variant,
                pretrained=pretrained,
                num_classes=num_classes,
            )
        self.model = model
        aux_head = getattr(getattr(self.model, "decoder", None), "aux_head", None)
        if aux_head is not None:
            for parameter in aux_head.parameters():
                parameter.requires_grad_(False)

    @property
    def backbone(self) -> nn.Module:
        return self.model.backbone

    def forward(self, images: Tensor) -> Tensor:
        output = self.model(images)
        return output[0] if isinstance(output, tuple) else output


@register_model("unetformer", aliases=("unet-former", "unet_former"))
def build_unetformer(
    variant: str,
    pretrained: bool,
    decoder: str,
    decoder_channels: int = 512,
) -> UNetFormerAdapter:
    del decoder
    # GeoSeg's published UNetFormer uses a compact 64-channel decoder by default.
    channels = 64 if decoder_channels == 512 else decoder_channels
    return UNetFormerAdapter(
        variant=variant,
        pretrained=pretrained,
        decoder_channels=channels,
    )
