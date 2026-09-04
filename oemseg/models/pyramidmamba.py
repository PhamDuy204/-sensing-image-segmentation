"""PyramidMamba adapter using the pinned upstream GeoSeg implementation."""

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
DEFAULT_BACKBONE = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"


def _load_upstream():
    if not (GEOSEG_DIR / "geoseg" / "models" / "PyramidMamba.py").exists():
        raise ImportError("PyramidMamba source is missing; run bash scripts/setup_unetformer.sh")
    if str(GEOSEG_DIR) not in sys.path:
        sys.path.insert(0, str(GEOSEG_DIR))
    return importlib.import_module("geoseg.models.PyramidMamba")


class PyramidMambaAdapter(SegmentationModelAdapter):
    def __init__(
        self,
        variant: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        decoder_channels: int = 128,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if model is None:
            upstream = _load_upstream()
            # ponytail: GeoSeg's published PyramidMamba constructor fixes its feature geometry
            # around 1024px inputs; extend the shared builder signature if variable sizes are needed.
            model = upstream.PyramidMamba(
                backbone_name=variant,
                pretrained=pretrained,
                num_classes=num_classes,
                decoder_channels=decoder_channels,
                last_feat_size=32,
                img_size=1024,
            )
        self.model = model

    @property
    def backbone(self) -> nn.Module:
        return self.model.backbone

    def forward(self, images: Tensor) -> Tensor:
        logits = self.model(images)
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits


@register_model("pyramidmamba", aliases=("pyramid-mamba", "pyramid_mamba"))
def build_pyramidmamba(
    variant: str,
    pretrained: bool,
    decoder: str,
    decoder_channels: int = 512,
) -> PyramidMambaAdapter:
    del decoder
    # ponytail: the shared CLI defaults decoder width to 512, while the published
    # PyramidMamba uses 128; a future per-model CLI default can remove this sentinel.
    channels = 128 if decoder_channels == 512 else decoder_channels
    return PyramidMambaAdapter(variant=variant, pretrained=pretrained, decoder_channels=channels)
