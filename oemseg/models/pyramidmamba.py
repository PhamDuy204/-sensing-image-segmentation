"""PyramidMamba adapter using the pinned upstream GeoSeg implementation."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import torch
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


class _Float32Module(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.module(x.float())


def _repair_mamba_dt_projection(model: nn.Module) -> None:
    for module in model.modules():
        dt_proj = getattr(module, "dt_proj", None)
        bias = getattr(dt_proj, "bias", None)
        if not isinstance(dt_proj, nn.Linear) or bias is None or not getattr(bias, "_no_reinit", False):
            continue

        bias._no_weight_decay = True
        for parameter_name in ("A_log", "D"):
            parameter = getattr(module, parameter_name, None)
            if isinstance(parameter, nn.Parameter):
                parameter._no_weight_decay = True

        if torch.count_nonzero(bias.detach()).item() != 0:
            continue

        dt_rank = int(getattr(module, "dt_rank", dt_proj.in_features))
        dt_init_std = dt_rank**-0.5
        with torch.no_grad():
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(
                torch.rand_like(bias) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            bias.copy_(dt + torch.log(-torch.expm1(-dt)))


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
        _repair_mamba_dt_projection(self.model)
        mamba_layer = getattr(getattr(getattr(self.model, "decoder", None), "b3", None), "mamba", None)
        if isinstance(mamba_layer, nn.Module) and not isinstance(mamba_layer, _Float32Module):
            self.model.decoder.b3.mamba = _Float32Module(mamba_layer)

    @property
    def backbone(self) -> nn.Module:
        return self.model.backbone

    def forward(self, images: Tensor) -> Tensor:
        # PyramidMamba is numerically unstable on Kaggle T4 when the Swin
        # backbone/decoder remain under fp16 autocast for long DDP runs.  Run
        # the complete model in float32 while keeping the shared trainer's AMP
        # policy unchanged for the other baselines.
        with torch.autocast(device_type=images.device.type, enabled=False):
            logits = self.model(images.float())
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
