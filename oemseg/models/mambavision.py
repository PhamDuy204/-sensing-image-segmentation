"""Official NVIDIA MambaVision-T backbone with a compact UPerNet decoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from oemseg.constants import NUM_CLASSES
from oemseg.models.base import SegmentationModelAdapter
from oemseg.models.registry import register_model
from oemseg.models.upernet import UPerNetHead

MAMBAVISION_REPOSITORY = "nvidia/MambaVision-T-1K"
MAMBAVISION_REVISION = "b1de77e17599566d98efb701c0231b1095dc3a67"
MAMBAVISION_CHANNELS = (80, 160, 320, 640)


def _raise_optional_dependency_error(error: Exception) -> None:
    raise ImportError(
        "MambaVision requires transformers, timm, einops, and mamba-ssm. Install requirements.txt in conda "
        "environment work-env."
    ) from error


def _official_backbone(pretrained: bool) -> nn.Module:
    try:
        from transformers import AutoConfig, AutoModel

        if pretrained:
            return AutoModel.from_pretrained(
                MAMBAVISION_REPOSITORY,
                revision=MAMBAVISION_REVISION,
                trust_remote_code=True,
            )
        config = AutoConfig.from_pretrained(
            MAMBAVISION_REPOSITORY,
            revision=MAMBAVISION_REVISION,
            trust_remote_code=True,
        )
        return AutoModel.from_config(config, trust_remote_code=True)
    except (ImportError, ModuleNotFoundError) as error:
        _raise_optional_dependency_error(error)


class MambaVisionAdapter(SegmentationModelAdapter):
    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = True,
        decoder: str = "upernet",
        num_classes: int = NUM_CLASSES,
        backbone: nn.Module | None = None,
        decoder_channels: int = 512,
    ) -> None:
        super().__init__()
        if variant.lower() not in {"tiny", "t"}:
            raise ValueError("Supported MambaVision variants: tiny")
        if decoder.lower() != "upernet":
            raise ValueError("MambaVision currently supports decoder: upernet")
        self._backbone = backbone if backbone is not None else _official_backbone(pretrained)
        self.head = UPerNetHead(
            MAMBAVISION_CHANNELS,
            channels=decoder_channels,
            num_classes=num_classes,
        )

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    def _features(self, images: Tensor) -> Sequence[Tensor]:
        output = self._backbone(images)
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError("Official MambaVision output must be (pooled, feature_maps)")
        features = output[1]
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise RuntimeError("Official MambaVision must return four feature maps")
        channels = [feature.shape[1] for feature in features]
        if channels != list(MAMBAVISION_CHANNELS):
            raise RuntimeError(
                f"MambaVision-T channels must be {list(MAMBAVISION_CHANNELS)}, got {channels}"
            )
        return features

    def forward(self, images: Tensor) -> Tensor:
        logits = self.head(self._features(images))
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


@register_model("mambavision", aliases=("mamba-vision", "mamba_vision"))
def build_mambavision(
    variant: str,
    pretrained: bool,
    decoder: str,
    decoder_channels: int = 512,
) -> MambaVisionAdapter:
    return MambaVisionAdapter(
        variant=variant,
        pretrained=pretrained,
        decoder=decoder,
        decoder_channels=decoder_channels,
    )
