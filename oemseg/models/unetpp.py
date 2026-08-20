"""UNet++ adapter backed by segmentation-models-pytorch."""

from __future__ import annotations

import segmentation_models_pytorch as smp
from torch import Tensor, nn

from oemseg.constants import NUM_CLASSES
from oemseg.models.base import SegmentationModelAdapter
from oemseg.models.registry import register_model


class UNetPlusPlusAdapter(SegmentationModelAdapter):
    def __init__(self, variant: str = "resnet18", pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=variant,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )

    @property
    def backbone(self) -> nn.Module:
        return self.model.encoder

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def state_dict(self, *args, **kwargs):
        """Keep legacy SMP key names instead of adding a `model.` prefix."""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.model.load_state_dict(state_dict, *args, **kwargs)


@register_model("unetpp", aliases=("unetplusplus", "unet++"))
def build_unetpp(variant: str, pretrained: bool, decoder: str, decoder_channels: int = 512) -> UNetPlusPlusAdapter:
    del decoder, decoder_channels
    return UNetPlusPlusAdapter(variant=variant, pretrained=pretrained)
