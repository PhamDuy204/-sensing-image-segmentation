"""U2-Net adapter using the official nested U-structure implementation."""

from __future__ import annotations

from torch import Tensor, nn

from oemseg.constants import NUM_CLASSES
from oemseg.losses.u2net import U2NetDeepSupervisionLoss
from oemseg.models.base import SegmentationModelAdapter
from oemseg.models.registry import register_model
from oemseg.models.u2net_upstream import U2NET_full, U2NET_lite


class U2NetAdapter(SegmentationModelAdapter):
    native_loss_name = "u2net"
    uses_native_loss = True

    def __init__(
        self,
        variant: str = "full",
        pretrained: bool = False,
        num_classes: int = NUM_CLASSES,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        key = variant.lower().replace("_", "-")
        if key not in {"full", "lite", "small", "u2netp"}:
            raise ValueError("Supported U2-Net variants: full, lite")
        # The paper trains U2-Net from scratch; its published SOD weights have a
        # binary output head and are not compatible with 9-class OEM training.
        del pretrained
        self.model = model or (U2NET_full(num_classes) if key == "full" else U2NET_lite(num_classes))
        self.loss = U2NetDeepSupervisionLoss()

    @property
    def backbone(self) -> nn.Module:
        return self.model

    def parameter_groups(self, base_lr: float, backbone_lr: float) -> list[dict[str, object]]:
        del backbone_lr
        encoder_stages = {f"stage{i}" for i in range(1, 7)}
        backbone = []
        main = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            top = name.split(".", 1)[0]
            (backbone if top in encoder_stages else main).append(parameter)
        if not backbone or not main:
            raise RuntimeError("U2-Net parameter groups must be nonempty")
        return [
            {"params": backbone, "lr": base_lr, "group_name": "backbone"},
            {"params": main, "lr": base_lr, "group_name": "main"},
        ]

    def forward(self, images: Tensor, targets: Tensor | None = None) -> Tensor:
        outputs = self.model(images)
        if targets is not None:
            return self.loss(outputs, targets)
        return outputs[0]


@register_model("u2net", aliases=("u2-net", "u2_net", "u2"))
def build_u2net(
    variant: str,
    pretrained: bool,
    decoder: str,
    decoder_channels: int = 512,
) -> U2NetAdapter:
    del decoder, decoder_channels
    return U2NetAdapter(variant=variant, pretrained=pretrained)
