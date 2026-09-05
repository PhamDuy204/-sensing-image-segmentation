"""SegNeXt-T adapter using MMSegmentation's official MSCAN/LightHamHead."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from oemseg.constants import NUM_CLASSES
from oemseg.models.base import SegmentationModelAdapter
from oemseg.models.registry import register_model

MSCAN_T_CHECKPOINT = (
    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/"
    "mscan_t_20230227-119e8c9f.pth"
)


def _openmmlab_classes():
    try:
        from mmseg.models.backbones import MSCAN
        from mmseg.models.decode_heads import LightHamHead
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            "SegNeXt requires the OpenMMLab baseline environment; run "
            "bash scripts/setup_openmmlab_baselines.sh and use conda env oem-openmmlab."
        ) from error
    return MSCAN, LightHamHead


class SegNeXtAdapter(SegmentationModelAdapter):
    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        backbone: nn.Module | None = None,
        decode_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if variant.lower() not in {"tiny", "t", "mscan-t", "mscan_t"}:
            raise ValueError("Supported SegNeXt variants: tiny")
        if (backbone is None) != (decode_head is None):
            raise ValueError("backbone and decode_head must be supplied together")
        if backbone is None:
            MSCAN, LightHamHead = _openmmlab_classes()
            norm_cfg = dict(type="BN", requires_grad=True)
            backbone = MSCAN(
                init_cfg=dict(type="Pretrained", checkpoint=MSCAN_T_CHECKPOINT) if pretrained else None,
                embed_dims=[32, 64, 160, 256],
                mlp_ratios=[8, 8, 4, 4],
                drop_rate=0.0,
                drop_path_rate=0.1,
                depths=[3, 3, 5, 2],
                attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                act_cfg=dict(type="GELU"),
                norm_cfg=norm_cfg,
            )
            if pretrained:
                backbone.init_weights()
            decode_head = LightHamHead(
                in_channels=[64, 160, 256],
                in_index=[1, 2, 3],
                channels=256,
                ham_channels=256,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
                align_corners=False,
                loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                ham_kwargs=dict(
                    MD_S=1,
                    MD_R=16,
                    train_steps=6,
                    eval_steps=7,
                    inv_t=100,
                    rand_init=True,
                ),
            )
        self._backbone = backbone
        self.decode_head = decode_head

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    def forward(self, images: Tensor) -> Tensor:
        features = self._backbone(images)
        with torch.autocast(device_type=images.device.type, enabled=False):
            logits = self.decode_head(tuple(feature.float() for feature in features))
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


@register_model("segnext", aliases=("seg-next", "seg_next"))
def build_segnext(variant: str, pretrained: bool, decoder: str, decoder_channels: int = 512) -> SegNeXtAdapter:
    del decoder, decoder_channels
    return SegNeXtAdapter(variant=variant, pretrained=pretrained)
