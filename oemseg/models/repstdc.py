"""RepSTDC-CA adapter using the pinned official RepSTDC implementation."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch.nn.functional as F
from torch import Tensor, nn

from oemseg.constants import NUM_CLASSES
from oemseg.losses.repstdc import repstdc_native_loss
from oemseg.models.base import SegmentationModelAdapter
from oemseg.models.registry import register_model

REPSTDC_DIR = Path(__file__).resolve().parents[2] / ".vendor" / "RepSTDC"


def _openmmlab_classes():
    source_root = REPSTDC_DIR / "mmseg_geo"
    if not (source_root / "mmseg_geo" / "models" / "backbones" / "repstdc.py").exists():
        raise ImportError("RepSTDC source is missing; run python scripts/paper_models.py setup repstdc")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from mmseg.models.decode_heads import FCNHead, STDCHead
        from mmseg_geo.models.backbones import RepSTDCContextPathNet
    except (ImportError, ModuleNotFoundError) as error:
        raise ImportError(
            "RepSTDC requires the OpenMMLab baseline environment; run "
            "bash scripts/setup_openmmlab_baselines.sh and use conda env oem-openmmlab."
        ) from error
    return RepSTDCContextPathNet, FCNHead, STDCHead


class RepSTDCAdapter(SegmentationModelAdapter):
    native_loss_name = "repstdc"

    def __init__(
        self,
        variant: str = "stdc1-ca",
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        backbone: nn.Module | None = None,
        decode_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if variant.lower() not in {"stdc1-ca", "repstdc-ca", "ca"}:
            raise ValueError("Supported RepSTDC variants: stdc1-ca")
        if (backbone is None) != (decode_head is None):
            raise ValueError("backbone and decode_head must be supplied together")
        injected_components = backbone is not None
        auxiliary_heads: nn.ModuleList = nn.ModuleList()
        if backbone is None:
            RepSTDCContextPathNet, FCNHead, STDCHead = _openmmlab_classes()
            if pretrained:
                warnings.warn(
                    "The official RepSTDC OEM config has no pretrained checkpoint; using published scratch initialization.",
                    stacklevel=2,
                )
            norm_cfg = dict(type="BN", requires_grad=True)
            ohem = dict(type="OHEMPixelSampler", thresh=0.7, min_kept=10000)
            backbone = RepSTDCContextPathNet(
                backbone_cfg=dict(
                    type="RepSTDCNet",
                    stdc_type="STDCNet1",
                    in_channels=3,
                    channels=(32, 64, 256, 512, 1024),
                    bottleneck_type="cat",
                    num_convs=4,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="ReLU"),
                    with_final_conv=False,
                ),
                last_in_channels=(1024, 512),
                out_channels=128,
                ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4),
                fusion_type="CA",
                norm_cfg=norm_cfg,
            )
            decode_head = FCNHead(
                in_channels=256,
                channels=256,
                num_convs=1,
                num_classes=num_classes,
                in_index=3,
                concat_input=False,
                dropout_ratio=0.1,
                norm_cfg=norm_cfg,
                align_corners=True,
                sampler=ohem,
                loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            )
            auxiliary_heads = nn.ModuleList(
                [
                    FCNHead(
                        in_channels=128,
                        channels=64,
                        num_convs=1,
                        num_classes=num_classes,
                        in_index=index,
                        concat_input=False,
                        norm_cfg=norm_cfg,
                        align_corners=False,
                        sampler=ohem,
                        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                    )
                    for index in (2, 1)
                ]
                + [
                    STDCHead(
                        in_channels=256,
                        channels=64,
                        num_convs=1,
                        num_classes=2,
                        boundary_threshold=0.1,
                        in_index=0,
                        concat_input=False,
                        norm_cfg=norm_cfg,
                        align_corners=True,
                        loss_decode=[
                            dict(
                                type="CrossEntropyLoss",
                                loss_name="loss_ce",
                                use_sigmoid=True,
                                loss_weight=1.0,
                            ),
                            dict(type="DiceLoss", loss_name="loss_dice", loss_weight=1.0),
                        ],
                    )
                ]
            )
        self._backbone = backbone
        self.decode_head = decode_head
        self.auxiliary_heads = auxiliary_heads
        self.uses_native_loss = not injected_components

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    def forward(self, images: Tensor, targets: Tensor | None = None) -> Tensor:
        features = self._backbone(images)
        if targets is not None:
            if not self.uses_native_loss:
                raise RuntimeError("RepSTDC native loss is unavailable for injected test components")
            return repstdc_native_loss(self.decode_head, self.auxiliary_heads, features, targets)
        logits = self.decode_head(features)
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


@register_model("repstdc", aliases=("rep-stdc", "rep_stdc"))
def build_repstdc(variant: str, pretrained: bool, decoder: str, decoder_channels: int = 512) -> RepSTDCAdapter:
    del decoder, decoder_channels
    return RepSTDCAdapter(variant=variant, pretrained=pretrained)
