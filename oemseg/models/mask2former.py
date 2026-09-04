"""Mask2Former semantic-segmentation adapter using Hugging Face Transformers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from oemseg.constants import CLASS_NAMES, NUM_CLASSES
from oemseg.models.base import SegmentationModelAdapter
from oemseg.models.registry import register_model

MODEL_ID = "facebook/mask2former-swin-tiny-ade-semantic"


def _transformers_classes():
    try:
        from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig
    except ImportError as error:
        raise ImportError("Mask2Former requires transformers; install requirements.txt") from error
    return Mask2FormerConfig, Mask2FormerForUniversalSegmentation, SwinConfig


def _tiny_config(num_classes: int):
    Mask2FormerConfig, _, SwinConfig = _transformers_classes()
    backbone = SwinConfig(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    id2label = {index: name for index, name in enumerate(CLASS_NAMES[:num_classes])}
    return Mask2FormerConfig(
        backbone_config=backbone.to_dict(),
        num_labels=num_classes,
        id2label=id2label,
        label2id={name: index for index, name in id2label.items()},
    )


class Mask2FormerAdapter(SegmentationModelAdapter):
    uses_native_loss = True

    def __init__(
        self,
        variant: str = "swin-tiny",
        pretrained: bool = True,
        num_classes: int = NUM_CLASSES,
        model: nn.Module | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if variant.lower() not in {"swin-tiny", "swin_tiny", "tiny"}:
            raise ValueError("Supported Mask2Former variants: swin-tiny")
        if model is None:
            _, Model, _ = _transformers_classes()
            if pretrained:
                id2label = {index: name for index, name in enumerate(CLASS_NAMES[:num_classes])}
                model = Model.from_pretrained(
                    MODEL_ID,
                    num_labels=num_classes,
                    id2label=id2label,
                    label2id={name: index for index, name in id2label.items()},
                    ignore_mismatched_sizes=True,
                )
            else:
                model = Model(_tiny_config(num_classes))
        self.model = model
        self._backbone = backbone or self.model.model.pixel_level_module.encoder
        self.num_classes = num_classes

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    @staticmethod
    def _targets(targets: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        masks: list[Tensor] = []
        labels: list[Tensor] = []
        for target in targets:
            present = torch.unique(target).long()
            labels.append(present)
            masks.append(torch.stack([target.eq(label) for label in present]).float())
        return masks, labels

    def forward(self, images: Tensor, targets: Tensor | None = None) -> Tensor:
        if targets is not None:
            mask_labels, class_labels = self._targets(targets)
            output = self.model(
                pixel_values=images,
                mask_labels=mask_labels,
                class_labels=class_labels,
            )
            if output.loss is None:
                raise RuntimeError("Mask2Former did not return its native training loss")
            return output.loss

        output = self.model(pixel_values=images)
        class_scores = output.class_queries_logits.softmax(dim=-1)[..., : self.num_classes]
        mask_scores = output.masks_queries_logits.sigmoid()
        semantic_scores = torch.einsum("bqc,bqhw->bchw", class_scores, mask_scores)
        return F.interpolate(
            semantic_scores,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


@register_model("mask2former", aliases=("mask-2-former", "mask_2_former"))
def build_mask2former(variant: str, pretrained: bool, decoder: str, decoder_channels: int = 512) -> Mask2FormerAdapter:
    del decoder, decoder_channels
    return Mask2FormerAdapter(variant=variant, pretrained=pretrained)
