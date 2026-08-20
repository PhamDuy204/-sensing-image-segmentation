"""SegFormer semantic-segmentation adapter using Hugging Face Transformers."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from oemseg.constants import CLASS_NAMES, NUM_CLASSES
from oemseg.models.base import SegmentationModelAdapter
from oemseg.models.registry import register_model


def _transformers_classes():
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as error:
        raise ImportError(
            "SegFormer requires transformers and safetensors. Install requirements.txt "
            "in conda environment work-env."
        ) from error
    return SegformerConfig, SegformerForSemanticSegmentation


def _b0_config(num_classes: int):
    SegformerConfig, _ = _transformers_classes()
    id2label = {index: name for index, name in enumerate(CLASS_NAMES[:num_classes])}
    return SegformerConfig(
        num_labels=num_classes,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        decoder_hidden_size=256,
        reshape_last_stage=True,
        id2label=id2label,
        label2id={name: index for index, name in id2label.items()},
    )


class SegFormerAdapter(SegmentationModelAdapter):
    def __init__(self, variant: str = "b0", pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super().__init__()
        if variant.lower() not in {"b0", "mit_b0", "mit-b0"}:
            raise ValueError("Supported SegFormer variants: b0")
        _, Model = _transformers_classes()
        if pretrained:
            id2label = {index: name for index, name in enumerate(CLASS_NAMES[:num_classes])}
            self.model, loading = Model.from_pretrained(
                "nvidia/mit-b0",
                num_labels=num_classes,
                id2label=id2label,
                label2id={name: index for index, name in id2label.items()},
                ignore_mismatched_sizes=True,
                output_loading_info=True,
            )
            unexpected_backbone_missing = [
                key for key in loading.get("missing_keys", []) if not key.startswith("decode_head.")
            ]
            if unexpected_backbone_missing:
                raise RuntimeError(
                    "SegFormer pretrained backbone did not load completely: "
                    + ", ".join(unexpected_backbone_missing[:5])
                )
        else:
            self.model = Model(_b0_config(num_classes))

    @property
    def backbone(self) -> nn.Module:
        return self.model.segformer

    def forward(self, images: Tensor) -> Tensor:
        logits = self.model(pixel_values=images).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


@register_model("segformer")
def build_segformer(variant: str, pretrained: bool, decoder: str, decoder_channels: int = 512) -> SegFormerAdapter:
    del decoder, decoder_channels
    return SegFormerAdapter(variant=variant, pretrained=pretrained)
