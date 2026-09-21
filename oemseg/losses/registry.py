"""Loss registry and model-specific loss policy."""

from __future__ import annotations

from torch import nn

from oemseg.constants import NUM_CLASSES
from oemseg.models.registry import normalize_name

_MODEL_DEFAULT_LOSSES = {
    "unet": "ce_dice",
    "u2net": "u2net",
    "unetpp": "ce_dice",
    "unetformer": "unetformer",
    "segformer": "ce",
    "segnext": "ce",
    "repstdc": "repstdc",
    "mambavision": "mambavision",
    "pyramidmamba": "ce_dice",
    "mask2former": "mask2former",
}
_MODEL_ONLY_LOSSES = {
    "unetformer": "unetformer",
    "u2net": "u2net",
    "soft_ce_dice": "unetformer",
    "repstdc": "repstdc",
    "mambavision": "mambavision",
    "mask2former": "mask2former",
}
_STRICT_NATIVE_MODELS = {"u2net": "u2net", "repstdc": "repstdc", "mambavision": "mambavision", "mask2former": "mask2former"}


def normalize_loss_name(name: str) -> str:
    key = normalize_name(name)
    return "ce_dice" if key == "cedice" else key


def available_losses() -> tuple[str, ...]:
    return (
        "auto",
        "ce",
        "ce_dice",
        "dice",
        "soft_ce_dice",
        "unetformer",
        "u2net",
        "mambavision",
        "repstdc",
        "mask2former",
    )


def resolve_loss_name(name: str, model_name: str, model_variant: str | None = None) -> str:
    key = normalize_loss_name(name)
    model = normalize_name(model_name)
    if key == "auto":
        try:
            return _MODEL_DEFAULT_LOSSES[model]
        except KeyError as error:
            raise ValueError(f"No default loss registered for model '{model_name}'") from error
    if key not in available_losses():
        raise ValueError(f"Unknown loss '{name}'. Valid losses: {', '.join(available_losses())}")

    required_native = _STRICT_NATIVE_MODELS.get(model)
    if required_native is not None and key != required_native:
        raise ValueError(
            f"Model '{model}' requires --loss {required_native}; "
            f"'{key}' would drop architecture-native supervision"
        )
    if model == "unetformer" and key != "unetformer":
        variant = normalize_name(model_variant or "")
        if variant not in {"swin_b", "swin_base", "swinb"}:
            raise ValueError(
                "UNetFormer/ResNet18 requires --loss unetformer so its auxiliary head remains supervised"
            )
    owner = _MODEL_ONLY_LOSSES.get(key)
    if owner is not None and model != owner:
        raise ValueError(f"Loss '{key}' is only valid with --model {owner}")
    return key


def build_loss(name: str) -> nn.Module:
    key = normalize_loss_name(name)
    if key in {"ce", "dice", "ce_dice", "mambavision", "u2net"}:
        from oemseg.losses.segmentation import CrossEntropyDiceLoss, CrossEntropyLoss, DiceLoss

        builders = {
            "ce": CrossEntropyLoss,
            "mambavision": CrossEntropyLoss,
            "u2net": CrossEntropyLoss,
            "dice": DiceLoss,
            "ce_dice": CrossEntropyDiceLoss,
        }
        return builders[key]()
    if key in {"unetformer", "soft_ce_dice"}:
        from oemseg.losses.unetformer import build_unetformer_reporting_loss

        return build_unetformer_reporting_loss(NUM_CLASSES)
    if key == "repstdc":
        from oemseg.losses.repstdc import build_repstdc_reporting_loss

        return build_repstdc_reporting_loss()
    if key == "mask2former":
        from oemseg.losses.mask2former import build_mask2former_reporting_loss

        return build_mask2former_reporting_loss()
    raise ValueError(f"Unknown loss '{name}'. Valid losses: {', '.join(available_losses())}")
