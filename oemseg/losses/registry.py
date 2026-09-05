"""Loss registry and model-specific loss policy."""

from __future__ import annotations

from torch import nn

from oemseg.constants import NUM_CLASSES
from oemseg.models.registry import normalize_name

_MODEL_DEFAULT_LOSSES = {
    "unet": "ce_dice",
    "unetpp": "ce_dice",
    "unetformer": "unetformer",
    "segformer": "ce",
    "segnext": "ce",
    "repstdc": "repstdc",
    "mambavision": "ce_dice",
    "pyramidmamba": "ce_dice",
    "mask2former": "mask2former",
}
_MODEL_ONLY_LOSSES = {
    "unetformer": "unetformer",
    "soft_ce_dice": "unetformer",
    "repstdc": "repstdc",
    "mask2former": "mask2former",
}
_STRICT_NATIVE_MODELS = {"mask2former": "mask2former"}


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
        "repstdc",
        "mask2former",
    )


def resolve_loss_name(name: str, model_name: str) -> str:
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
            f"a dense loss such as '{key}' is not compatible with its set-prediction objective"
        )
    owner = _MODEL_ONLY_LOSSES.get(key)
    if owner is not None and model != owner:
        raise ValueError(f"Loss '{key}' is only valid with --model {owner}")
    return key


def build_loss(name: str) -> nn.Module:
    key = normalize_loss_name(name)
    if key in {"ce", "dice", "ce_dice"}:
        from oemseg.losses.segmentation import CrossEntropyDiceLoss, CrossEntropyLoss, DiceLoss

        builders = {
            "ce": CrossEntropyLoss,
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
