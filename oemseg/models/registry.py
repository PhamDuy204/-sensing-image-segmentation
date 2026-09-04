"""Lazy model registry."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable

from oemseg.models.base import SegmentationModelAdapter

Builder = Callable[[str, bool, str, int], SegmentationModelAdapter]
_REGISTRY: dict[str, Builder | None] = {
    "unet": None,
    "unetpp": None,
    "unetformer": None,
    "segformer": None,
    "mambavision": None,
    "pyramidmamba": None,
}
_ALIASES = {
    "unetplusplus": "unetpp",
    "unet_plus_plus": "unetpp",
    "mamba_vision": "mambavision",
    "pyramid_mamba": "pyramidmamba",
}


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return _ALIASES.get(normalized, normalized)


def register_model(name: str, aliases: tuple[str, ...] = ()):
    key = normalize_name(name)

    def decorator(builder: Builder) -> Builder:
        _REGISTRY[key] = builder
        for alias in aliases:
            _ALIASES[normalize_name(alias)] = key
        return builder

    return decorator


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _load_builder(name: str) -> Builder:
    key = normalize_name(name)
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Valid models: {', '.join(available_models())}")
    if _REGISTRY[key] is None:
        if key == "unet":
            from oemseg.models import unet  # noqa: F401
        elif key == "unetpp":
            from oemseg.models import unetpp  # noqa: F401
        elif key == "unetformer":
            from oemseg.models import unetformer  # noqa: F401
        elif key == "segformer":
            from oemseg.models import segformer  # noqa: F401
        elif key == "mambavision":
            from oemseg.models import mambavision  # noqa: F401
        elif key == "pyramidmamba":
            from oemseg.models import pyramidmamba  # noqa: F401
    builder = _REGISTRY[key]
    if builder is None:
        raise RuntimeError(f"Model '{key}' failed to register")
    return builder


def build_model_from_values(
    name: str,
    variant: str,
    pretrained: bool,
    decoder: str = "upernet",
    decoder_channels: int = 512,
) -> SegmentationModelAdapter:
    return _load_builder(name)(variant, pretrained, decoder, decoder_channels)


def build_model(args: argparse.Namespace) -> SegmentationModelAdapter:
    return build_model_from_values(
        args.model, args.model_variant, args.pretrained, args.decoder, args.decoder_channels
    )
