from argparse import Namespace

from torch import nn

from oemseg.models import registry


class FakeNativeModel(nn.Module):
    native_loss_name = "unetformer"
    uses_native_loss = True


def _args(loss: str) -> Namespace:
    return Namespace(
        model="unetformer",
        model_variant="swin-b",
        pretrained=False,
        decoder="upernet",
        decoder_channels=512,
        loss=loss,
    )


def test_build_model_keeps_native_loss_when_selected(monkeypatch):
    model = FakeNativeModel()
    monkeypatch.setattr(registry, "build_model_from_values", lambda *args, **kwargs: model)

    assert registry.build_model(_args("unetformer")).uses_native_loss is True


def test_build_model_disables_native_loss_for_explicit_dense_override(monkeypatch):
    model = FakeNativeModel()
    monkeypatch.setattr(registry, "build_model_from_values", lambda *args, **kwargs: model)

    assert registry.build_model(_args("ce_dice")).uses_native_loss is False
