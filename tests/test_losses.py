import pytest
import torch

from oemseg.losses.registry import build_loss, resolve_loss_name


def test_registered_dense_losses_have_finite_backward():
    for name in ("ce", "dice", "ce_dice", "ce-dice", "cedice"):
        logits = torch.randn(2, 9, 16, 16, requires_grad=True)
        target = torch.randint(0, 9, (2, 16, 16))
        loss = build_loss(name)(logits, target)
        assert torch.isfinite(loss)
        loss.backward()


def test_auto_loss_is_model_specific():
    expected = {
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
    assert {model: resolve_loss_name("auto", model) for model in expected} == expected


def test_unetformer_and_repstdc_allow_explicit_dense_loss_override():
    assert resolve_loss_name("ce_dice", "unetformer") == "ce_dice"
    assert resolve_loss_name("soft_ce_dice", "unetformer") == "soft_ce_dice"
    assert resolve_loss_name("ce", "repstdc") == "ce"


def test_mask2former_rejects_dense_loss_override():
    assert resolve_loss_name("mask2former", "mask2former") == "mask2former"
    with pytest.raises(ValueError):
        resolve_loss_name("ce_dice", "mask2former")


def test_model_specific_loss_cannot_be_used_on_the_wrong_model():
    for loss, model in (
        ("unetformer", "unet"),
        ("soft_ce_dice", "unet"),
        ("repstdc", "segnext"),
        ("mask2former", "segformer"),
    ):
        with pytest.raises(ValueError):
            resolve_loss_name(loss, model)
