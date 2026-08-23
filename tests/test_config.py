import pytest

from oemseg.config import parse_args
from oemseg.models.registry import available_models, normalize_name


def test_default_cli_preserves_training_behavior_with_staged_evaluation():
    args = parse_args([])
    assert args.model == "unetpp"
    assert args.model_variant == "resnet18"
    assert args.loss == "ce_dice"
    assert args.optimizer == "adamw"
    assert args.epochs == 45
    assert args.grad_accumulation == 1
    assert args.internal_val_fraction == 0.1
    assert args.eval_start_epoch == 30
    assert args.eval_start_fraction is None
    assert args.eval_every == 1
    assert args.test_every_validations == 3
    assert args.bad_predict_top_n == 30
    assert args.mixed_precision == "fp16"
    assert args.channels_last is False


def test_val_fraction_accepts_new_and_legacy_flag_names():
    assert parse_args(["--val-fraction", "0.2"]).internal_val_fraction == 0.2
    assert parse_args(["--internal-val-fraction", "0.25"]).internal_val_fraction == 0.25


def test_legacy_encoder_maps_to_model_variant():
    args = parse_args(["--encoder", "resnet34", "--encoder-weights", "none"])
    assert args.model == "unetpp"
    assert args.model_variant == "resnet34"
    assert args.pretrained is False


def test_component_names_are_normalized():
    assert normalize_name("CE-Dice") == "ce_dice"
    assert {"unet", "unetpp"} <= set(available_models())


def test_unet_uses_resnet18_default_variant():
    args = parse_args(["--model", "unet", "--no-pretrained"])
    assert args.model_variant == "resnet18"
    assert args.pretrained is False


def test_invalid_gradient_accumulation_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--grad-accumulation", "0"])


def test_channels_last_is_opt_in():
    assert parse_args(["--channels-last"]).channels_last is True
