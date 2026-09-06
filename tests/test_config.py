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
    assert args.internal_val_fraction == 0.0
    assert args.patience == 5
    assert args.eval_start_epoch == 30
    assert args.eval_start_fraction is None
    assert args.eval_every == 1
    assert args.test_every_validations == 3
    assert args.bad_predict_top_n == 30
    assert args.mixed_precision == "no"
    assert args.channels_last is False
    assert args.wandb_project == "sensing image segmentation"
    assert args.wandb_entity == "phamdinhanhduy-university-of-information-and-technology"


def test_auto_loss_resolves_to_model_specific_published_recipe():
    expected = {
        "unet": "ce_dice",
        "unetformer": "unetformer",
        "segformer": "ce",
        "segnext": "ce",
        "repstdc": "repstdc",
        "mambavision": "ce_dice",
        "pyramidmamba": "ce_dice",
        "mask2former": "mask2former",
    }
    for model, loss in expected.items():
        assert parse_args(["--model", model, "--no-pretrained"]).loss == loss


def test_unetformer_swin_b_accepts_explicit_dense_loss_override():
    assert parse_args(["--model", "unetformer", "--loss", "ce_dice", "--no-pretrained"]).loss == "ce_dice"
    assert parse_args(["--model", "unetformer", "--loss", "soft_ce_dice", "--no-pretrained"]).loss == "soft_ce_dice"


def test_native_auxiliary_models_reject_overrides_that_drop_supervision():
    with pytest.raises(SystemExit):
        parse_args(["--model", "unetformer", "--model-variant", "resnet18", "--loss", "ce_dice", "--no-pretrained"])
    with pytest.raises(SystemExit):
        parse_args(["--model", "repstdc", "--loss", "ce", "--no-pretrained"])


def test_mask2former_rejects_incompatible_loss_override():
    with pytest.raises(SystemExit):
        parse_args(["--model", "mask2former", "--loss", "ce_dice", "--no-pretrained"])


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


def test_unetformer_is_registered_with_paper_swin_b_default():
    assert "unetformer" in available_models()
    args = parse_args(["--model", "unetformer", "--no-pretrained"])
    assert args.model_variant == "swin-b"
    assert args.loss == "unetformer"


def test_pyramidmamba_is_registered_with_published_swin_default():
    assert "pyramidmamba" in available_models()
    args = parse_args(["--model", "pyramid-mamba", "--no-pretrained"])
    assert args.model == "pyramidmamba"
    assert args.model_variant == "swin_base_patch4_window12_384.ms_in22k_ft_in1k"
    assert args.pretrained is False
    assert args.loss == "ce_dice"


def test_new_paper_baselines_are_registered_with_published_defaults():
    assert {"segnext", "repstdc", "mask2former"} <= set(available_models())
    assert parse_args(["--model", "segnext", "--no-pretrained"]).model_variant == "tiny"
    assert parse_args(["--model", "repstdc", "--no-pretrained"]).model_variant == "stdc1-ca"
    assert parse_args(["--model", "mask2former", "--no-pretrained"]).model_variant == "swin-tiny"


def test_gradient_clipping_is_opt_in_and_validated():
    assert parse_args([]).max_grad_norm == 0.0
    assert parse_args(["--max-grad-norm", "0.01"]).max_grad_norm == pytest.approx(0.01)
    with pytest.raises(SystemExit):
        parse_args(["--max-grad-norm", "-1"])
