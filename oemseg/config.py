"""Command-line configuration for OEM segmentation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from oemseg.models.registry import available_models, normalize_name

LOSS_NAMES = ("ce", "dice", "ce_dice")
OPTIMIZER_NAMES = ("adam", "adamw")
MODEL_DEFAULT_VARIANTS = {
    "unet": "resnet18",
    "unetpp": "resnet18",
    "unetformer": "swin-b",
    "segformer": "b0",
    "mambavision": "tiny",
    "pyramidmamba": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "segnext": "tiny",
    "repstdc": "stdc1-ca",
    "mask2former": "swin-tiny",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train semantic-segmentation models on OpenEarthMap")
    parser.add_argument("--data-root", type=Path, default=Path("datasets/OpenEarthMap/OpenEarthMap"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model", default="unetpp")
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--decoder", default="upernet")
    parser.add_argument("--decoder-channels", type=int, default=512)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--encoder", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--encoder-weights", choices=["imagenet", "none"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--loss", default="ce_dice")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--encoder-lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument(
        "--eval-start-epoch",
        type=int,
        default=30,
        help="number of initial train-only epochs before validation begins",
    )
    parser.add_argument(
        "--eval-start-fraction",
        type=float,
        default=None,
        help="legacy fraction-based schedule; when set, validation and test run together",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--test-every-validations", type=int, default=3)
    parser.add_argument("--tta-scales", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    parser.add_argument("--no-tta-flips", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-fraction", "--internal-val-fraction", dest="internal_val_fraction", type=float, default=0.0,
        help="fraction of official train used for internal validation; 0 selects checkpoints by train loss",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--bad-predict-top-n", type=int, default=30)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sensing image segmentation")
    parser.add_argument("--wandb-entity", default="phamdinhanhduy-university-of-information-and-technology")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--notify-email", default=None, help="send an end-of-run summary to this address")
    parser.add_argument("--smtp-host", default="smtp.gmail.com")
    parser.add_argument("--smtp-port", type=int, default=587)
    parser.add_argument("--smtp-user", default=None)
    parser.add_argument("--smtp-from", default=None)
    parser.add_argument("--smtp-password-env", default="SMTP_PASSWORD")
    parser.add_argument("--smtp-no-starttls", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one train and one evaluation batch")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.model = normalize_name(args.model)
    if args.model not in available_models():
        parser.error(f"unknown model '{args.model}'; valid models: {', '.join(available_models())}")

    if args.encoder is not None:
        if args.model != "unetpp":
            parser.error("--encoder is only valid with --model unetpp")
        args.model_variant = args.encoder
    if args.model_variant is None:
        args.model_variant = MODEL_DEFAULT_VARIANTS[args.model]

    if args.encoder_weights is not None:
        if args.model != "unetpp":
            parser.error("--encoder-weights is only valid with --model unetpp")
        args.pretrained = args.encoder_weights != "none"
    if args.pretrained is None:
        args.pretrained = True

    args.loss = normalize_name(args.loss)
    if args.loss == "cedice":
        args.loss = "ce_dice"
    if args.loss not in LOSS_NAMES:
        parser.error(f"unknown loss '{args.loss}'; valid losses: {', '.join(LOSS_NAMES)}")
    args.optimizer = normalize_name(args.optimizer)
    if args.optimizer not in OPTIMIZER_NAMES:
        parser.error(f"unknown optimizer '{args.optimizer}'; valid optimizers: {', '.join(OPTIMIZER_NAMES)}")

    if args.decoder_channels < 1:
        parser.error("--decoder-channels must be >= 1")
    if args.grad_accumulation < 1:
        parser.error("--grad-accumulation must be >= 1")
    if args.bad_predict_top_n < 1:
        parser.error("--bad-predict-top-n must be >= 1")
    if not 0 <= args.internal_val_fraction < 1:
        parser.error("--internal-val-fraction must be in [0, 1)")
    if args.eval_start_epoch < 0 or args.eval_every < 1 or args.test_every_validations < 1:
        parser.error("evaluation requires --eval-start-epoch >= 0 and positive evaluation intervals")
    if args.eval_start_fraction is not None and not 0 <= args.eval_start_fraction <= 1:
        parser.error("--eval-start-fraction must be in [0, 1]")
    if args.smtp_port < 1:
        parser.error("--smtp-port must be >= 1")
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("epochs and batch sizes must be >= 1")
    return args
