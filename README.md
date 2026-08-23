# Modular OpenEarthMap semantic segmentation

A compact research framework for interchangeable semantic-segmentation models on OpenEarthMap. All models share the same data pipeline, losses, optimizer/scheduler factories, evaluator, checkpointing, error analysis, W&B tracking, and single-/multi-GPU training engine.

OpenEarthMap's official `train.txt` (3000 labeled images) is split deterministically into **2700 train + 300 internal validation** images by default. The public `val.txt` (500 labeled images) is treated as the reported paper **test** set. The official 1500-image benchmark test is kept only for optional submission/export because public masks are unavailable.

## Project structure

```text
OEM_Segmentation/
├── train.py                    # thin training entry point
├── oemseg/
│   ├── config.py               # experiment CLI
│   ├── data/                   # dataset + DataLoaders
│   ├── models/                 # model registry/adapters
│   ├── losses/                 # CE, Dice, CE + Dice
│   ├── optimizers/             # Adam / AdamW
│   ├── schedulers/             # LR + evaluation scheduling
│   ├── metrics/                # segmentation metrics
│   ├── engine/                 # trainer/evaluator/checkpoints/error analysis
│   └── utils/                  # logging, seeding, TTA, notifications
├── scripts/
│   ├── setup_env.sh            # reproducible CUDA/Python environment setup
│   ├── launch.py               # N-GPU launcher
│   └── setup_dataset.sh        # reproducible OEM dataset bootstrap
├── notebooks/
│   └── kaggle_multi_gpu.ipynb
└── tests/
```

Every model adapter returns logits shaped `[B, 9, H, W]` and exposes backbone/main parameter groups, so the trainer contains no model-specific training loop.

## Environment

PyTorch/torchvision are intentionally omitted from `requirements.txt` so pip cannot silently replace the CUDA build supplied by the machine/Kaggle image.

Use the setup script from the repository root:

```bash
bash scripts/setup_env.sh
```

The script:

1. verifies that the existing PyTorch build can see CUDA,
2. installs normal project dependencies,
3. installs the pinned `mamba-ssm==2.2.6.post3` separately with `--no-build-isolation`, and
4. verifies imports/versions after setup.

`MAX_JOBS` can be used to limit Mamba compilation parallelism on memory-constrained machines:

```bash
MAX_JOBS=2 bash scripts/setup_env.sh
```

The project has been developed around CUDA-enabled PyTorch and these pinned/limited dependencies: `transformers==4.50.0`, `timm==1.0.15`, `einops==0.8.1`, and `mamba-ssm==2.2.6.post3`.

## Dataset

```text
datasets/OpenEarthMap/
├── OpenEarthMap_wo_xBD/
├── xBD_huggingface/
└── OpenEarthMap/          # 3000 official-train + 500 labeled reported-test images
```

Prepare the dataset once after cloning:

```bash
./scripts/setup_dataset.sh
```

The dataset script downloads the official OpenEarthMap package, verifies it, fetches only the xBD images referenced by the split metadata, builds the final layout, and verifies the result. Re-running it is safe.

Verify an already prepared dataset without downloading/rebuilding:

```bash
./scripts/setup_dataset.sh --verify-only
```

Split interpretation used by this project:

- official `train.txt`: 3000 labeled images, internally split for training/validation;
- official `val.txt`: 500 labeled images, reported as `test_*` metrics;
- official `test.txt`: 1500 benchmark images without public masks, not used for paper metrics.

## Supported components

| Component | CLI names |
|---|---|
| Model | `unet`, `unetpp`, `segformer`, `mambavision` |
| Loss | `ce`, `dice`, `ce_dice` |
| Optimizer | `adam`, `adamw` |
| MambaVision decoder | `upernet` |

### Default experiment

```bash
python train.py
```

Defaults: UNet++/ResNet18 with ImageNet initialization, 9 classes, 1024×1024 input, batch size 2, AdamW, CE + Dice, **45 epochs**, five warmup epochs, polynomial LR decay, fp16, decoder LR `6e-4`, backbone LR `6e-5`, and weight decay `0.01`.

Examples:

```bash
# U-Net
python train.py --model unet --model-variant resnet18

# SegFormer-B0
python train.py --model segformer --model-variant b0

# MambaVision-T + compact UPerNet
python train.py \
  --model mambavision \
  --model-variant tiny \
  --decoder upernet \
  --decoder-channels 512 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --grad-accumulation 2
```

MambaVision uses NVIDIA's official `MambaVision-T-1K` implementation/weights through Hugging Face at the revision pinned in `oemseg/models/mambavision.py` and feeds its four feature maps into the local UPerNet decoder. Use `--no-pretrained` when weights must not be downloaded.

## Default evaluation and checkpoint schedule

The default 45-epoch research run now avoids expensive evaluation during the early optimization phase:

```text
epochs  1-30 : train only
epoch  31     : train + val
epoch  32     : train + val
epoch  33     : train + val + test
epoch  34     : train + val
...
epoch  36     : train + val + test
...
epoch  45     : train + val + test   (final epoch is always val + test)
```

Equivalent defaults:

```text
--epochs 45
--eval-start-epoch 30
--eval-every 1
--test-every-validations 3
```

`--eval-start-epoch 30` means **30 complete train-only epochs**; validation starts at epoch 31. Every third validation also evaluates the reported test split. The final configured epoch always runs both validation and test even if it does not land on the normal interval.

Model selection remains validation-only: `best_val_miou.pt` is selected from internal-validation mIoU. Reported `test_*` metrics never select a checkpoint or drive early stopping.

The older fraction-based behavior remains available explicitly for compatibility. When `--eval-start-fraction` is supplied, validation and test run together using that legacy schedule.

The default TTA averages scales `0.75`, `1.0`, and `1.25` with original/horizontal/vertical predictions. For quick smoke/exploratory runs, use `--tta-scales 1.0 --no-tta-flips`.

## Single GPU and N-GPU launch modes

Normal single-GPU training remains unchanged:

```bash
python train.py --model unetpp
```

`scripts/launch.py` validates requested physical GPU IDs and model names before launching work. `--gpus` accepts comma-separated IDs or `all`.

### N GPUs -> one model

All selected GPUs cooperate on one distributed training job through Hugging Face Accelerate:

```bash
python scripts/launch.py distributed \
  --gpus 0,1,2,3 \
  --model mambavision \
  -- --batch-size 1 --grad-accumulation 2 --wandb
```

With one selected GPU the launcher simply calls the normal `train.py`; with multiple GPUs it launches one Accelerate process per selected GPU.

### N GPUs -> N independent models

Run one independent experiment per GPU:

```bash
python scripts/launch.py parallel \
  --gpus 0,1,2,3 \
  --models unet,unetpp,segformer,mambavision \
  -- --epochs 45 --wandb
```

Parallel mode intentionally requires exactly one model per selected GPU. Invalid GPU IDs, duplicate IDs, unknown models, or mismatched GPU/model counts fail before training starts. Shared training arguments are forwarded after `--` directly to `train.py`.

Inspect a launch without starting training:

```bash
python scripts/launch.py distributed --gpus 0,1 --model unetpp --dry-run
python scripts/launch.py parallel --gpus 0,1 --models unetpp,segformer --dry-run
```

The Kaggle notebook uses the same launcher rather than maintaining a second training implementation.

## Throughput settings

The shared engine already uses:

- fp16 mixed precision by default;
- high float32 matmul precision;
- cuDNN benchmarking for the fixed-size CUDA workload;
- pinned-memory DataLoaders;
- persistent DataLoader workers when `--workers > 0`;
- non-blocking Accelerate transfers;
- distributed sampling/gathering through Accelerate.

`--channels-last` remains opt-in because its gain depends on the architecture. `torch.compile` is intentionally not enabled globally; benchmark it per model before using it in a paper run.

Skipping the first 30 epochs of validation/test and evaluating the test split only every third validation also removes substantial evaluation/TTA overhead without changing the optimization loop.

## W&B reproducibility artifacts

W&B remains optional:

```bash
python train.py --wandb --wandb-project oem-segmentation
python train.py --wandb --wandb-mode offline --smoke
```

When W&B is enabled, each run logs metrics plus a reproducibility artifact containing the actual training entry point, run `config.json`, requirements, `oemseg/` source (models/data/losses/trainer/etc.), scripts, and Kaggle launcher notebook. This keeps the code/config used by an experiment attached to the run rather than relying on the current repository state months later.

## End-of-run email

Email notification uses Python's standard library; no extra dependency is required. The SMTP password is deliberately read from an environment variable so it is not stored in CLI history, W&B config, or checkpoints.

```bash
export SMTP_PASSWORD='your-app-password-or-smtp-password'

python train.py \
  --notify-email recipient@example.com \
  --smtp-user sender@example.com \
  --smtp-host smtp.gmail.com \
  --smtp-port 587
```

Change the password variable name with `--smtp-password-env`. STARTTLS is used by default; `--smtp-no-starttls` exists for SMTP servers that do not use it.

The completion mail includes the run name, best validation mIoU/epoch, best observed test mIoU/epoch, final test mIoU, and output directory. Test results are informational only and never select model weights.

## Outputs and checkpoints

Each run writes under `outputs/<run-name>/`:

```text
config.json
train.log
metrics.jsonl
sample_scores.jsonl
bad_predictions_val.tsv
bad_predictions_test.tsv
bad_predictions_val_best.tsv
bad_predictions_test_at_best_val.tsv
splits/{train,val,test}.txt
last.pt
best_train_loss.pt
best_val_miou.pt
```

`sample_scores.jsonl` stores compact filename/region/loss/OA/mIoU/worst-class metadata rather than image tensors. The TSV files keep the worst samples for failure analysis.

## Verification

CPU/unit checks where dependencies are available:

```bash
python -m compileall -q train.py oemseg tests scripts
python -m pytest -q
```

A cheap real-GPU smoke run evaluates only one train batch and one validation/test batch:

```bash
python train.py \
  --model unetpp \
  --no-pretrained \
  --smoke \
  --tta-scales 1.0 \
  --no-tta-flips
```

The same `--smoke` arguments can be forwarded through either N-GPU launch mode for Kaggle verification.

## Extending the framework

To add a model, create an adapter in `oemseg/models/`, subclass `SegmentationModelAdapter`, return full-resolution 9-class logits, expose `backbone`, and register the builder with `@register_model(...)`.

To add a loss, implement an `nn.Module` and register it in the existing loss registry. Extend optimizer/scheduler factories in place rather than creating model-specific training loops.
