# Modular OpenEarthMap semantic segmentation

A compact research framework for interchangeable semantic-segmentation models on OpenEarthMap. All models share the same data split, optimizer/scheduler plumbing, evaluator, checkpointing, error analysis, W&B tracking, and single-/multi-GPU training engine. `--loss auto` is model-aware, so architectures with native auxiliary/set-prediction objectives are not silently forced through CE+Dice.

OpenEarthMap's official `train.txt` (**3000 labeled images**) is used in full by the accuracy-first default. The public `val.txt` (500 labeled images) is treated as the reported paper **test** set, matching the PyramidMamba paper because the official 1500-image benchmark test masks are not public. By default, checkpoint selection therefore uses minimum training loss; an internal validation split remains available explicitly with `--val-fraction > 0`.

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
│   ├── setup_unetformer.sh     # pin GeoSeg + verify official Swin-B weights
│   ├── setup_openmmlab_baselines.sh # isolated env for SegNeXt/RepSTDC
│   ├── launch.py               # N-GPU launcher
│   └── setup_dataset.sh        # reproducible OEM dataset bootstrap
├── notebooks/
│   └── kaggle_multi_gpu.ipynb
└── tests/
```

Every model adapter returns evaluation logits shaped `[B, 9, H, W]` and exposes backbone/main parameter groups. During optimization, UNetFormer and Mask2Former ask the same trainer to use their upstream-native objectives; all other registered dense-logit baselines use the shared loss registry.

## Environment

PyTorch/torchvision are intentionally omitted from `requirements.txt` so pip cannot silently replace the CUDA build supplied by the machine/Kaggle image.

Use the setup script from the repository root:

```bash
bash scripts/setup_env.sh
```

The script:

1. pins the upstream GeoSeg source used by the model adapters under gitignored `.vendor/GeoSeg`,
2. verifies that the existing PyTorch build can see CUDA,
3. installs normal project dependencies,
4. detects the current Python/PyTorch/CUDA/CXX11-ABI stack and installs the matching prebuilt `mamba-ssm==2.3.2.post1` wheel directly from the official Mamba GitHub release, refusing to fall back to a long source compilation, and
5. verifies imports plus a real CUDA `selective_scan_fn` smoke test.

If the Python environment is already prepared, run `bash scripts/setup_unetformer.sh` before UNetFormer. It pins GeoSeg commit `9453fe48209c4626b29e35e61bab93b61212c4b1` and downloads/verifies GeoSeg's official `stseg_base.pth` for the paper-comparable Swin-B variant. Mamba setup uses the same pinned source with `--source-only`, so it does not download the 486 MB Swin weight unnecessarily.

The verified Kaggle 2x T4 environment uses Python 3.12, PyTorch 2.10.0+cu128, CUDA 12.8, CXX11 ABI enabled, and the `cu12torch2.10cxx11abiTRUE-cp312` Mamba wheel. The project otherwise keeps the existing pinned/limited dependencies such as `transformers==4.50.0`, `timm==1.0.15`, and `einops==0.8.1`.

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

- official `train.txt`: 3000 labeled images, all used for training by default; optional internal validation is opt-in;
- official `val.txt`: 500 labeled images, reported as `test_*` metrics;
- official `test.txt`: 1500 benchmark images without public masks, not used for paper metrics.

## Supported components

| Component | CLI names |
|---|---|
| Model | `unet`, `unetpp`, `unetformer`, `segformer`, `segnext`, `repstdc`, `mambavision`, `pyramidmamba`, `mask2former` |
| Loss | `auto`, `ce`, `dice`, `ce_dice`, `soft_ce_dice`, `unetformer`, `repstdc`, `mask2former` |
| Optimizer | `adam`, `adamw` |
| MambaVision decoder | `upernet` |

`--loss auto` resolves to the model-specific benchmark objective:

| Model | Effective loss |
|---|---|
| U-Net / U-Net++ | `ce_dice` |
| UNetFormer Swin-B | `unetformer` = GeoSeg soft-CE + Dice |
| UNetFormer ResNet18 | `unetformer` = main soft-CE + Dice + `0.4 ×` auxiliary soft-CE |
| SegFormer-B0 | `ce` |
| SegNeXt-T | `ce` |
| RepSTDC-CA | `repstdc` = main OHEM CE + 2 auxiliary OHEM CE + boundary CE + Dice |
| MambaVision-T | `ce_dice` |
| PyramidMamba | `ce_dice` |
| Mask2Former | `mask2former` = native Hungarian class/mask/Dice objective |

For Swin-B UNetFormer, an explicit dense override such as `--loss soft_ce_dice` is allowed. UNetFormer/ResNet18, RepSTDC, and Mask2Former reject incompatible dense overrides because they would drop auxiliary, boundary, or matching supervision.

### Default experiment

```bash
python train.py
```

Defaults: UNet++/ResNet18 with ImageNet initialization, 9 classes, 1024×1024 input, batch size 2, AdamW, CE + Dice, **45 epochs**, five warmup epochs, polynomial LR decay, full precision, decoder LR `6e-4`, backbone LR `6e-5`, weight decay `0.01`, all 3000 official-train images, and patience 5 on minimum training loss.

Examples:

```bash
# U-Net
python train.py --model unet --model-variant resnet18

# UNetFormer / PyramidMamba Table-8 comparable Swin-B recipe.
# `unetformer` defaults to GeoSeg FTUNetFormer with the official stseg_base.pth.
bash scripts/setup_unetformer.sh
python train.py --model unetformer

# Optional lightweight official GeoSeg UNetFormer: SWSL-ResNet18 + auxiliary CE.
python train.py --model unetformer --model-variant resnet18

# PyramidMamba from the same pinned upstream GeoSeg checkout
bash scripts/setup_unetformer.sh
python train.py --model pyramidmamba --model-variant swin_base_patch4_window12_384.ms_in22k_ft_in1k

# SegFormer-B0
python train.py --model segformer --model-variant b0

# SegNeXt-T and RepSTDC-CA reuse OpenMMLab / official RepSTDC code in an isolated env.
bash scripts/setup_openmmlab_baselines.sh
~/miniconda3/bin/conda run -n oem-openmmlab python train.py --model segnext --model-variant tiny
~/miniconda3/bin/conda run -n oem-openmmlab python train.py --model repstdc --model-variant stdc1-ca

# Mask2Former-Swin-Tiny reuses the existing Transformers dependency.
python train.py --model mask2former --model-variant swin-tiny

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

The paper baseline order used by this repository is **U-Net → UNetFormer → SegFormer-B0 → SegNeXt-T → RepSTDC-CA → MambaVision-T → PyramidMamba → Mask2Former-Swin-Tiny → Our Model**. SegNeXt-T uses the official MSCAN-T + LightHamHead settings from MMSegmentation. RepSTDC-CA reuses the pinned official `mmseg_geo` architecture but changes only the input/output contract needed by this benchmark: RGB input and all 9 OpenEarthMap classes including background. Mask2Former keeps its own Hungarian matching/class-mask objective instead of replacing the method with dense CE+Dice.

### Accuracy-first PyramidMamba-paper reproduction

The PyramidMamba OpenEarthMap protocol is the reference for these runs: all 3000 official training images, 1024×1024 inputs, AdamW, Poly LR (`power=0.9`), LR `6e-4` / encoder LR `6e-5`, weight decay `0.01`, batch size 2, 45 epochs, five warmup epochs, early stopping, and multi-scale/flip TTA. Two GPUs still train **one model**, but each process uses batch 1 and synchronized BatchNorm so the global batch/statistics stay at 2 instead of silently changing the experiment to batch 4.

```bash
python scripts/launch.py distributed \
  --gpus 0,1 \
  --model unet \
  -- \
  --batch-size 1 \
  --grad-accumulation 1 \
  --val-fraction 0 \
  --patience 5 \
  --mixed-precision no \
  --wandb
```

Replace only `unet` with the requested model. In no-validation mode, the selected checkpoint is `best_train_loss.pt`; the 500-image public OpenEarthMap validation split is evaluated as the reported `test_*` split and never selects weights. The paper reports **60.4% mIoU for U-Net**, **68.0% for UNetFormer/Swin-B**, and **70.8% for PyramidMamba/Swin-B**. `--model unetformer` now selects the pinned official GeoSeg FTUNetFormer/Swin-B path by default; use `--model-variant resnet18` only when intentionally benchmarking the lighter SWSL-ResNet18 model.


### Headless Kaggle T4x2 + automatic W&B sync

On the configured PC, `scripts/kaggle_pipeline.py` replaces the manual dataset-page → New Notebook → Add-ons → paste-code → Save Version flow. It generates a private Kaggle notebook through the Kaggle CLI, attaches `duy18102004/oem-dataset`, requests `NvidiaTeslaT4` with internet enabled, waits for the version to finish, downloads the outputs, and then runs `wandb sync` locally. Kaggle itself uses W&B **offline** mode, so the W&B API key never appears in the notebook or Kaggle metadata.

The PC-side Kaggle/W&B client lives outside the repository at `~/.local/share/oem-kaggle-client`; the default authorized token is read from `~/.config/kaggle/accounts/account_1/access_token`. Generated notebooks, downloaded outputs, watcher logs, and state JSON are stored under `~/.local/state/oem-kaggle`, also outside the repository.

Run a short integration version:

```bash
~/.local/share/oem-kaggle-client/bin/python scripts/kaggle_pipeline.py \
  --model unet \
  --smoke \
  --detach
```

Run the full 45-epoch paper recipe by changing only the model name and omitting `--smoke`:

```bash
~/.local/share/oem-kaggle-client/bin/python scripts/kaggle_pipeline.py \
  --model unetformer \
  --detach
```

`--detach` returns immediately while a local watcher continues `push → status polling → output download → W&B sync`. The final `state.json` is marked `SYNCED` only after W&B upload succeeds; Kaggle failures/cancellations do not get reported as successful syncs. Supported model names are `unet`, `unetformer`, `segformer`, `segnext`, `repstdc`, `mambavision`, `pyramidmamba`, and `mask2former`.

## Default evaluation and checkpoint schedule

The 45-epoch research schedule avoids expensive evaluation during the early optimization phase. The table below shows the full schedule when internal validation is enabled; with the default `--val-fraction 0`, validation rows are simply skipped while the reported test checkpoints remain informational only:

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

Checkpoint selection has two modes, both using the same trainer:

```bash
# Validation mode: 80% train / 20% internal validation; fixed reported test remains unchanged.
python train.py --val-fraction 0.2

# No-validation mode (default): all 3000 official-train images train the model.
# The selected checkpoint is the minimum training-loss checkpoint.
python train.py --val-fraction 0 --patience 5
```

With `--val-fraction > 0`, `best_val_miou.pt` is selected by internal-validation mIoU and `--patience` counts validation evaluations without improvement. With `--val-fraction 0`, `best_train_loss.pt` is selected by minimum training loss and `--patience` counts consecutive training epochs without a new minimum. `--patience 0` disables early stopping in either mode. The reported `test_*` metrics never select a checkpoint or drive early stopping.

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

## Official Q1-paper sidecars

PyramidMamba fits the local adapter contract, so it uses the same OpenEarthMap split, 9-class logits, losses, W&B logging, checkpointing, and evaluator as the other native models. The existing `scripts/setup_unetformer.sh` checkout is intentionally shared because the pinned GeoSeg revision contains both UNetFormer and PyramidMamba.

GeoSA-BaSA and HG-RSOVSSeg retain their **official upstream protocol** because their domain-generalization/open-vocabulary tasks are not the same closed-set experiment. RepSTDC now has two paths: `train.py --model repstdc` is the apples-to-apples 9-class baseline, while `scripts/paper_models.py train repstdc` remains the untouched official 512×512 reproduction path. OpenMMLab dependencies stay out of the main `requirements.txt` and live in the isolated `oem-openmmlab` environment.

```bash
# Clone all three official repositories at their exact pinned revisions.
python scripts/paper_models.py setup all

# Inspect commands without starting a job.
python scripts/paper_models.py train geosa_basa --dry-run
python scripts/paper_models.py train hg_rsovsseg --dry-run
python scripts/paper_models.py train repstdc --dry-run

# RepSTDC's official direct-OpenEarthMap config.
python scripts/paper_models.py train repstdc --dry-run -- --work-dir runs/repstdc
# config/repstdc/repstdc-ca_512x512_80k_oem.py

# Evaluation examples; replace the checkpoint paths with real files.
python scripts/paper_models.py eval geosa_basa --checkpoint checkpoints/whumix_dinov2_geosa_basa.pth --dry-run
python scripts/paper_models.py eval hg_rsovsseg --checkpoint result/HG-RSOVSSeg/OpenEarthMap/iter_80000.pth --dry-run
python scripts/paper_models.py eval repstdc --checkpoint work_dir/repstdc/latest.pth --dry-run
```

The wrapper deliberately does not hide dataset conversion. GeoSA-BaSA expects its official preprocessing step, for example `python tools/convert_datasets/preprocess_oem.py --oem-root /path/to/oem`; HG-RSOVSSeg expects its documented `data/OpenEarthMap_512` layout; RepSTDC uses its own MMSegmentation dataset/config layout. Reuse or symlink the project's OEM files where compatible rather than duplicating them.

These sidecar results are **not directly comparable** to the local closed-set 1024×1024 benchmark unless you deliberately align the protocol. GeoSA-BaSA is a domain-generalization experiment, HG-RSOVSSeg is open-vocabulary/cross-dataset segmentation, and RepSTDC's published OEM recipe is a 512×512 real-time MMSegmentation setup. For the paper comparison table, use the registered `train.py --model repstdc` path instead of the sidecar reproduction command.

## Throughput settings

The shared engine already uses:

- full precision by default for accuracy-first paper reproduction (`--mixed-precision fp16` remains opt-in);
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

When W&B is enabled, each run logs epoch metrics plus two artifacts:

- a reproducibility artifact containing the actual training entry point, run `config.json`, requirements, `oemseg/` source, scripts, and Kaggle launcher notebook;
- a `model` artifact containing the **selected checkpoint**, `best_checkpoint_summary.json`, complete selected-checkpoint per-sample score tables, and below-mean error-analysis TSVs. Validation mode contains both `below_mean_val.tsv` and `below_mean_test.tsv`; no-validation mode contains test analysis only.

The W&B Summary records `best/epoch`, `best/selection_mode`, `best/selection_metric`, selected-checkpoint train loss, and the complete selected-checkpoint validation/test metric set when applicable: loss, OA, mIoU, macro F1/precision/recall, and per-class IoU/F1/precision/recall. Test metrics stay informational and never choose weights.

Best-checkpoint qualitative panels convert the single-channel class-ID target/prediction masks to RGB using the OpenEarthMap class palette before logging. W&B also receives `best_checkpoint/legend`, which maps all nine IDs/colors (`background`, `bareland`, `rangeland`, `developed`, `road`, `tree`, `water`, `agriculture`, `building`). The same legend and rendered grids are stored under the run's `visualizations/` directory and included in the model artifact.

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
bad_predictions_val_best.tsv              # validation mode only
bad_predictions_test_at_best_val.tsv       # validation mode when sampled during training
best_checkpoint_summary.json
best_checkpoint_val_scores.tsv             # validation mode only
best_checkpoint_test_scores.tsv
below_mean_val.tsv                         # validation mode only; selected checkpoint
below_mean_test.tsv                        # selected checkpoint
visualizations/{legend,best_*_examples}.png # when W&B visualization runs
splits/{train,val,test}.txt
last.pt
best_train_loss.pt
best_val_miou.pt                           # validation mode only
```

`sample_scores.jsonl` stores compact filename/region/loss/OA/mIoU/worst-class metadata rather than image tensors. `below_mean_<split>.tsv` is the selected-checkpoint error analysis: every listed sample has sample mIoU below that split's mean sample mIoU. `best_checkpoint_summary.json` contains the complete selected-checkpoint metric dictionaries, including all per-class metrics.

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


A real one-epoch no-validation W&B verification run (epoch 1 must be selected) is:

```bash
python train.py \
  --model unet \
  --epochs 1 \
  --val-fraction 0 \
  --wandb \
  --tta-scales 1.0 \
  --no-tta-flips
```


## Extending the framework

To add a model, create an adapter in `oemseg/models/`, subclass `SegmentationModelAdapter`, return full-resolution 9-class logits, expose `backbone`, and register the builder with `@register_model(...)`.

To add a loss, implement an `nn.Module` and register it in the existing loss registry. Extend optimizer/scheduler factories in place rather than creating model-specific training loops.
