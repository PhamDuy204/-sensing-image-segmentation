# Modular OpenEarthMap semantic segmentation

This project trains interchangeable semantic-segmentation models on OpenEarthMap through one shared PyTorch engine. The default command remains backward-compatible with the completed UNet++/ResNet18 experiment.

OpenEarthMap's official `train.txt` (3000 labeled images) is split deterministically into **2700 train + 300 internal validation** images by default. The public `val.txt` (500 labeled images) is the reported paper **test** set and therefore uses the `test_*` prefix. The official 1500-image benchmark test is reserved for optional submission/export because its labels are not available locally. Reported-test metrics are logged for analysis but never drive checkpoint selection or early stopping.

## Project structure

```text
OEM_Segmentation/
├── train.py                    # thin CLI entry point
├── oemseg/
│   ├── config.py               # argparse and legacy option mapping
│   ├── constants.py            # nine OEM classes
│   ├── data/                   # dataset and DataLoader construction
│   ├── models/                 # registry and model adapters
│   │   ├── unetpp.py
│   │   ├── segformer.py
│   │   ├── mambavision.py
│   │   └── upernet.py
│   ├── losses/                 # CE, Dice, and CE + Dice
│   ├── optimizers/             # Adam and AdamW
│   ├── schedulers/             # warmup + polynomial decay
│   ├── metrics/                # confusion-matrix metrics
│   ├── engine/                 # trainer, evaluator, checkpoints
│   └── utils/                  # logging, seeding, and TTA
├── scripts/                    # dataset preparation helpers
└── tests/                      # focused unit/integration tests
```

Every model adapter returns logits shaped `[B, 9, H, W]` and exposes separate backbone/main parameter groups. The trainer therefore has no model-specific branches.

## Environment

Use only the existing `work-env` conda environment. PyTorch and torchvision are intentionally omitted from `requirements.txt` so pip cannot replace the verified CUDA builds.

```bash
conda activate work-env
python -m pip install --no-build-isolation -r requirements.txt
```

The verified environment includes:

```text
torch 2.13.0+cu132
torchvision 0.28.0+cu132
segmentation-models-pytorch 0.5.0
transformers 4.50.0
timm 1.0.15
einops 0.8.1
mamba-ssm 2.2.6.post3
rasterio 1.4.4
wandb 0.28.1
pytest 8.4.2
```

`mamba-ssm` must be installed with `--no-build-isolation` so its extension compiles against the already installed CUDA-enabled PyTorch. Version `2.2.6.post3` is used because its official build script supports CUDA 13 while retaining the Selective Scan API needed by MambaVision.

## Dataset

```text
datasets/OpenEarthMap/
├── OpenEarthMap_wo_xBD/   # original incomplete package
├── xBD_huggingface/       # selectively extracted required xBD images
└── OpenEarthMap/          # 3000 official-train + 500 labeled reported-test images
```

After cloning, prepare the dataset with one command (run it from the project environment):

```bash
./scripts/setup_dataset.sh
```

The script downloads the official 9.1 GB `OpenEarthMap.zip` from Zenodo, verifies its published MD5, keeps the distributed package as `OpenEarthMap_wo_xBD`, selectively downloads only the 1,162 xBD RGB images referenced by `xbd_files.csv`, builds `OpenEarthMap`, and verifies the final split structure. Re-running it is safe; an already valid prepared dataset exits through the verification path.

To verify an existing dataset without downloading or rebuilding anything:

```bash
./scripts/setup_dataset.sh --verify-only
```

The project intentionally interprets the splits as follows: official `train.txt` contains 3000 labeled images and is split internally for training/validation; official `val.txt` contains 500 labeled images and is used as the reported test set; official `test.txt` contains 1500 images without public masks and is kept only for benchmark submission/export.

## Supported components

| Component | CLI names |
|---|---|
| Model | `unet`, `unetpp`, `segformer`, `mambavision` |
| Loss | `ce`, `dice`, `ce_dice` (`ce-dice` and `cedice` aliases) |
| Optimizer | `adam`, `adamw` |
| MambaVision decoder | `upernet` |

### Default UNet++ experiment

```bash
python train.py
```

This still selects UNet++ with an ImageNet-pretrained ResNet18 encoder, 9 classes, 1024×1024 inputs, batch size 2, AdamW, CE + Dice, 45 epochs, 5 warmup epochs, polynomial decay, decoder LR `6e-4`, backbone LR `6e-5`, and weight decay `0.01`.

A custom UNet++ run:

```bash
python train.py \
  --model unetpp \
  --model-variant resnet34 \
  --loss ce_dice \
  --optimizer adamw \
  --epochs 30
```

Legacy `--encoder` and `--encoder-weights` arguments remain accepted for UNet++.

### U-Net

```bash
python train.py \
  --model unet \
  --model-variant resnet18 \
  --loss ce_dice \
  --optimizer adamw \
  --epochs 45
```

U-Net uses `segmentation_models_pytorch.Unet` with an ImageNet-pretrained encoder by default and the same shared 9-class training pipeline.

### SegFormer-B0

```bash
python train.py \
  --model segformer \
  --model-variant b0 \
  --loss ce_dice \
  --optimizer adamw \
  --epochs 30
```

The adapter uses Hugging Face `SegformerForSemanticSegmentation`, initializes its MiT-B0 encoder from `nvidia/mit-b0`, keeps background label `0`, creates a 9-class decode head, and resizes logits to the input resolution.

### MambaVision-T + UPerNet

```bash
python train.py \
  --model mambavision \
  --model-variant tiny \
  --decoder upernet \
  --decoder-channels 512 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --grad-accumulation 2
```

The adapter loads NVIDIA's official `MambaVision-T-1K` implementation and weights through Hugging Face at pinned revision `b1de77e17599566d98efb701c0231b1095dc3a67`. Its four feature maps have channels `[80, 160, 320, 640]` and feed the local compact UPerNet decoder. `--no-pretrained` constructs the same official architecture from its pinned remote config without loading weights.

On the available RTX 3060 12 GB, one real OEM train/eval smoke batch at 1024² with UPerNet channels 512 observed about 3555 MiB of process VRAM. This is a smoke-test measurement, not a guaranteed peak for a long training run.

MambaVision source is governed by NVIDIA's non-commercial source license, and NVIDIA's pretrained weights are published under CC-BY-NC-SA-4.0. Review the official repository and model card before commercial use:

- https://github.com/NVlabs/MambaVision
- https://huggingface.co/nvidia/MambaVision-T-1K

## Research split, evaluation, and model selection

By default `--val-fraction 0.1` (legacy alias: `--internal-val-fraction`) performs a deterministic, region-aware split of official `train.txt`: 2700 optimization images and 300 internal-validation images with seed 42. Exact names are saved under `splits/train.txt`, `splits/val.txt`, and `splits/test.txt` in every run directory. Official `val.txt` remains untouched and is the 500-image reported test set.

Training uses random horizontal/vertical flips. **Validation and reported test are evaluated every epoch by default** (`--eval-start-fraction 0 --eval-every 1`) so their trends can be compared during research. `best_val_miou.pt` and early stopping use only internal-validation mIoU; `test_*` never participates in model selection. Legacy evaluation-frequency flags remain available for faster exploratory runs.

The default TTA averages scales `0.75`, `1.0`, and `1.25` with original, horizontal-flip, and vertical-flip predictions. These exact scales are an explicit project assumption because the reference paper states multi-scale TTA without publishing scale values.

Logged metrics are:

- `test_oa`: overall pixel accuracy
- `test_miou`: mean class IoU
- macro `test_f1`, `test_precision`, and `test_recall`
- `test_iou_<class>` for all nine classes

Internal validation uses the `val_*` prefix; official OEM `val.txt` uses `test_*`. The official 1500-image `test.txt` benchmark split is not part of these paper metrics.


## Single-GPU, multi-GPU, and Kaggle

The same trainer is used in every mode. Hugging Face Accelerate handles device placement, distributed sampling, gradient accumulation, mixed precision, compact metric gathering, and model unwrapping.

Single GPU (backward-compatible):

```bash
python train.py --model unetpp --val-fraction 0.1
```

Two or more GPUs from a shell/server:

```bash
accelerate launch --num_processes=2 train.py \
  --model unetpp --val-fraction 0.1 --mixed-precision fp16
```

For Kaggle, open `notebooks/kaggle_multi_gpu.ipynb`, enable two GPUs, fix `PROJECT_ROOT` and `DATA_ROOT`, and run it from a fresh kernel. The notebook calls `accelerate.notebook_launcher(..., num_processes=2)` and imports `run_training`; it contains no copy of the training/evaluation loop. Do not initialize CUDA in earlier notebook cells before `notebook_launcher`.

### Throughput settings

The shared engine enables high float32 matmul precision and cuDNN benchmarking on CUDA for the fixed-size image workload. DataLoaders already use pinned memory, persistent workers (when `--workers > 0`), and Accelerate handles non-blocking device placement/mixed precision. `--mixed-precision fp16` is the default. `--channels-last` is available as an **opt-in** experiment for CNN-heavy models; it is intentionally not enabled globally because benefits depend on the adapter/model. `torch.compile` is not enabled by default.

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

`sample_scores.jsonl` stores compact filename/region/loss/OA/mIoU/worst-class metadata, not image tensors or prediction masks. The TSV files keep the worst `--bad-predict-top-n` samples (30 by default), making failure analysis cheap in disk space. Checkpoints retain the legacy top-level keys `epoch`, `model`, `optimizer`, `scheduler`, and `args`, plus model/distributed metadata. Existing UNet++ checkpoints remain strict-load compatible.

W&B remains optional:

```bash
python train.py --wandb --wandb-project oem-segmentation
python train.py --wandb --wandb-mode offline --smoke
```

## Smoke tests and verification

```bash
python -m compileall -q train.py oemseg tests scripts
python -m pytest -q

python train.py --model unetpp --smoke \
  --run-name smoke-unetpp --tta-scales 1.0 --no-tta-flips

python train.py --model segformer --smoke \
  --run-name smoke-segformer --tta-scales 1.0 --no-tta-flips

python train.py --model mambavision --model-variant tiny \
  --decoder upernet --batch-size 1 --grad-accumulation 2 --smoke \
  --run-name smoke-mambavision --tta-scales 1.0 --no-tta-flips
```

## Extending the framework

To add a model, create an adapter in `oemseg/models/`, subclass `SegmentationModelAdapter`, return full-resolution 9-class logits, expose `backbone`, and register a builder with `@register_model(...)`.

To add a loss, implement an `nn.Module` with `forward(logits, target)` and add its normalized name to `oemseg/losses/registry.py`.

To add an optimizer or scheduler, extend only the corresponding factory. The trainer does not need modification when the stable interfaces are preserved.
