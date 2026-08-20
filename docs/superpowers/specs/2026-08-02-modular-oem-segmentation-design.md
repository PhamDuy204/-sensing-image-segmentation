# Modular OpenEarthMap Segmentation Framework Design

Date: 2026-08-02
Status: Approved architecture, pending implementation plan

## 1. Goal

Refactor the current single-file `train.py` pipeline into a small, modular semantic-segmentation framework that preserves the verified OpenEarthMap behavior while making models, losses, optimizers, schedulers, metrics, and training settings easy to swap from the command line.

The first supported models will be:

- UNet++ with a configurable SMP encoder
- SegFormer, initially B0
- MambaVision-T with UPerNet

The current `test_oa` metric remains the overall pixel accuracy. No duplicate `test_acc` alias will be added.

## 2. Non-goals

This change will not:

- replace the current training engine with MMEngine or MMSegmentation
- rewrite established implementations that are already available in maintained libraries
- change the OpenEarthMap class mapping, train/test split convention, delayed evaluation schedule, metric prefixes, TTA behavior, or output directory format without an explicit configuration override
- retrain the already completed UNet++ experiment
- add SAM, MambaUNet, or other future models in this implementation cycle

The architecture must, however, make those additions straightforward later.

## 3. Selected approach

Use a lightweight registry-based framework inside the existing repository.

Reasons:

- keeps the verified PyTorch training loop and OEM data path
- avoids forcing all models through MMSegmentation
- limits new dependencies
- gives one stable model interface to the engine
- supports future CLI-selectable components without a large framework migration

Configuration will remain argparse-based for now. YAML/Hydra is intentionally deferred because the current scale does not require another configuration framework.

## 4. Target repository structure

```text
OEM_Segmentation/
├── train.py
├── requirements.txt
├── README.md
├── oemseg/
│   ├── __init__.py
│   ├── constants.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── loaders.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── base.py
│   │   ├── unetpp.py
│   │   ├── segformer.py
│   │   ├── mambavision.py
│   │   └── upernet.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   └── segmentation.py
│   ├── optimizers/
│   │   ├── __init__.py
│   │   └── factory.py
│   ├── schedulers/
│   │   ├── __init__.py
│   │   └── factory.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── segmentation.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── checkpoint.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── reproducibility.py
│       └── tta.py
└── tests/
    ├── test_config.py
    ├── test_data.py
    ├── test_losses.py
    ├── test_metrics.py
    ├── test_models.py
    └── test_training_smoke.py
```

`train.py` becomes a thin entry point that parses arguments, builds components, and calls the trainer. It must not contain model definitions, dataset implementations, metric formulas, or the training loop.

## 5. Stable interfaces

### 5.1 Model interface

Every model adapter must be a `torch.nn.Module` whose forward method returns logits shaped:

```text
[B, 9, H, W]
```

The adapter is responsible for resizing native model outputs back to the input spatial size when necessary.

Each model adapter also exposes parameter groups through a small helper:

```python
model.parameter_groups(base_lr=..., backbone_lr=...)
```

This lets the optimizer factory preserve the current lower learning rate for pretrained backbones without assuming that every model has an attribute named `encoder`.

### 5.2 Loss interface

All registered losses use:

```python
loss(logits, target) -> scalar tensor
```

Initial names:

- `ce`
- `dice`
- `ce_dice`

`ce_dice` is the default and preserves current behavior: multiclass cross entropy plus multiclass Dice loss.

The registry will accept aliases such as `cedice` and `ce-dice`, normalized internally to `ce_dice`.

### 5.3 Optimizer interface

The optimizer factory accepts a normalized name and model parameter groups.

Initial names:

- `adam`
- `adamw`

`adamw` remains the default with the existing weight decay and separate backbone/main learning rates.

### 5.4 Scheduler interface

The existing warmup plus polynomial decay scheduler remains the default. Scheduler construction moves to its own factory so other schedules can be added later without editing the trainer.

### 5.5 Metrics interface

The existing confusion-matrix implementation moves unchanged in meaning to `oemseg/metrics/segmentation.py`.

Reported values remain:

- `oa`
- `miou`
- macro `f1`
- macro `precision`
- macro `recall`
- per-class IoU for all nine OEM classes

The external prefixes remain `val_*` for optional internal validation and `test_*` for the official OEM validation split used as test.

## 6. Model implementations

### 6.1 UNet++

Use `segmentation_models_pytorch.UnetPlusPlus`.

Default behavior must match the completed run:

- model name: `unetpp`
- encoder: `resnet18`
- encoder weights: `imagenet`
- input channels: 3
- output classes: 9

Backward compatibility requirement:

```bash
python train.py
```

must still select UNet++/ResNet18 and the current default experiment settings.

### 6.2 SegFormer

Use Hugging Face `SegformerForSemanticSegmentation` rather than reimplementing MiT or the decoder.

Initial variant:

- CLI name: `segformer`
- variant: `b0`
- pretrained source: NVIDIA SegFormer B0 ImageNet-pretrained checkpoint suitable for replacing the segmentation classifier with 9 classes
- `num_labels=9`
- background label retained
- label reduction disabled

SegFormer emits lower-resolution logits, so its adapter must bilinearly resize logits to the input image size before the shared loss and metric code sees them.

The adapter must expose the SegFormer encoder as the backbone parameter group and the decode head as the main parameter group.

### 6.3 MambaVision-T + UPerNet

Use the official MambaVision-T backbone implementation and pretrained weights. Do not rewrite the backbone.

The official Tiny segmentation configuration uses four hierarchical feature stages with channel dimensions:

```text
[80, 160, 320, 640]
```

The project adapter will attach a UPerNet-style decoder to these four stages and produce 9-class logits. The decoder should be reused from an installed maintained library when a compatible standalone implementation is available; otherwise, implement only the compact UPerNet head and auxiliary components required by this project, not the full MMSegmentation stack.

The design follows the official MambaVision semantic-segmentation pairing of MambaVision-T and UPerNet, but keeps the existing project trainer instead of importing the MMEngine runtime.

The model adapter must:

- return four feature maps from the backbone
- feed them to the UPerNet decoder
- resize final logits to the input size
- expose backbone and decoder parameter groups separately
- support pretrained and randomly initialized modes
- fail with a clear message when optional MambaVision dependencies are missing

Because MambaVision-T + UPerNet is substantially larger than UNet++, the shared trainer will support gradient accumulation. The default for other models stays `1`; the documented example for MambaVision will use batch size `1` with accumulation `2` as an initial RTX 3060 12 GB-safe setting, subject to an actual GPU smoke test.

## 7. Command-line design

The core interface will support:

```bash
python train.py \
  --model unetpp \
  --model-variant resnet18 \
  --loss ce_dice \
  --optimizer adamw \
  --epochs 30
```

SegFormer example:

```bash
python train.py \
  --model segformer \
  --model-variant b0 \
  --loss ce_dice \
  --optimizer adamw \
  --epochs 30
```

MambaVision example:

```bash
python train.py \
  --model mambavision \
  --model-variant tiny \
  --decoder upernet \
  --batch-size 1 \
  --grad-accumulation 2
```

Required new arguments:

- `--model`
- `--model-variant`
- `--decoder`
- `--pretrained` / `--no-pretrained`
- `--loss`
- `--optimizer`
- `--grad-accumulation`

Existing arguments remain supported where still meaningful. Legacy `--encoder` and `--encoder-weights` remain accepted for UNet++ during the compatibility period and map to the new model settings.

Invalid component names must produce an argparse error listing valid registered names.

## 8. Training and evaluation flow

The engine flow remains:

1. Parse and validate configuration.
2. Seed Python, NumPy, and PyTorch.
3. Read OEM train and official validation split files.
4. Optionally split internal validation images from the training set.
5. Construct loaders.
6. Build selected model, loss, optimizer, scheduler, scaler, and optional W&B run.
7. Train with AMP and gradient accumulation.
8. Save best training-loss checkpoint and last checkpoint.
9. Start evaluation only after the configured fraction of epochs, then repeat at the configured interval.
10. Log `val_*` only for an internal validation split and `test_*` for the official OEM validation split.
11. Finish W&B and write a completion record.

Evaluation TTA and multi-scale behavior remain shared across all models. The evaluator receives only logits, so it does not need model-specific branches.

## 9. Checkpoint compatibility

New checkpoints retain the existing top-level keys:

```text
epoch
model
optimizer
scheduler
args
```

The implementation may add metadata such as `model_name`, `model_variant`, and `format_version`, but must not remove the existing keys.

The completed UNet++ checkpoints must remain loadable by constructing the corresponding UNet++ model and reading the `model` state dictionary. Resume-training support is not added in this cycle unless it is already trivial after refactoring.

Weight-only export is outside the current requested scope.

## 10. Dependencies

Keep dependencies minimal and install them only in conda environment `work-env`.

Expected additions:

- `transformers` for SegFormer
- official `mambavision` package or official repository package for the backbone
- a maintained UPerNet implementation dependency only when it integrates cleanly without pulling in the full MMEngine stack

Avoid installing MMSegmentation, MMEngine, MMCV, MMDetection, and MMPretrain into the main environment unless the standalone MambaVision integration proves impossible. The official MambaVision segmentation repository pins a separate OpenMMLab stack and a much older PyTorch/CUDA combination than the current environment, so mixing that runtime into the verified `work-env` is a last resort, not the default design.

All dependency versions actually installed must be pinned or bounded in `requirements.txt` and documented in `README.md`.

## 11. Error handling

The framework must fail early and clearly for:

- unknown model, loss, optimizer, scheduler, or decoder names
- missing optional model dependencies
- invalid OpenEarthMap labels outside `[0, 8]`
- missing image/label pairs
- incompatible pretrained classifier shapes
- invalid gradient accumulation values
- CUDA unavailable for the paper-style experiment
- output directory collisions

Optional model imports must be lazy so that UNet++ remains usable even when MambaVision or Transformers is not installed.

## 12. Testing strategy

Tests will be split by responsibility.

### Unit tests

- registry lookup and invalid-name messages
- config normalization and legacy argument compatibility
- CE, Dice, and CE+Dice finite backward pass
- confusion-matrix metrics, including perfect prediction
- optimizer selection and separate learning-rate groups
- scheduler warmup and final decay
- evaluation schedule
- checkpoint key compatibility

### Model tests

For each model:

- instantiate without downloading weights when possible
- forward a small tensor
- verify output shape `[B, 9, H, W]`
- run one loss backward pass
- verify backbone/main parameter groups are nonempty and nonoverlapping

### Integration smoke tests

- one train batch and one evaluation batch on real OEM data for UNet++
- the same for SegFormer
- the same for MambaVision-T + UPerNet on RTX 3060 12 GB
- verify logs and `metrics.jsonl` retain `train_loss`, `test_oa`, `test_miou`, and per-class IoU keys
- verify old UNet++ default CLI still runs
- verify W&B offline mode still initializes and finishes

No full 45-epoch training is required for this refactor task.

## 13. Documentation

Update `README.md` with:

- new project tree
- installation commands for `work-env`
- component registry names
- examples for all three models
- memory-oriented MambaVision example
- explanation that `test_oa` is overall pixel accuracy
- explanation that official OEM validation is used as test because test labels are unavailable
- license notice for MambaVision source and pretrained weights

## 14. Acceptance criteria

The task is complete when:

- `train.py` is a thin entry point
- the framework is organized under `oemseg/`
- UNet++, SegFormer-B0, and MambaVision-T + UPerNet all expose the same logits interface
- model, loss, optimizer, epoch count, and key training settings are selectable from CLI
- current UNet++ defaults and metric names remain compatible
- `test_oa` remains present and no duplicate `test_acc` is added
- unit tests pass
- all three models pass forward/backward smoke tests
- all three models pass a real-data smoke run on the available GPU, using gradient accumulation where needed
- README and requirements are updated
- the repository is clean after commit
- project permissions are reset recursively to `777` after implementation and verification

## 15. External implementation references

- NVIDIA MambaVision official repository and semantic segmentation configuration:
  - https://github.com/NVlabs/MambaVision
  - https://github.com/NVlabs/MambaVision/blob/main/semantic_segmentation/configs/mamba_vision/mamba_vision_160k_ade20k-512x512_tiny.py
- Hugging Face SegFormer documentation:
  - https://huggingface.co/docs/transformers/en/model_doc/segformer

These references define the intended upstream implementations. Project-specific training behavior remains governed by this specification and the existing verified OEM pipeline.
