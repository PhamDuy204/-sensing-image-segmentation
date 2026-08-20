# OEM UNet++ Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete OpenEarthMap dataset and a reproducible UNet++ training pipeline matching the paper's published OEM settings where possible.

**Architecture:** Download public xBD archives beside the existing OEM data, map only required pre-disaster images into a completed OEM copy, then train UNet++ from a compact PyTorch CLI. The official OEM validation split is reported as `test` and is never used for checkpoint selection.

**Tech Stack:** Python 3.11, PyTorch, torchvision, segmentation-models-pytorch, Pillow, NumPy, W&B optional.

## Global Constraints
- Run only in conda environment `work-env`.
- Dataset root remains under `/home/duypham/workspace/OEM_Segmentation/datasets/OpenEarthMap`.
- Nine semantic classes including background.
- Paper settings: 1024 input, batch 2, AdamW, LR `6e-4`, encoder LR `6e-5`, weight decay `0.01`, 45 epochs, warmup 5, Poly power `0.9`, CE + Dice.
- Begin evaluation after two-thirds of training, then evaluate every three epochs and prefix official validation metrics with `test_`.
- W&B is optional and off by default.
- Apply `chmod -R 777` to the project after implementation and verification.

---

### Task 1: Prepare and verify full OEM dataset

**Files:**
- Create: `scripts/prepare_oem_xbd.py`
- Create: `datasets/OpenEarthMap/xBD_huggingface/`
- Create: `datasets/OpenEarthMap/OpenEarthMap/`

**Interfaces:**
- Produces a complete OEM root accepted by the training CLI.

- [x] Parse the split Hugging Face ZIP64 directory and range-extract only the 1,162 images referenced by `xbd_files.csv`.
- [x] Store verified PNGs and a manifest under `xBD_huggingface` without downloading the complete archive.
- [x] Implement mapping from `xbd_files.csv`, preserving GeoTIFF metadata from OEM labels.
- [x] Verify 3000 train and 500 validation image-label pairs, 1500 public test images, and label IDs 0 through 8.

### Task 2: Implement minimal UNet++ training CLI

**Files:**
- Create: `train.py`
- Create: `requirements.txt`

**Interfaces:**
- `python train.py --data-root ... [--wandb] [--smoke]`
- Produces run logs and checkpoints in `outputs/`.

- [x] Install only missing Python dependencies into `work-env`.
- [x] Implement dataset loading, augmentations, model construction, CE + Dice, optimizer parameter groups, warmup + polynomial LR, AMP, metrics, checkpoints, and optional W&B.
- [x] Log training loss each epoch and `test_*` metrics every third epoch after two-thirds of training.
- [x] Keep fixed-epoch training as default and never select by test metrics.

### Task 3: Add runnable checks and smoke test

**Files:**
- Create: `tests/test_pipeline.py`
- Create: `README.md`

**Interfaces:**
- `python tests/test_pipeline.py`
- `python train.py --smoke`

- [x] Add assert-based checks for labels, loss, metrics, ZIP extraction, dataset preparation, and one synthetic forward/backward step.
- [x] Run checks in `work-env`.
- [x] Run one-batch real-data GPU smoke tests with local and W&B-offline logging and inspect generated files.
- [x] Document exact commands for normal and W&B runs.

### Task 4: Final verification and permissions

- [x] Verify project tree, Git diff, dataset counts, GPU smoke result, and log field names.
- [x] Apply `chmod -R 777 /home/duypham/workspace/OEM_Segmentation`.
- [x] Commit code and documentation without committing datasets or outputs.
