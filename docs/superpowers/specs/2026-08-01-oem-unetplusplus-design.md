# OEM UNet++ Reproduction Design

## Goal
Reproduce the OpenEarthMap experiment settings described in PyramidMamba with a UNet++ model, using the public OEM validation split as the reported test set because the official test labels are unavailable.

## Dataset layout
- Keep the current incomplete source at `datasets/OpenEarthMap/OpenEarthMap_wo_xBD` unchanged.
- Selectively extract the 1,162 required xBD PNGs from the public split Hugging Face ZIP into `datasets/OpenEarthMap/xBD_huggingface`; do not download the complete 32.6 GB archive.
- Build the completed dataset at `datasets/OpenEarthMap/OpenEarthMap` by copying the existing OEM tree and inserting only the missing xBD RGB images according to `xbd_files.csv`.
- Verify exact paired counts: 3000 train images/labels and 500 validation images/labels. The official 1500-image test split is retained but not evaluated because labels are absent.

## Training protocol
- Model: `segmentation_models_pytorch.UnetPlusPlus` with a ResNet18 encoder and nine output classes.
- Input: RGB resized to 1024 x 1024.
- Loss: cross entropy plus multiclass Dice loss.
- Optimizer: AdamW, decoder/head LR `6e-4`, encoder LR `6e-5`, weight decay `0.01`.
- Schedule: 45 epochs, five-epoch linear warmup, polynomial decay with power `0.9`.
- Batch size: 2; AMP enabled by default.
- Augmentation: random horizontal and vertical flips while training.
- Reporting: evaluate official OEM `val.txt` as `test` every three epochs beginning after two-thirds of training. Do not use test metrics for early stopping or checkpoint selection. Save `last.pt` and the lowest-training-loss checkpoint.
- Optional internal validation mode may split the official training list deterministically; it is disabled by default.
- Optional W&B logging is disabled by default and enabled with a CLI flag.

## Logging and outputs
Each run creates a timestamped directory in `outputs/` containing:
- `train.log` with `train_loss`, and on evaluation epochs `test_oa`, `test_miou`, `test_f1`, `test_precision`, `test_recall`, plus per-class IoU.
- `metrics.jsonl` with the same structured metrics.
- `config.json`, `last.pt`, and `best_train_loss.pt`.

## Verification
- Dataset preparation script has a dry-run/verification mode and fails on missing or duplicate mappings.
- A small runnable test validates label handling, loss, metric accumulation, and one forward/backward step on synthetic data.
- A one-batch smoke run on the real dataset confirms data paths, GPU execution, logging, and checkpoint writing.
