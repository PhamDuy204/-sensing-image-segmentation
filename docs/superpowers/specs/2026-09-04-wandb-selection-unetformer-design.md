# W&B Best-Checkpoint, Error Analysis, Visualization, and UNetFormer Design

## Goal
Keep one shared training/evaluation pipeline while making the selected checkpoint explicit in both validation and no-validation modes, publishing complete selected-checkpoint metrics and artifacts to W&B, using the OpenEarthMap label palette for qualitative outputs, and adding UNetFormer by reusing the pinned upstream GeoSeg implementation.

## Selection modes
`--val-fraction` is the mode switch already present in the CLI; no second flag is needed.

- `--val-fraction 0`: all official training samples remain in train, the official held-out split remains test, the best checkpoint is the lowest training-loss checkpoint, patience counts consecutive epochs without a lower training loss, and selected-checkpoint error analysis is test-only.
- `--val-fraction > 0`: the existing deterministic region-aware train/validation split is used (for example `0.2` means 80/20), the official held-out split remains test, the best checkpoint is the highest validation mIoU checkpoint, and patience keeps its validation-based behavior.

The test split never selects a checkpoint in either mode.

## Metrics and W&B summary
The shared confusion-matrix metric object will expose OA, mIoU, macro F1, macro precision, macro recall, plus per-class IoU/F1/precision/recall. The selected checkpoint is re-evaluated once after training. W&B Summary receives `best/epoch`, `best/selection_mode`, `best/selection_metric`, and every flattened metric from the selected checkpoint's test evaluation. Validation metrics are also written when validation exists.

## Artifacts and error analysis
The selected checkpoint is uploaded as a W&B `model` artifact together with `best_checkpoint_summary.json`, complete per-sample selected-checkpoint score tables, and the below-mean error-analysis TSVs. In validation mode this includes both `below_mean_val.tsv` and `below_mean_test.tsv`; in no-validation mode only test analysis is produced. Existing reproducibility artifacts remain unchanged.

## Visualization
Class IDs remain uint8/long single-channel masks internally. Only presentation converts masks to RGB. The RGB palette is the OpenEarthMap class palette with class 0 background black. A generated legend image lists every class name beside its color. Selected-checkpoint W&B media includes the legend and train/val/test grids that are applicable to the selected mode.

## UNetFormer
Reuse `WangLibo1995/GeoSeg` at pinned commit `9453fe48209c4626b29e35e61bab93b61212c4b1` under gitignored `.vendor/GeoSeg`. A small adapter supplies the existing model contract, applies a runtime PyTorch-2.13 reflect-padding compatibility shim, returns only primary logits, and freezes the unused auxiliary head. No UNetFormer architecture code is copied into this repository.

## Verification
Add focused tests for full per-class metrics, both checkpoint-selection modes, below-mean artifact contents, RGB palette/legend, and UNetFormer registry/forward contract. Repair the stale Kaggle orchestration test so the baseline suite matches the current `scripts/launch.py` flow. Then run the full suite, a UNetFormer forward smoke test, and one real one-epoch online W&B UNet training on the RTX 3060. Finally merge to `main`, push, fetch, and verify local `main` equals `origin/main` with a clean worktree.
