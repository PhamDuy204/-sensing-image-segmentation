# OEM Training Research Workflow Design

**Date:** 2026-08-14

## Goal

Upgrade the existing OpenEarthMap segmentation workspace without replacing its modular trainer. The new workflow must use a paper-safe split, log validation and reported-test metrics every epoch, select checkpoints from validation only, record compact per-image failure information, support one-command single-GPU and Accelerate-based multi-GPU/Kaggle training, and remove avoidable evaluation/throughput bottlenecks.

## Dataset protocol

- Official `train.txt` (3000 labeled images) is the only source for optimization train and internal validation.
- Default deterministic split: 2700 train / 300 internal val (`val_fraction=0.1`, seed 42 unless overridden).
- Preserve geographic coverage by stratifying the 3000-image split by `region_for(filename)` using deterministic stdlib logic; do not add a dependency just for splitting.
- Official `val.txt` (500 labeled images) is the experiment/reporting `test` set.
- Official `test.txt` (1500 benchmark images) is outside paper metrics and is only for optional future submission export.
- Persist the exact resolved names for every run under `splits/train.txt`, `splits/val.txt`, and `splits/test.txt`.
- Train/val/test labeled sets must be disjoint.

## Evaluation and checkpointing

Each normal epoch runs train -> internal val -> reported test -> logging. Validation and test metrics are both visible each epoch, but only internal validation may drive `best_val_miou.pt` and early stopping. Test values are observational and must never drive checkpoint selection, scheduler state, or stopping.

Keep `last.pt` and the existing `best_train_loss.pt` for compatibility; document `best_val_miou.pt` as the primary paper checkpoint.

## Lightweight error analysis

Do not dump masks or images for every failure. During val/test evaluation, record compact per-sample information: epoch, split, filename, region, loss, OA, mIoU, lowest-IoU class and its IoU. Append these rows to `sample_scores.jsonl`.

For each evaluated epoch write top-N worst samples (default 30) to `bad_predictions_val.tsv` and `bad_predictions_test.tsv`. Whenever validation mIoU reaches a new best, also snapshot `bad_predictions_val_best.tsv` and the same epoch's reported-test list as `bad_predictions_test_at_best_val.tsv`.

## Metrics performance

The current confusion matrix sends full 1024x1024 predictions/targets to CPU every batch. Accumulate bincount/confusion matrices on the active device and move only compact matrices/scalars to CPU for logging. In distributed evaluation reduce only compact tensors. Per-image statistics are also computed from compact confusion matrices.

## Accelerate / multi-GPU / Kaggle

Use Hugging Face Accelerate instead of hand-written DDP or `nn.DataParallel`.

- Plain `python3 train.py ...` remains valid for one GPU.
- Multi-GPU CLI uses `accelerate launch --num_processes=2 train.py ...`.
- Kaggle uses `accelerate.notebook_launcher(training_function, args=(...), num_processes=2)` and reuses the real trainer rather than copying the training loop into the notebook.
- Construct `Accelerator` inside `run_training`/the launched function.
- Prepare model/optimizer/loaders/scheduler via Accelerator.
- Use Accelerator gradient accumulation and backward handling.
- Only the main process may create/write run files, W&B state, rankings, and checkpoints.
- Unwrap the model before checkpoint serialization.
- Gather/reduce compact metrics and per-sample records across ranks.

## Throughput improvements

Enable safe shared optimizations: pinned memory, persistent workers, non-blocking transfers, AMP through Accelerate, cuDNN benchmark for fixed-size image training, and high float32 matmul precision/TF32 on supported NVIDIA GPUs. Keep `optimizer.zero_grad(set_to_none=True)`.

Do not enable `torch.compile` by default. `channels_last` may be exposed as an optional flag only if a smoke benchmark confirms compatibility; it must not become a model-specific fork in the trainer.

## Logging / compatibility

Keep the current thin `train.py`, `metrics.jsonl`, `train.log`, and optional W&B flow. Each normal epoch should contain both `val_*` and `test_*`. Record split sizes/seed/fraction, world size, mixed precision, and performance flags in configuration/logging. Existing model adapters and old single-GPU CLI usage should keep working.

## Tests and verification

Add the smallest runnable tests covering deterministic 2700/300/500 resolution, region coverage/disjointness/manifests, device-local confusion matrix behavior, per-image ranking, val-only checkpoint/stopping decision helpers, main-process write behavior, and existing CLI compatibility. Run the full suite plus compile checks and a real single-GPU smoke run. If two GPUs are unavailable locally, verify notebook/Accelerate launch structure statically and leave the actual 2-GPU smoke command documented for Kaggle.

## Integration

Implement in an isolated feature worktree so the existing uncommitted U-Net changes in the original checkout remain untouched. After verification, integrate the completed feature back into the repository's primary branch (`main`/`master`, whichever exists or is established as the primary line) without discarding unrelated local work.
