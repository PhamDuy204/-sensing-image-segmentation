# W&B Selection and UNetFormer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task. Each behavior starts with a failing test.

**Goal:** Publish the actually selected segmentation checkpoint, complete selected-checkpoint metrics/error analysis/visuals to W&B in val and no-val modes, and add a reusable upstream UNetFormer adapter.

**Architecture:** Reuse the existing loader/evaluator/trainer and make checkpoint selection policy depend only on whether an internal validation loader exists. Extend the existing metric representation and visualization helpers rather than creating model-specific paths. Keep UNetFormer upstream in a pinned gitignored checkout and adapt it to the existing model interface.

**Tech Stack:** PyTorch, Accelerate, W&B, PIL, timm, segmentation-models-pytorch, pytest, Git.

**Spec:** `docs/superpowers/specs/2026-09-04-wandb-selection-unetformer-design.md`

## Global Constraints
- `--val-fraction 0` means no-validation selection by minimum training loss.
- `--val-fraction > 0` means existing region-aware validation selection by maximum val mIoU.
- Test metrics never select a checkpoint.
- OpenEarthMap official palette is used for visualization.
- UNetFormer source is reused from pinned GeoSeg rather than copied/reimplemented.

### Task 1: Complete metrics and OpenEarthMap visualization
- [ ] Add failing tests for per-class F1/precision/recall flattening and official RGB palette/legend.
- [ ] Run focused tests and confirm failures.
- [ ] Extend shared metrics and visualization minimally.
- [ ] Run focused tests green.

### Task 2: Unified selected-checkpoint policy and W&B artifact/summary
- [ ] Add failing tests for loss-selection state, mode-dependent selected checkpoint, summary flattening, and model artifact file list.
- [ ] Run focused tests and confirm failures.
- [ ] Refactor the shared trainer to track best train epoch and no-val patience, re-evaluate the selected checkpoint, write mode-appropriate below-mean analysis, log complete summary metrics, and upload selected checkpoint + analysis as a model artifact.
- [ ] Run focused tests green.

### Task 3: Reuse upstream UNetFormer
- [ ] Add failing registry/adapter tests.
- [ ] Confirm failures.
- [ ] Add pinned upstream setup script, gitignore entry, adapter, registry/default variant wiring, and PyTorch compatibility shim.
- [ ] Run setup and forward tests green.

### Task 4: Documentation and stale baseline test
- [ ] Change the stale Kaggle test to assert the actual shared `scripts/launch.py` path and confirm the old assertion is the only baseline mismatch.
- [ ] Update README with both selection modes, W&B outputs, palette/legend, UNetFormer setup/use, and one-epoch verification command.
- [ ] Run the complete test suite.

### Task 5: Real PC verification and integration
- [ ] Run one online-W&B UNet epoch on the RTX 3060 with fixed test and no-validation mode so epoch 1 must be selected.
- [ ] Inspect local run outputs and W&B sync result without printing credentials.
- [ ] Verify all tests and git diff.
- [ ] Merge feature branch into `main`, push `origin/main`, fetch, and prove `HEAD == origin/main` and clean status.
