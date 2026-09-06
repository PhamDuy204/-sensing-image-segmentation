# Kaggle Resumable Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make long OpenEarthMap Kaggle jobs finish the 45-epoch paper protocol without losing work at Kaggle's 12-hour cutoff, then relaunch UNetFormer, MambaVision, and PyramidMamba.

**Architecture:** Keep the existing trainer and Kaggle CLI workflow. Add exact checkpoint restore to the shared trainer, then let the existing pipeline run 15-epoch Kaggle chunks as separate committed kernels; each next chunk attaches the previous kernel output via `kernel_sources` and resumes `last.pt`. Intermediate chunks save checkpoints/history only, while the final chunk performs the official 500-image evaluation/TTA. W&B offline chunks are synced by the PC watcher into one stable online run ID using append semantics.

**Tech Stack:** Python, PyTorch, Accelerate, W&B offline sync, Kaggle CLI/kernel metadata, Bash, pytest.

**Spec:** Current conversation requirements and the verified PyramidMamba/OpenEarthMap paper-reproduction protocol.

## Global Constraints

- Keep 45 total epochs, FP32, AdamW, LR 6e-4, encoder LR 6e-5, weight decay 0.01, Poly power 0.9, warmup 5, batch 1/GPU on T4x2 for these three models.
- Preserve model-native/published losses already implemented.
- Do not rewrite model architectures or add dependencies.
- Never use the public 500-image reported split for checkpoint selection.
- Keep W&B credentials off Kaggle; sync offline runs from the PC.
- Prefer accuracy/reproducibility over speed.

---

### Task 1: Exact trainer resume

**Files:**
- Modify: `oemseg/config.py`
- Modify: `oemseg/engine/checkpoint.py`
- Modify: `oemseg/engine/trainer.py`
- Test: `tests/test_config.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Produces CLI `--resume-from PATH` and `--stop-after-epoch N`.
- Produces `restore_checkpoint(path, model, optimizer, scheduler, args, world_size, map_location)`.
- Checkpoints persist `training_state` containing best/stale state needed across chunks.

- [x] **Step 1: Write failing tests** for CLI options, checkpoint training-state round trip, and incompatible-recipe rejection.
- [x] **Step 2: Run tests and verify RED** due to missing options/helper.
- [ ] **Step 3: Implement minimal checkpoint save/restore and CLI validation.** Resume validation compares model/variant/world-size and training-affecting recipe fields (`epochs`, size, batch, accumulation, optimizer, LR values, WD, warmup, Poly power, loss, seed).
- [ ] **Step 4: Integrate resume in `run_training`.** Restore model/optimizer/scheduler after `accelerator.prepare`, copy prior `metrics.jsonl` and best checkpoint into the new run directory, resume from absolute `checkpoint['epoch'] + 1`, and preserve best/stale state.
- [ ] **Step 5: Stop cleanly at a chunk boundary.** When `stop_after_epoch < epochs`, write `chunk_state.json`, finish W&B, and return without final official evaluation/TTA. Early stopping still proceeds to final evaluation.
- [ ] **Step 6: Run focused and full trainer tests.**

### Task 2: Chunk-aware Kaggle launcher

**Files:**
- Modify: `scripts/kaggle_pipeline.py`
- Modify: `scripts/kaggle_paper_repro.sh`
- Test: `tests/test_kaggle_pipeline.py`

**Interfaces:**
- Produces `chunk_end_epochs(total_epochs, chunk_epochs)`.
- Extends `build_kernel_files(..., chunk_end_epoch=None, previous_kernel=None)`.
- Adds CLI `--chunk-epochs`; `0` keeps current one-version behavior.

- [x] **Step 1: Write failing tests** for `CANCEL_ACKNOWLEDGED`, 15-epoch chunk plan, `kernel_sources`, resume env, and final-only evaluation.
- [x] **Step 2: Run tests and verify RED** due to missing chunk support.
- [ ] **Step 3: Implement status parsing and chunk metadata.** Chunk 45 epochs into `[15, 30, 45]`; part 2+ attaches the immediately previous kernel output.
- [ ] **Step 4: Make Bash resume from Kaggle input.** Find exactly one `*/oem_outputs/${RUN_NAME}/last.pt`, pass it as `--resume-from`, pass absolute `--stop-after-epoch`, and set `--eval-start-epoch 44` for chunked jobs so official TTA happens only at completion.
- [ ] **Step 5: Orchestrate chunks sequentially.** Each part must reach COMPLETE, download `oem_outputs`, sync W&B, inspect `chunk_state.json`, and launch the next part only when training is incomplete.
- [ ] **Step 6: Use one W&B online run per model.** Generate one stable run ID per chunked pipeline execution; sync part 1 to that ID and use `wandb sync --append --id` for later parts.
- [ ] **Step 7: Run focused and full pipeline tests plus shell/compile checks.**

### Task 3: Kaggle integration smoke and deployment

**Files:**
- No new production files.
- Git branch: `feature/kaggle-resumable-training` -> `main`.

- [ ] **Step 1: Push feature branch and run a two-part Kaggle smoke/probe** that verifies a completed kernel output is mounted through `kernel_sources` and `last.pt` is discovered by the next version.
- [ ] **Step 2: Verify the resumed checkpoint epoch advances rather than restarting at epoch 1.**
- [ ] **Step 3: Run fresh full verification:** `pytest -q tests`, `bash -n`, `compileall`, `git diff --check`.
- [ ] **Step 4: Fast-forward merge into `main`, push GitHub, and verify local/origin/remote SHA equality.**
- [ ] **Step 5: Relaunch UNetFormer, MambaVision, PyramidMamba with `--chunk-epochs 15` on accounts 2, 6, 7 respectively.**
- [ ] **Step 6: Confirm each part-1 kernel is a saved version and reaches QUEUED/RUNNING, with detached watcher PPID=1.**
