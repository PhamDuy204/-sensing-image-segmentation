# Kaggle Paper-Repro Auto-Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a one-command headless Kaggle lifecycle that automatically syncs offline W&B data when training finishes.

**Architecture:** Keep Kaggle training logic in one shell entrypoint already living in the repository, and keep remote orchestration in one stdlib Python script. The Python script generates the minimal notebook/metadata dynamically, runs Kaggle CLI as a subprocess with `KAGGLE_API_TOKEN`, polls until terminal state, downloads outputs, and runs local `wandb sync`.

**Tech Stack:** Python 3.11 stdlib, Bash, Kaggle CLI 2.2.4, W&B CLI, existing OEM training stack.

**Spec:** `docs/superpowers/specs/2026-09-06-kaggle-auto-sync-design.md`

## Global Constraints
- One authorized Kaggle account per run.
- Never persist Kaggle/W&B credentials in the repository or generated notebook.
- Use T4 x2, FP32, global batch 2, no internal validation, `--loss auto`.
- Default full run is 45 epochs; smoke mode is explicitly separate.

---

### Task 1: Kaggle-side paper recipe

**Files:**
- Create: `scripts/kaggle_paper_repro.sh`
- Test: `tests/test_kaggle_pipeline.py`

**Interfaces:**
- Consumes: environment `MODEL_NAME`, `DATA_ROOT`, `OUTPUT_ROOT`, optional `SMOKE=1`.
- Produces: `/kaggle/working/oem_outputs`, offline W&B run directories, and a nonzero exit code on validation/training failure.

- [ ] Write a failing test asserting the shell entrypoint contains global-batch-2/FP32/no-val/auto-loss/offline-W&B arguments.
- [ ] Run the targeted test and confirm RED because the script is absent.
- [ ] Implement the minimal shell entrypoint using existing setup scripts and `scripts/launch.py`.
- [ ] Run the targeted test and confirm GREEN.

### Task 2: PC lifecycle orchestrator

**Files:**
- Create: `scripts/kaggle_pipeline.py`
- Modify: `tests/test_kaggle_pipeline.py`

**Interfaces:**
- `build_kernel_files(...) -> tuple[dict, dict]` returns notebook JSON and kernel metadata with private T4/internet/dataset settings.
- `normalize_status(text: str) -> str` maps Kaggle CLI status output to a stable uppercase status.
- CLI `python scripts/kaggle_pipeline.py --model <model> [--smoke] [--detach]` owns push -> poll -> output download -> W&B sync.

- [ ] Add failing tests for metadata/notebook security, status parsing, and command-line recipe generation.
- [ ] Run targeted tests and confirm RED for missing functions/module.
- [ ] Implement only the functions and lifecycle needed by the tests, using `subprocess` and `pathlib`.
- [ ] Run targeted tests and confirm GREEN.

### Task 3: Local client and live smoke integration

**Files:**
- Modify: `README.md`

**Interfaces:**
- Local tool venv: `~/.local/share/oem-kaggle-client` containing Kaggle CLI 2.2.4 and W&B CLI.
- Account token default: `~/.config/kaggle/accounts/account_1/access_token`.

- [ ] Create/reuse the local client venv without changing project requirements.
- [ ] Verify Kaggle auth with account_1 without printing the token.
- [ ] Run a generated smoke kernel on T4x2, wait automatically, download outputs, and sync W&B offline data.
- [ ] Document the one-command full 45-epoch invocation and output/state locations.

### Task 4: Verification and integration

**Files:**
- All files above.

- [ ] Run `pytest -q` in the established project env and require all tests green.
- [ ] Run `bash -n scripts/kaggle_paper_repro.sh`, `python -m compileall scripts/kaggle_pipeline.py`, and `git diff --check`.
- [ ] Review staged diff for secrets/generated run data.
- [ ] Commit the focused change, fast-forward main, push, and verify local `main == origin/main`.
