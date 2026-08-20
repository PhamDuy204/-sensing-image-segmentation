# OEM Training Research Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible 2700/300/500 research protocol, compact failure analysis, Accelerate multi-GPU/Kaggle execution, and shared throughput fixes without rewriting the existing model adapters/trainer architecture.

**Architecture:** Keep `train.py` thin and extend the shared data/evaluator/trainer path. Resolve splits before DataLoader construction, evaluate with device-local compact confusion matrices and per-sample records, and let `Accelerator` own device/distributed/AMP/gradient accumulation while rank 0 owns files/checkpoints/W&B. Add only one thin Kaggle launcher notebook and no duplicated training loop.

**Tech Stack:** Python 3.11, PyTorch 2.13, Hugging Face Accelerate, torchvision, existing model/loss/optimizer/scheduler factories, pytest.

## Global Constraints

- Official `train.txt` 3000 -> default 2700 train / 300 internal val, deterministic and region-stratified.
- Official `val.txt` 500 -> reported `test`; official 1500 benchmark test is not a paper metric.
- `best_val_miou.pt` and early stopping use validation only; reported-test metrics never influence model selection.
- No bulk prediction-image storage; store compact per-sample scores and top-N TSV rankings.
- Plain `python3 train.py` must still work; multi-GPU goes through Accelerate; Kaggle reuses the same training function.
- Do not overwrite or include unrelated uncommitted U-Net changes from the original checkout.
- Merge the completed, verified feature back to the repository's primary branch at finish.

---

### Task 1: Deterministic research split and manifests

**Files:**
- Modify: `oemseg/config.py`
- Modify: `oemseg/data/loaders.py`
- Test: `tests/test_config.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produce: `split_train_val(names: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]`
- Produce: `write_split_manifests(run_dir: Path, loaders: LoaderBundle) -> None`
- Preserve `LoaderBundle.train`, `.internal_val`, `.test` and count fields.

- [ ] Write failing tests for default `internal_val_fraction == 0.1`, `--val-fraction`/legacy alias parsing, deterministic exact 2700/300 split, region representation, disjoint reported test, and manifest contents.
- [ ] Run targeted config/data tests and confirm they fail for the missing behavior.
- [ ] Implement stdlib deterministic per-region allocation with exact requested validation count; change the default to 0.1 while accepting both CLI flag names; persist manifests from the pre-Accelerate loaders.
- [ ] Run targeted tests and the full suite.
- [ ] Commit only Task 1 files.

### Task 2: Device-local metrics and compact per-sample failure records

**Files:**
- Modify: `oemseg/data/dataset.py`
- Modify: `oemseg/data/loaders.py`
- Modify: `oemseg/metrics/segmentation.py`
- Modify: `oemseg/engine/evaluator.py`
- Create: `oemseg/engine/error_analysis.py`
- Test: `tests/test_data.py`
- Test: `tests/test_metrics.py`
- Create/Test: `tests/test_error_analysis.py`

**Interfaces:**
- `OEMDataset(..., return_name: bool = False)` returns `(image, mask, name)` only for eval loaders when requested.
- `ConfusionMatrix(device: torch.device | None = None)` keeps `.matrix` on the update device and provides compact metrics.
- `EvaluationResult(loss: float, metrics: SegmentationMetrics, samples: list[dict[str, object]])` returned by `evaluate`.
- `write_error_analysis(run_dir, epoch, split, samples, top_n, best_snapshot=False)` writes JSONL/TSV artifacts.

- [ ] Write failing tests proving eval samples expose filenames, confusion matrices remain on the tensor device, per-image scores map to the right name, and ranking/top-N output is sorted without image payloads.
- [ ] Run targeted tests and confirm expected failures.
- [ ] Implement optional dataset names, device-local confusion accumulation, compact per-image confusion/metrics, and TSV/JSONL writers.
- [ ] Run targeted tests and the full suite.
- [ ] Commit Task 2 files.

### Task 3: Validation-only selection helpers and every-epoch val/test protocol

**Files:**
- Modify: `oemseg/config.py`
- Modify: `oemseg/engine/trainer.py`
- Test: `tests/test_config.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Produce small pure helper `update_validation_state(val_miou: float, best_val_miou: float, stale: int) -> tuple[float, int, bool]` where the boolean indicates a new best.
- Normal training evaluates both val and reported test every epoch; smoke still evaluates one batch each.

- [ ] Write failing tests proving test mIoU cannot affect validation state, validation improvement resets stale, and paper-oriented defaults evaluate every epoch.
- [ ] Run targeted tests and confirm failure.
- [ ] Implement the pure validation-state helper; make normal defaults evaluate every epoch while keeping legacy evaluation options accepted; connect error-analysis snapshots to the same new-best validation event.
- [ ] Run targeted tests and full suite.
- [ ] Commit Task 3 files.

### Task 4: Accelerate single/multi-GPU integration

**Files:**
- Modify: `requirements.txt`
- Modify: `oemseg/engine/checkpoint.py`
- Modify: `oemseg/engine/evaluator.py`
- Modify: `oemseg/engine/trainer.py`
- Test: `tests/test_engine.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- `run_training(args) -> Path` remains the public entry point.
- `train_one_epoch(..., accelerator, ...)` uses `accelerator.accumulate(model)` and `accelerator.backward(loss)`; no manual GradScaler.
- `evaluate(..., accelerator, ...)` reduces compact matrices/loss sums and gathers compact sample records.
- `save_checkpoint(..., model_state_dict: dict | None = None)` preserves legacy keys while allowing the caller to pass the unwrapped state dict.

- [ ] Add `accelerate>=1.10,<2` and install it into `work-env`.
- [ ] Write failing CPU-safe unit tests for Accelerator-backed gradient accumulation, main-process-only artifact helper behavior, and unwrapped checkpoint state serialization.
- [ ] Run targeted tests and confirm failure.
- [ ] Replace manual device/GradScaler/distribution handling with `Accelerator`; prepare model/optimizer/loaders/scheduler; reduce evaluation tensors and gather sample objects; guard run files/W&B/checkpoints on `accelerator.is_main_process`.
- [ ] Verify plain single-process CPU-safe unit tests and full suite.
- [ ] Commit Task 4 files.

### Task 5: Safe throughput settings

**Files:**
- Modify: `oemseg/config.py`
- Modify: `oemseg/engine/trainer.py`
- Test: `tests/test_config.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Produce `configure_torch_performance(device: torch.device) -> None` enabling high matmul precision and cuDNN benchmark only where relevant.
- Optional `--channels-last` flag defaults false; no default `torch.compile` behavior.

- [ ] Write failing tests for performance defaults/flags and a CPU-safe performance helper.
- [ ] Run targeted tests and confirm failure.
- [ ] Implement only high matmul precision, CUDA cuDNN benchmark/TF32-compatible settings, and optional channels-last conversion; do not add compile/model-specific branches.
- [ ] Run tests and compile checks.
- [ ] Commit Task 5 files.

### Task 6: Kaggle launcher and documentation

**Files:**
- Create: `notebooks/kaggle_multi_gpu.ipynb`
- Modify: `README.md`
- Modify: `tests/test_training_smoke.py`

**Interfaces:**
- Notebook imports `parse_args` and `run_training`, defines a zero-argument launched wrapper around real project code, and invokes `notebook_launcher(..., num_processes=2)` without CUDA initialization beforehand.

- [ ] Write failing static tests asserting the notebook uses `notebook_launcher`, references `run_training`, and does not contain duplicated `train_one_epoch`/`evaluate` implementations.
- [ ] Run the static test and confirm failure because the notebook does not exist.
- [ ] Add the minimal notebook and README sections documenting 2700/300/500 semantics, val-only selection, error artifacts, plain one-GPU command, `accelerate launch`, Kaggle launcher, and official 1500-image benchmark limitation.
- [ ] Run targeted and full tests.
- [ ] Commit Task 6 files.

### Task 7: Real verification and integration

**Files:**
- No new production files unless verification exposes a bug.

- [ ] Run `python -m pytest -q` in `work-env` and require zero failures.
- [ ] Run `python -m compileall -q oemseg train.py`.
- [ ] Run one real GPU smoke command with TTA reduced for speed and confirm config, split manifests, metrics, checkpoints, JSONL, and TSV artifacts are created.
- [ ] Inspect `git diff`, `git status`, and original checkout status to prove the U-Net edits remain untouched.
- [ ] Read and follow the Superpowers verification-before-completion and finishing-a-development-branch skills.
- [ ] Identify/establish the primary branch from repository history, integrate `research-workflow` into it without losing unrelated work, and rerun the fast test suite on the integrated branch.
