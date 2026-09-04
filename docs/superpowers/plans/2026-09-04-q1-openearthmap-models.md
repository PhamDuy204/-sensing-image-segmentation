# Q1 OpenEarthMap Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate PyramidMamba natively and add reproducible official-upstream runners for GeoSA-BaSA, HG-RSOVSSeg and RepSTDC.

**Architecture:** Reuse the existing pinned GeoSeg checkout for PyramidMamba and the local trainer contract. Keep the three OpenMMLab-style projects in gitignored, pinned `.vendor/` checkouts controlled by a standard-library sidecar CLI.

**Tech Stack:** Python 3.11, PyTorch, existing OEM trainer, Git, official GitHub repositories, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-q1-openearthmap-models-design.md`

## Global Constraints
- Do not copy official third-party model source into this repository.
- Do not add new runtime dependencies for the sidecar wrapper.
- Preserve the existing OEM dataset/training/evaluation behavior for native models.
- Pin every upstream repository to an exact commit.
- Merge only after the full test suite passes.

---

### Task 1: PyramidMamba native adapter

**Files:**
- Create: `oemseg/models/pyramidmamba.py`
- Modify: `oemseg/models/registry.py`
- Modify: `oemseg/config.py`
- Modify: `scripts/setup_unetformer.sh`
- Test: `tests/test_models.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `.vendor/GeoSeg/geoseg/models/PyramidMamba.py` at the existing GeoSeg revision.
- Produces: `build_model_from_values("pyramidmamba", ...) -> SegmentationModelAdapter`.

- [ ] Add a failing config test requiring `pyramidmamba` registration/default variant.
- [ ] Add a failing fake-upstream adapter contract test for full-resolution 9-class logits and parameter groups.
- [ ] Run only the new tests and confirm they fail because PyramidMamba is not registered/implemented.
- [ ] Implement the minimal adapter/registry/default-variant change and rename the GeoSeg setup script messaging to reflect shared GeoSeg use without changing its pin.
- [ ] Run the new tests and existing model/config tests to green.

### Task 2: Pinned official sidecar registry and CLI

**Files:**
- Create: `oemseg/upstreams.py`
- Create: `scripts/paper_models.py`
- Test: `tests/test_paper_models.py`

**Interfaces:**
- Produces: immutable upstream specs for `geosa_basa`, `hg_rsovsseg`, `repstdc`; `ensure_checkout(name)`, `build_train_command(name, passthrough)`, and `build_eval_command(name, checkpoint, passthrough)`.

- [ ] Write failing tests for exact repo/revision pins and normalized model aliases.
- [ ] Write failing tests for train/eval dry-run command construction and unsupported operations.
- [ ] Run `tests/test_paper_models.py` and confirm RED.
- [ ] Implement minimal standard-library registry, safe checkout helper and CLI.
- [ ] Run `tests/test_paper_models.py` to GREEN.

### Task 3: Documentation and upstream verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_orchestration.py`

**Interfaces:**
- Documents commands for all four new models and the distinction between local-protocol PyramidMamba and official-protocol sidecars.

- [ ] Add a failing orchestration/doc test requiring all four model names, setup commands and pinned-source wording.
- [ ] Run it and confirm RED.
- [ ] Update README minimally with setup/train/eval examples and protocol caveats.
- [ ] Run the doc/orchestration test to GREEN.
- [ ] Execute sidecar `--dry-run` commands for all supported operations and verify exact official paths.

### Task 4: Full verification, merge and sync

**Files:** no new production files.

- [ ] Run `python -m pytest -q` in the prepared project environment and require all tests to pass.
- [ ] Run `git diff --check`, inspect `git diff`, and confirm `.vendor/`/datasets are not tracked.
- [ ] Clone/setup each official upstream via the wrapper and verify each checkout HEAD equals its pinned revision; reuse the existing GeoSeg pin for PyramidMamba.
- [ ] Commit implementation on `feat/q1-oem-models`.
- [ ] Merge the feature branch into local `main` without discarding unrelated work.
- [ ] Push `main` to `origin`.
- [ ] Fetch and compare local `main`, `origin/main`, and GitHub main SHA; all must match.
