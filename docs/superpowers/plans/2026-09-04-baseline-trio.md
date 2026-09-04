# Baseline Trio Implementation Plan

**Goal:** Add trainable SegNeXt-T, RepSTDC-CA, and Mask2Former-Swin-Tiny baselines to the shared OEM trainer.

**Spec:** `docs/superpowers/specs/2026-09-04-baseline-trio-design.md`

## Tasks
1. Add failing registry/adapter/native-loss tests for the three models.
2. Add SegNeXt adapter using MMSegmentation MSCAN + LightHamHead.
3. Add RepSTDC adapter using pinned `.vendor/RepSTDC/mmseg_geo` source.
4. Add Mask2Former adapter using Transformers and preserve its native Hungarian loss during training.
5. Add the minimal trainer branch for models that expose native training loss.
6. Add a setup script for a compatible OpenMMLab environment without modifying `work-env`.
7. Update CLI defaults and README commands/baseline table.
8. Run focused tests, full tests, real forward/backward smoke checks, then merge to `main` and push.
