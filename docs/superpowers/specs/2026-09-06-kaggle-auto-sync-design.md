# Kaggle Paper-Repro Auto-Sync Design

## Goal
Run one OpenEarthMap model on one authorized Kaggle account entirely headlessly: create/push a private T4x2 notebook, wait for completion, download outputs, and sync the W&B offline run automatically from the PC without putting the W&B key in Kaggle.

## Architecture
- `scripts/kaggle_paper_repro.sh` is the Kaggle-side, paper-faithful training entrypoint. It validates 3000/500/1500 splits, 2 GPUs, installs only the model-specific environment, trains with global batch 2, FP32, Poly/AdamW defaults, no internal validation, `--loss auto`, and W&B offline mode.
- `scripts/kaggle_pipeline.py` is the PC-side stdlib orchestrator. It reads one local Kaggle access token file, generates notebook + `kernel-metadata.json`, invokes the official Kaggle CLI, polls kernel status, downloads outputs, then invokes `wandb sync` locally. Credentials are passed only through subprocess environment and never written into notebook metadata/source.
- The orchestrator supports `--smoke` for a fast integration version before a 45-epoch run, and a foreground mode that owns the full lifecycle. A detached launcher uses the same command and log/state directory so the user does not have to watch it.

## Constraints
- One authorized Kaggle account per run; no multi-account quota pooling.
- Dataset source: `duy18102004/oem-dataset`.
- Accelerator: `NvidiaTeslaT4`, GPU enabled, internet enabled, private kernel.
- Paper protocol: 1024 input, 45 epochs, AdamW LR 6e-4 / encoder LR 6e-5, weight decay 0.01, Poly power 0.9, warmup 5, effective global batch 2, FP32, full 3000-image train split, official 500-image validation used for reported comparison, TTA enabled.
- `--loss auto` selects the published/native loss per model.
- W&B mode on Kaggle is `offline`; sync occurs only on the PC using existing local W&B credentials.
- No token, API key, `.netrc`, or generated credential-bearing file is committed or printed.
