# Q1 OpenEarthMap model integrations design

## Goal
Add four recent OpenEarthMap paper implementations to the existing OEM research project while reusing official upstream code and preserving the current training/evaluation pipeline wherever the upstream architecture can cleanly satisfy it.

## Scope
- PyramidMamba (GeoSeg): native project adapter because the already-pinned GeoSeg source contains the model and fits the existing `images -> logits` contract.
- GeoSA-BaSA: official upstream sidecar runner, pinned at `67cec152349db6b29bf14e991e80f01fd365f73c`.
- HG-RSOVSSeg: official upstream sidecar runner, pinned at `58e1df1f68f15920e02320dc54fe7441acf37770`.
- RepSTDC: official upstream sidecar runner, pinned at `549c2e0cf81963aeae8e5c6b9c83a3aa496a8407`.
  This official historical revision is intentional: it contains the same CA OpenEarthMap config as the current upstream head while still retaining the referenced `config/_base_` files, so the published config is self-contained.

GeoSeg remains pinned at the project's existing revision `9453fe48209c4626b29e35e61bab93b61212c4b1` so UNetFormer and PyramidMamba share one upstream checkout.

## Architecture
PyramidMamba becomes a first-class `oemseg` model adapter: it returns full-resolution 9-class logits and exposes backbone/main parameter groups through the existing `SegmentationModelAdapter` contract. No PyramidMamba architecture code is copied into this repository.

The other three models keep their official OpenMMLab/training stacks. A single project-side helper manages pinned vendor checkouts and constructs the official train/eval commands. This avoids rewriting loss/data-sample/open-vocabulary/domain-generalization behavior into the local trainer and keeps reproduction close to the papers.

## Vendor management
Create one declarative Python module containing repository URL, pinned revision, destination and official OEM entrypoints. Setup commands clone/fetch only the requested repository and detach at the pinned revision. `.vendor/` stays gitignored; this repository stores no copied third-party model source.

The helper must be idempotent: an already-correct checkout is left untouched; a wrong revision is corrected. Existing non-git content at a destination is rejected instead of deleted silently.

## CLI
Add `scripts/paper_models.py` with subcommands:
- `setup <model|all>`: materialize pinned official source in `.vendor/`.
- `train <model> [-- ...upstream args]`: print/execute the official upstream training command.
- `eval <model> [--checkpoint PATH] [-- ...upstream args]`: print/execute the official evaluation command when the upstream project exposes one.
- `--dry-run`: never executes subprocesses and is sufficient for CI/unit tests.

PyramidMamba is trained through the existing `train.py --model pyramidmamba`; sidecar CLI is only for GeoSA-BaSA, HG-RSOVSSeg and RepSTDC.

## Dataset handling
The project dataset remains the source of truth. The sidecar CLI does not duplicate data. Where upstream expects a different directory layout or preprocessing, documentation points to the official conversion command and lets the user pass/prepare the dataset path; no lossy automatic remapping is hidden inside the wrapper.

For PyramidMamba, the existing OpenEarthMap loader/splits and 9-class convention remain unchanged, so comparisons with current models use the same project protocol.

## Dependencies
Do not add OpenMMLab stacks to the main `requirements.txt`. GeoSA-BaSA, HG-RSOVSSeg and RepSTDC have mutually older/different dependency constraints and should use isolated environments described by their official repos. The project wrapper itself uses only the Python standard library.

PyramidMamba relies on dependencies already required by GeoSeg/Mamba integration. Setup documentation tells the user to run the existing environment setup before the model smoke test.

## Testing
- Registry/config tests prove PyramidMamba is discoverable and gets a stable default variant.
- A fake-upstream adapter test proves PyramidMamba's adapter contract without downloading weights.
- Sidecar tests verify exact repository pins, idempotent/dry-run command construction, official OEM config/entrypoint selection, and rejection of unsupported operations.
- Existing full suite remains green.
- Runtime smoke checks clone the lightweight filtered upstream checkouts and verify pinned revisions/files. PyramidMamba gets an import/forward smoke check if the installed environment supports its native dependencies; otherwise the adapter test remains the deterministic gate and the missing runtime dependency is reported explicitly.

## Merge and synchronization
Implementation occurs on `feat/q1-oem-models` in an isolated worktree. After all tests pass, merge into local `main`, push `origin/main`, then verify the local `main`, `origin/main`, and GitHub default branch resolve to the same commit SHA.
