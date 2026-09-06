#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${OEM_OPENMMLAB_ENV:-oem-openmmlab}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_EXE:-${HOME}/miniconda3/bin/conda}"

if [[ ! -x "$CONDA_BIN" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "conda not found; expected ~/miniconda3/bin/conda or CONDA_EXE" >&2
  exit 1
fi

cd "$ROOT"
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" create -y -n "$ENV_NAME" python=3.10 pip
fi

run() { "$CONDA_BIN" run -n "$ENV_NAME" "$@"; }

# Keep OpenMMLab off work-env: its compiled MMCV wheel targets Torch 2.0/CUDA 11.7.
run python -m pip install --upgrade pip "setuptools<81" tifffile
run python -m pip install \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu117
run python -m pip install \
  "numpy<2" \
  "segmentation-models-pytorch>=0.5,<0.6" \
  "rasterio>=1.4,<2" \
  "wandb>=0.28,<0.29" \
  "tqdm>=4.67,<5" \
  "accelerate>=1.10,<2" \
  "timm==1.0.15" \
  "einops==0.8.1"
run python -m pip install "numpy<2" "opencv-python<4.12" \
  mmengine==0.10.7 mmsegmentation==1.1.2 mmpretrain==1.2.0
run python -m pip install "numpy<2" "opencv-python<4.12" mmcv==2.0.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
run python scripts/paper_models.py setup repstdc

run python -c 'import numpy, torch, mmcv, mmengine, mmseg, mmpretrain; from mmcv.ops import sigmoid_focal_loss; from mmseg.models.backbones import MSCAN; from mmseg.models.decode_heads import LightHamHead; print("numpy", numpy.__version__); print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available()); print("mmcv", mmcv.__version__); print("mmengine", mmengine.__version__); print("mmseg", mmseg.__version__); print("mmpretrain", mmpretrain.__version__); print("components", sigmoid_focal_loss.__name__, MSCAN.__name__, LightHamHead.__name__)'

echo "Ready. Run SegNeXt/RepSTDC with: $CONDA_BIN run -n $ENV_NAME python train.py --model <segnext|repstdc> ..."
