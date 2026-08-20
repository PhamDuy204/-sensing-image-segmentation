#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="$ROOT_DIR/datasets/OpenEarthMap"
SOURCE_DIR="$DATA_DIR/OpenEarthMap_wo_xBD"
DEST_DIR="$DATA_DIR/OpenEarthMap"
ARCHIVE="$DATA_DIR/OpenEarthMap.zip"
ARCHIVE_PART="$ARCHIVE.part"
DOWNLOAD_URL="https://zenodo.org/records/7223446/files/OpenEarthMap.zip?download=1"
EXPECTED_MD5="64155d1dc9d3b68536063f79878e1a67"

usage() {
  echo "Usage: $0 [--verify-only]" >&2
  exit 2
}

for cmd in python3 unzip md5sum; do
  command -v "$cmd" >/dev/null || { echo "Missing required command: $cmd" >&2; exit 2; }
done

python3 - <<'PY' || {
import numpy  # noqa: F401
import rasterio  # noqa: F401
from PIL import Image  # noqa: F401
from tqdm import tqdm  # noqa: F401
PY
  echo "Missing Python dataset dependencies. Activate the project environment and install requirements first:" >&2
  echo "  python -m pip install --no-build-isolation -r requirements.txt" >&2
  exit 2
}

if [[ $# -gt 1 ]]; then
  usage
fi
if [[ ${1:-} == "--verify-only" ]]; then
  exec python3 scripts/prepare_oem_xbd.py --verify-only
elif [[ $# -eq 1 ]]; then
  usage
fi

mkdir -p "$DATA_DIR"

if [[ -f "$DEST_DIR/train.txt" && -f "$DEST_DIR/val.txt" && -f "$DEST_DIR/test.txt" ]]; then
  if python3 scripts/prepare_oem_xbd.py --verify-only; then
    echo "OpenEarthMap dataset is already prepared."
    exit 0
  fi
  echo "Existing prepared dataset is incomplete; rebuilding it." >&2
fi

if [[ ! -f "$SOURCE_DIR/xbd_files.csv" ]]; then
  if [[ ! -f "$ARCHIVE" ]]; then
    echo "Downloading official OpenEarthMap archive from Zenodo..."
    if command -v curl >/dev/null; then
      curl -fL --retry 5 --retry-all-errors -C - -o "$ARCHIVE_PART" "$DOWNLOAD_URL"
    elif command -v wget >/dev/null; then
      wget -c -O "$ARCHIVE_PART" "$DOWNLOAD_URL"
    else
      echo "Need curl or wget to download OpenEarthMap." >&2
      exit 2
    fi
    mv "$ARCHIVE_PART" "$ARCHIVE"
  fi

  echo "$EXPECTED_MD5  $ARCHIVE" | md5sum -c -

  EXTRACT_DIR="$DATA_DIR/.extract-openearthmap-$$"
  rm -rf "$EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR"
  trap 'rm -rf "$EXTRACT_DIR"' EXIT
  unzip -q "$ARCHIVE" -d "$EXTRACT_DIR"

  mapfile -t CANDIDATES < <(find "$EXTRACT_DIR" -type f -name xbd_files.csv -printf '%h\n')
  if [[ ${#CANDIDATES[@]} -ne 1 ]]; then
    echo "Expected exactly one extracted OpenEarthMap root, found ${#CANDIDATES[@]}." >&2
    exit 1
  fi

  rm -rf "$SOURCE_DIR"
  mv "${CANDIDATES[0]}" "$SOURCE_DIR"
  rm -rf "$EXTRACT_DIR"
  trap - EXIT
  rm -f "$ARCHIVE"
fi

python3 scripts/extract_xbd_from_hf.py
python3 scripts/prepare_oem_xbd.py
python3 scripts/prepare_oem_xbd.py --verify-only

echo
echo "Dataset ready at: datasets/OpenEarthMap/OpenEarthMap"
echo "Train with, for example: python train.py --model unetpp"
