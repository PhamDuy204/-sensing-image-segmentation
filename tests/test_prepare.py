#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_oem_xbd import write_geotiff


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        png = root / "image.png"
        tif = root / "image.tif"
        Image.fromarray(np.zeros((8, 9, 3), dtype=np.uint8)).save(png)
        write_geotiff(png, None, tif)
        with Image.open(tif) as image:
            assert image.size == (9, 8) and image.mode == "RGB"
    print("prepare checks: OK")


if __name__ == "__main__":
    main()
