#!/usr/bin/env python3
"""OpenEarthMap semantic-segmentation training entry point."""

from oemseg.config import parse_args
from oemseg.engine.trainer import run_training


if __name__ == "__main__":
    run_training(parse_args())
