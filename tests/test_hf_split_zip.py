#!/usr/bin/env python3
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_xbd_from_hf import PART_SIZE, parse_zip64_extra, part_ranges


def main() -> None:
    assert part_ranges(PART_SIZE - 10, 20) == [(0, PART_SIZE - 10, PART_SIZE - 1), (1, 0, 9)]
    extra = struct.pack("<HHQQQ", 1, 24, 123, 45, 678)
    assert parse_zip64_extra(extra, True, True, True) == (123, 45, 678)
    print("split zip checks: OK")


if __name__ == "__main__":
    main()
