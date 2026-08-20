#!/usr/bin/env python3
"""Range-extract only OEM-required xBD PNGs from a split ZIP on Hugging Face."""

from __future__ import annotations

import argparse
import binascii
import csv
import json
import struct
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPOSITORY = "iamzihan/xView2"
REVISION = "0619541b03efe77d820ae172d26cb5d6711fc325"
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
PART_SIZE = 2_147_483_648
PART_SIZES = [PART_SIZE] * 15 + [357_845_677]
TOTAL_SIZE = sum(PART_SIZES)
CENTRAL_HEADER = struct.Struct("<4s6H3I5H2I")
LOCAL_HEADER = struct.Struct("<4s5H3I2H")


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


def part_ranges(offset: int, length: int) -> list[tuple[int, int, int]]:
    """Map a global concatenated-file range to inclusive ranges in split parts."""
    if offset < 0 or length < 0 or offset + length > TOTAL_SIZE:
        raise ValueError(f"Invalid global range offset={offset} length={length}")
    result: list[tuple[int, int, int]] = []
    remaining, local_offset = length, offset
    for index, size in enumerate(PART_SIZES):
        if local_offset >= size:
            local_offset -= size
            continue
        take = min(remaining, size - local_offset)
        if take:
            result.append((index, local_offset, local_offset + take - 1))
            remaining -= take
        if not remaining:
            break
        local_offset = 0
    return result


def part_url(index: int) -> str:
    return f"{BASE_URL}/images_part_{index:02d}?download=true"


def http_range(index: int, start: int, end: int, retries: int = 6) -> bytes:
    request = urllib.request.Request(
        part_url(index),
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "OEM-xBD-range-extractor/1.0"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
            expected = end - start + 1
            if len(data) != expected:
                raise IOError(f"Part {index}: expected {expected} bytes, received {len(data)}")
            return data
        except (OSError, urllib.error.HTTPError) as error:
            if attempt + 1 == retries:
                raise
            wait = min(30, 2**attempt)
            print(f"retry part={index} range={start}-{end} in {wait}s: {error}")
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_global(offset: int, length: int) -> bytes:
    return b"".join(http_range(index, start, end) for index, start, end in part_ranges(offset, length))


def parse_zip64_extra(
    extra: bytes, need_uncompressed: bool, need_compressed: bool, need_offset: bool
) -> tuple[int | None, int | None, int | None]:
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id, size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        payload = extra[cursor : cursor + size]
        cursor += size
        if field_id != 1:
            continue
        values: list[int | None] = []
        payload_cursor = 0
        for needed in (need_uncompressed, need_compressed, need_offset):
            if needed:
                if payload_cursor + 8 > len(payload):
                    raise ValueError("Truncated ZIP64 extra field")
                values.append(struct.unpack_from("<Q", payload, payload_cursor)[0])
                payload_cursor += 8
            else:
                values.append(None)
        return tuple(values)  # type: ignore[return-value]
    raise ValueError("Missing ZIP64 extra field")


def central_directory_location() -> tuple[int, int]:
    tail_size = min(131_072, PART_SIZES[-1])
    tail = http_range(15, PART_SIZES[-1] - tail_size, PART_SIZES[-1] - 1)
    position = tail.rfind(b"PK\x06\x06")
    if position < 0:
        raise ValueError("ZIP64 end-of-central-directory record not found")
    record = tail[position : position + 56]
    if len(record) < 56:
        raise ValueError("Truncated ZIP64 end-of-central-directory record")
    unpacked = struct.unpack("<4sQ2H2I4Q", record)
    return unpacked[-1], unpacked[-2]


def parse_central_directory() -> list[ZipEntry]:
    offset, size = central_directory_location()
    data = fetch_global(offset, size)
    entries: list[ZipEntry] = []
    cursor = 0
    while cursor < len(data):
        if cursor + CENTRAL_HEADER.size > len(data):
            raise ValueError("Truncated central directory")
        fields = CENTRAL_HEADER.unpack_from(data, cursor)
        if fields[0] != b"PK\x01\x02":
            raise ValueError(f"Invalid central directory signature at {cursor}")
        flags, method, crc32 = fields[3], fields[4], fields[7]
        compressed, uncompressed = fields[8], fields[9]
        name_length, extra_length, comment_length = fields[10], fields[11], fields[12]
        local_offset = fields[16]
        start = cursor + CENTRAL_HEADER.size
        name_bytes = data[start : start + name_length]
        extra = data[start + name_length : start + name_length + extra_length]
        name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        zip64 = compressed == 0xFFFFFFFF or uncompressed == 0xFFFFFFFF or local_offset == 0xFFFFFFFF
        if zip64:
            z_uncompressed, z_compressed, z_offset = parse_zip64_extra(
                extra,
                uncompressed == 0xFFFFFFFF,
                compressed == 0xFFFFFFFF,
                local_offset == 0xFFFFFFFF,
            )
            uncompressed = z_uncompressed if z_uncompressed is not None else uncompressed
            compressed = z_compressed if z_compressed is not None else compressed
            local_offset = z_offset if z_offset is not None else local_offset
        entries.append(ZipEntry(name, method, crc32, compressed, uncompressed, local_offset))
        cursor = start + name_length + extra_length + comment_length
    return entries


def required_names(csv_path: Path) -> set[str]:
    with csv_path.open(newline="") as file:
        return {source.strip() for source, _ in csv.reader(file) if source.strip()}


def select_entries(entries: list[ZipEntry], required: set[str]) -> dict[str, ZipEntry]:
    selected: dict[str, ZipEntry] = {}
    for entry in entries:
        basename = Path(entry.name).name
        if basename not in required or "/images/" not in entry.name.replace("\\", "/"):
            continue
        if basename in selected:
            raise ValueError(f"Duplicate required filename in ZIP: {basename}")
        selected[basename] = entry
    return selected


def valid_png(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except OSError:
        return False


def extract_entry(entry: ZipEntry, destination: Path) -> dict[str, object]:
    output = destination / "images" / Path(entry.name).name
    if valid_png(output):
        return {**asdict(entry), "output": str(output), "status": "existing"}
    header = fetch_global(entry.local_offset, LOCAL_HEADER.size)
    fields = LOCAL_HEADER.unpack(header)
    if fields[0] != b"PK\x03\x04":
        raise ValueError(f"Invalid local header for {entry.name}")
    name_length, extra_length = fields[-2], fields[-1]
    data_offset = entry.local_offset + LOCAL_HEADER.size + name_length + extra_length
    compressed = fetch_global(data_offset, entry.compressed_size)
    if entry.method == 0:
        content = compressed
    elif entry.method == 8:
        content = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"Unsupported ZIP method {entry.method} for {entry.name}")
    if len(content) != entry.uncompressed_size:
        raise ValueError(f"Size mismatch for {entry.name}")
    if binascii.crc32(content) & 0xFFFFFFFF != entry.crc32:
        raise ValueError(f"CRC mismatch for {entry.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_bytes(content)
    if not valid_png(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Invalid PNG payload for {entry.name}")
    temporary.replace(output)
    return {**asdict(entry), "output": str(output), "status": "downloaded"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("datasets/OpenEarthMap/OpenEarthMap_wo_xBD/xbd_files.csv"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("datasets/OpenEarthMap/xBD_huggingface/extracted"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    required = required_names(args.csv)
    print("reading remote ZIP central directory...")
    selected = select_entries(parse_central_directory(), required)
    missing = sorted(required - selected.keys())
    compressed_gib = sum(entry.compressed_size for entry in selected.values()) / 1024**3
    print(f"required={len(required)} found={len(selected)} missing={len(missing)} compressed={compressed_gib:.2f} GiB")
    if missing:
        raise FileNotFoundError(f"Required files absent from mirror: {missing[:20]}")
    if args.list_only:
        return

    args.destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_entry, entry, args.destination): name for name, entry in selected.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="extract xBD"):
            manifest.append(future.result())
    manifest.sort(key=lambda item: str(item["name"]))
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"xBD extraction complete: {len(manifest)} images")


if __name__ == "__main__":
    main()
