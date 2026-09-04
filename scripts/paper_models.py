#!/usr/bin/env python3
"""Setup and run pinned official paper model repositories."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oemseg.upstreams import (  # noqa: E402
    UPSTREAMS,
    build_eval_command,
    build_train_command,
    ensure_checkout,
    get_upstream,
)


def _forwarded(args: list[str]) -> list[str]:
    return args[1:] if args[:1] == ["--"] else args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    setup = sub.add_parser("setup", help="clone/pin one official repository or all")
    setup.add_argument("model", help="geosa_basa, hg_rsovsseg, repstdc, or all")
    setup.add_argument("--dry-run", action="store_true")

    train = sub.add_parser("train", help="run the official training entrypoint")
    train.add_argument("model")
    train.add_argument("--dry-run", action="store_true")

    evaluate = sub.add_parser("eval", help="run the official evaluation entrypoint")
    evaluate.add_argument("model")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if args.action == "setup":
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        names = list(UPSTREAMS) if args.model == "all" else [get_upstream(args.model).name]
        for name in names:
            spec = get_upstream(name)
            if args.dry_run:
                print(f"DRY RUN setup {name}: {spec.repo}@{spec.revision} -> {spec.dest}")
            else:
                path = ensure_checkout(name)
                print(f"Pinned {name}: {path} @ {spec.revision}")
        return 0

    spec = get_upstream(args.model)
    passthrough = _forwarded(unknown)
    if args.action == "train":
        command = build_train_command(spec.name, passthrough)
    else:
        command = build_eval_command(spec.name, args.checkpoint, passthrough)
    prefix = "DRY RUN " if args.dry_run else ""
    print(f"{prefix}{spec.name}: {shlex.join(command)}")
    if args.dry_run:
        return 0
    ensure_checkout(spec.name)
    return subprocess.run(command, cwd=spec.dest).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
