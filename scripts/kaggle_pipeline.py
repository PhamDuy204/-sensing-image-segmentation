#!/usr/bin/env python3
"""Submit one paper-repro Kaggle notebook, wait for it, download it, then sync W&B."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://github.com/PhamDuy204/-sensing-image-segmentation.git"
DATASET_SOURCE = "duy18102004/oem-dataset"
MODELS = (
    "unet",
    "unetformer",
    "segformer",
    "segnext",
    "repstdc",
    "mambavision",
    "pyramidmamba",
    "mask2former",
)
TERMINAL_STATUSES = {"COMPLETE", "ERROR", "CANCELLED"}


def build_kernel_files(
    *, owner: str, slug: str, model: str, smoke: bool, repo_ref: str
) -> tuple[dict[str, object], dict[str, object]]:
    code_file = f"{slug}.ipynb"
    repo_dir = "/kaggle/working/OEM_Segmentation"
    source = [
        "%%bash\n",
        "set -Eeuo pipefail\n",
        f"REPO_URL={shlex.quote(REPO_URL)}\n",
        f"REPO_REF={shlex.quote(repo_ref)}\n",
        f"REPO_DIR={shlex.quote(repo_dir)}\n",
        "rm -rf \"$REPO_DIR\"\n",
        "git clone --filter=blob:none --no-checkout \"$REPO_URL\" \"$REPO_DIR\"\n",
        "git -C \"$REPO_DIR\" fetch --depth 1 origin \"$REPO_REF\"\n",
        "git -C \"$REPO_DIR\" checkout --detach FETCH_HEAD\n",
        "cd \"$REPO_DIR\"\n",
        "echo \"repo_head=$(git rev-parse HEAD)\"\n",
        f"MODEL_NAME={shlex.quote(model)} SMOKE={'1' if smoke else '0'} "
        "bash scripts/kaggle_paper_repro.sh\n",
    ]
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": f"{owner}/{slug}",
        "title": slug.replace("-", " "),
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET_SOURCE],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    return notebook, metadata


def normalize_status(text: str) -> str:
    upper = text.upper()
    for status in (
        "CANCEL_REQUESTED",
        "COMPLETE",
        "CANCELLED",
        "ERROR",
        "RUNNING",
        "QUEUED",
        "PENDING",
    ):
        if re.search(rf"\b{status}\b", upper):
            return status
    raise ValueError(f"unrecognized Kaggle kernel status: {text.strip()!r}")


def find_offline_runs(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("offline-run-*") if path.is_dir())


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _load_account(token_file: Path) -> tuple[str, str]:
    token = token_file.read_text().strip()
    if not token:
        raise RuntimeError(f"empty Kaggle access token: {token_file}")
    index_file = token_file.parent.parent / "index.json"
    index = json.loads(index_file.read_text())
    account = index.get(token_file.parent.name)
    if not isinstance(account, dict) or not account.get("username"):
        raise RuntimeError(f"missing username for {token_file.parent.name} in {index_file}")
    return str(account["username"]), token


def _run(command: list[str], *, env: dict[str, str] | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def _tool(client_dir: Path, name: str) -> Path:
    path = client_dir / "bin" / name
    if not path.is_file():
        raise RuntimeError(f"missing {name}: {path}; install the local Kaggle client first")
    return path


def _sync_wandb(wandb_bin: Path, output_dir: Path) -> list[str]:
    runs = find_offline_runs(output_dir)
    if not runs:
        raise RuntimeError(f"no offline W&B runs found under {output_dir}")
    synced = []
    for run in runs:
        _run([str(wandb_bin), "sync", str(run)])
        synced.append(str(run))
    return synced


def _state_update(path: Path, state: dict[str, object], **updates: object) -> None:
    state.update(updates)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(path, state)


def _foreground(args: argparse.Namespace) -> int:
    token_file = args.token_file.expanduser().resolve()
    owner, token = _load_account(token_file)
    client_dir = args.client_dir.expanduser().resolve()
    kaggle_bin = _tool(client_dir, "kaggle")
    wandb_bin = _tool(client_dir, "wandb")

    suffix = "-smoke" if args.smoke else ""
    slug = args.slug or f"oem-{args.model}-paper-repro{suffix}"
    kernel = f"{owner}/{slug}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = args.state_root.expanduser().resolve() / slug / timestamp
    kernel_dir = run_root / "kernel"
    output_dir = run_root / "output"
    kernel_dir.mkdir(parents=True)
    output_dir.mkdir()

    notebook, metadata = build_kernel_files(
        owner=owner,
        slug=slug,
        model=args.model,
        smoke=args.smoke,
        repo_ref=args.repo_ref,
    )
    notebook_path = kernel_dir / str(metadata["code_file"])
    _write_json(notebook_path, notebook)
    _write_json(kernel_dir / "kernel-metadata.json", metadata)

    state_path = run_root / "state.json"
    state: dict[str, object] = {
        "model": args.model,
        "smoke": args.smoke,
        "repo_ref": args.repo_ref,
        "kernel": kernel,
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "status": "SUBMITTING",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _state_update(state_path, state)

    kaggle_env = os.environ.copy()
    kaggle_env["KAGGLE_API_TOKEN"] = token
    print(f"Submitting {kernel} ({'smoke' if args.smoke else 'full'})")
    _run(
        [str(kaggle_bin), "kernels", "push", "-p", str(kernel_dir), "--accelerator", "NvidiaTeslaT4"],
        env=kaggle_env,
    )
    _state_update(state_path, state, status="SUBMITTED")

    last_status = None
    consecutive_failures = 0
    while True:
        try:
            result = _run(
                [str(kaggle_bin), "kernels", "status", kernel],
                env=kaggle_env,
                capture=True,
            )
            status = normalize_status(result.stdout + "\n" + result.stderr)
            consecutive_failures = 0
        except (subprocess.CalledProcessError, ValueError) as error:
            consecutive_failures += 1
            if consecutive_failures >= 10:
                _state_update(state_path, state, status="STATUS_ERROR", error=str(error))
                raise RuntimeError(f"could not read Kaggle status for {kernel}") from error
            time.sleep(args.poll_seconds)
            continue

        if status != last_status:
            print(f"Kaggle status: {status}", flush=True)
            _state_update(state_path, state, status=status)
            last_status = status
        if status in TERMINAL_STATUSES:
            break
        time.sleep(args.poll_seconds)

    if status != "COMPLETE":
        raise RuntimeError(f"Kaggle run ended with status {status}: {kernel}")

    print("Downloading Kaggle outputs", flush=True)
    _run(
        [str(kaggle_bin), "kernels", "output", kernel, "-p", str(output_dir), "-o"],
        env=kaggle_env,
    )
    _state_update(state_path, state, status="DOWNLOADED")

    print("Syncing offline W&B run(s)", flush=True)
    synced = _sync_wandb(wandb_bin, output_dir)
    _state_update(state_path, state, status="SYNCED", synced_wandb_runs=synced)
    print(f"DONE: {kernel}; synced {len(synced)} W&B run(s); state={state_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--detach", action="store_true", help="run the watcher in the background")
    parser.add_argument("--repo-ref", default="main", help="Git branch/tag/SHA fetched by the Kaggle notebook")
    parser.add_argument("--slug", default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path("~/.config/kaggle/accounts/account_1/access_token"),
    )
    parser.add_argument(
        "--client-dir",
        type=Path,
        default=Path("~/.local/share/oem-kaggle-client"),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("~/.local/state/oem-kaggle"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    if args.poll_seconds < 1:
        raise SystemExit("--poll-seconds must be >= 1")
    if not args.detach:
        return _foreground(args)

    state_root = args.state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = state_root / f"launcher-{args.model}-{timestamp}.log"
    child_args = [item for item in raw if item != "--detach"]
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *child_args],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=Path(__file__).resolve().parents[1],
        )
    print(f"Detached watcher pid={process.pid} log={log_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error
