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
import uuid
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
TERMINAL_STATUSES = {"COMPLETE", "ERROR", "CANCELLED", "CANCEL_ACKNOWLEDGED"}


def chunk_end_epochs(total_epochs: int, chunk_epochs: int) -> list[int]:
    if total_epochs < 1 or chunk_epochs < 1:
        raise ValueError("total_epochs and chunk_epochs must be >= 1")
    return list(range(chunk_epochs, total_epochs, chunk_epochs)) + [total_epochs]


def build_kernel_files(
    *,
    owner: str,
    slug: str,
    model: str,
    smoke: bool,
    repo_ref: str,
    chunk_end_epoch: int | None = None,
    previous_kernel: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    code_file = f"{slug}.ipynb"
    repo_dir = "/kaggle/tmp/OEM_Segmentation"
    machine_shape = "NvidiaTeslaP100" if model == "unet" else "NvidiaTeslaT4"
    accelerator_kind = "P100" if model == "unet" else "T4X2"
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
        (
            f"ACCELERATOR_KIND={accelerator_kind} "
            f"MODEL_NAME={shlex.quote(model)} "
            f"SMOKE={'1' if smoke else '0'} "
            f"CHUNK_END_EPOCH={chunk_end_epoch or 0} "
            f"RESUME_FROM_INPUT={'1' if previous_kernel else '0'} "
            "bash scripts/kaggle_paper_repro.sh\n"
        ),
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
        "machine_shape": machine_shape,
        "dataset_sources": [DATASET_SOURCE],
        "competition_sources": [],
        "kernel_sources": [previous_kernel] if previous_kernel else [],
        "model_sources": [],
    }
    return notebook, metadata


def normalize_status(text: str) -> str:
    upper = text.upper()
    for status in (
        "CANCEL_ACKNOWLEDGED",
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


def resolve_repo_ref(repo_root: Path, repo_ref: str) -> str:
    result = _run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{repo_ref}^{{commit}}"],
        capture=True,
    )
    resolved = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise RuntimeError(f"could not resolve repository ref {repo_ref!r} to a commit SHA")
    return resolved


def wandb_sync_command(
    wandb_bin: Path,
    run: Path,
    *,
    target_id: str | None = None,
    append: bool = False,
) -> list[str]:
    command = [str(wandb_bin), "sync"]
    if target_id is not None:
        command.append("--legacy")
        command += ["--id", target_id]
    if append:
        command.append("--append")
    command.append(str(run))
    return command


def _sync_wandb(
    wandb_bin: Path,
    output_dir: Path,
    *,
    target_id: str | None = None,
    append: bool = False,
) -> list[str]:
    runs = find_offline_runs(output_dir)
    if not runs:
        raise RuntimeError(f"no offline W&B runs found under {output_dir}")
    synced = []
    for run in runs:
        _run(wandb_sync_command(wandb_bin, run, target_id=target_id, append=append))
        synced.append(str(run))
    return synced


def _state_update(path: Path, state: dict[str, object], **updates: object) -> None:
    state.update(updates)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(path, state)


def _run_kernel_once(
    *,
    args: argparse.Namespace,
    owner: str,
    token: str,
    kaggle_bin: Path,
    wandb_bin: Path,
    slug: str,
    run_root: Path,
    chunk_end_epoch: int | None = None,
    previous_kernel: str | None = None,
    wandb_target_id: str | None = None,
    wandb_append: bool = False,
) -> dict[str, object]:
    kernel = f"{owner}/{slug}"
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
        chunk_end_epoch=chunk_end_epoch,
        previous_kernel=previous_kernel,
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
        "machine_shape": metadata["machine_shape"],
        "chunk_end_epoch": chunk_end_epoch,
        "previous_kernel": previous_kernel,
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
        [
            str(kaggle_bin),
            "kernels",
            "push",
            "-p",
            str(kernel_dir),
            "--accelerator",
            str(metadata["machine_shape"]),
        ],
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
        [
            str(kaggle_bin),
            "kernels",
            "output",
            kernel,
            "-p",
            str(output_dir),
            "-o",
            "-q",
            "--file-pattern",
            r"^oem_outputs/",
        ],
        env=kaggle_env,
    )
    _state_update(state_path, state, status="DOWNLOADED")

    print("Syncing offline W&B run(s)", flush=True)
    synced = _sync_wandb(
        wandb_bin,
        output_dir,
        target_id=wandb_target_id,
        append=wandb_append,
    )
    _state_update(state_path, state, status="SYNCED", synced_wandb_runs=synced)
    print(f"DONE: {kernel}; synced {len(synced)} W&B run(s); state={state_path}")
    return {
        "kernel": kernel,
        "output_dir": output_dir,
        "state_path": state_path,
        "synced_wandb_runs": synced,
    }


def _foreground(args: argparse.Namespace) -> int:
    args.repo_ref = resolve_repo_ref(Path(__file__).resolve().parents[1], args.repo_ref)
    token_file = args.token_file.expanduser().resolve()
    owner, token = _load_account(token_file)
    client_dir = args.client_dir.expanduser().resolve()
    kaggle_bin = _tool(client_dir, "kaggle")
    wandb_bin = _tool(client_dir, "wandb")

    suffix = "-smoke" if args.smoke else ""
    base_slug = args.slug or f"oem-{args.model}-paper-repro{suffix}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    state_root = args.state_root.expanduser().resolve()

    if not args.chunk_epochs:
        run_root = state_root / base_slug / timestamp
        _run_kernel_once(
            args=args,
            owner=owner,
            token=token,
            kaggle_bin=kaggle_bin,
            wandb_bin=wandb_bin,
            slug=base_slug,
            run_root=run_root,
        )
        return 0

    run_root = state_root / base_slug / timestamp
    run_root.mkdir(parents=True)
    state_path = run_root / "state.json"
    wandb_target_id = uuid.uuid4().hex[:8]
    state: dict[str, object] = {
        "model": args.model,
        "smoke": args.smoke,
        "repo_ref": args.repo_ref,
        "chunk_epochs": args.chunk_epochs,
        "wandb_run_id": wandb_target_id,
        "run_root": str(run_root),
        "status": "CHUNKING",
        "chunks": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _state_update(state_path, state)

    previous_kernel = None
    chunks: list[dict[str, object]] = []
    for part, end_epoch in enumerate(chunk_end_epochs(45, args.chunk_epochs), start=1):
        slug = f"{base_slug}-part{part}"
        part_root = run_root / f"part-{part:02d}"
        _state_update(
            state_path,
            state,
            status=f"PART_{part}_SUBMITTING",
            active_part=part,
            active_kernel=f"{owner}/{slug}",
        )
        result = _run_kernel_once(
            args=args,
            owner=owner,
            token=token,
            kaggle_bin=kaggle_bin,
            wandb_bin=wandb_bin,
            slug=slug,
            run_root=part_root,
            chunk_end_epoch=end_epoch,
            previous_kernel=previous_kernel,
            wandb_target_id=wandb_target_id,
            wandb_append=part > 1,
        )
        markers = list(Path(result["output_dir"]).rglob("chunk_state.json"))
        if len(markers) != 1:
            raise RuntimeError(
                f"expected exactly one chunk_state.json for {result['kernel']}, got {len(markers)}"
            )
        chunk_state = json.loads(markers[0].read_text())
        chunks.append(
            {
                "part": part,
                "kernel": result["kernel"],
                "end_epoch": end_epoch,
                "chunk_state": chunk_state,
            }
        )
        complete = bool(chunk_state.get("complete"))
        _state_update(
            state_path,
            state,
            status="SYNCED" if complete else f"PART_{part}_SYNCED",
            chunks=chunks,
            completed_epoch=chunk_state.get("epoch"),
        )
        if complete:
            print(
                f"DONE: {args.model} completed at epoch {chunk_state.get('epoch')} "
                f"across {part} Kaggle part(s); state={state_path}"
            )
            return 0
        previous_kernel = str(result["kernel"])

    raise RuntimeError(f"chunk plan ended without a completed training marker: {state_path}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--detach", action="store_true", help="run the watcher in the background")
    parser.add_argument("--repo-ref", default="main", help="Git branch/tag/SHA fetched by the Kaggle notebook")
    parser.add_argument("--slug", default=None)
    parser.add_argument(
        "--chunk-epochs",
        type=int,
        default=0,
        help="split a 45-epoch paper run into committed Kaggle parts; 0 disables chunking",
    )
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
    if args.chunk_epochs < 0:
        raise SystemExit("--chunk-epochs must be >= 0")
    if args.smoke and args.chunk_epochs:
        raise SystemExit("--chunk-epochs is only valid for full runs")
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
