#!/usr/bin/env python3
"""Run all OEM baselines across a pool of Kaggle accounts with quota-aware phase resume."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_pipeline as kp

MODELS = kp.MODELS
CHUNKED = frozenset(MODELS) - {"segnext", "repstdc"}
PREFERRED_ACCOUNT = {model: index for index, model in enumerate(MODELS, start=1)}
MIN_GPU_HOURS = 12.0


def _hours(value: str) -> float:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)h\s*", value)
    if not match:
        raise ValueError(f"invalid Kaggle quota value: {value!r}")
    return float(match.group(1))


def _token_file(account: int) -> Path:
    return Path(f"/home/duypham/.config/kaggle/accounts/account_{account}/access_token")


def gpu_quota(account: int, kaggle_bin: Path) -> tuple[str, str, float, str]:
    owner, token = kp._load_account(_token_file(account))
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    result = subprocess.run(
        [str(kaggle_bin), "quota", "--format", "json"],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    gpu = next(row for row in json.loads(result.stdout) if row.get("resource") == "GPU")
    return owner, token, _hours(str(gpu["remaining"])), str(gpu.get("refreshAt") or "")


def phase_plan(model: str) -> list[int | None]:
    return kp.chunk_end_epochs(45, 15) if model in CHUNKED else [None]


def _try_lock(lock_root: Path, account: int):
    lock_root.mkdir(parents=True, exist_ok=True)
    handle = (lock_root / f"account_{account}.lock").open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def choose_account(
    *,
    preferred: int,
    previous_owner: str | None,
    lock_root: Path,
    kaggle_bin: Path,
) -> tuple[int, object, str, str, float]:
    while True:
        candidates = []
        for account in range(1, 9):
            lock = _try_lock(lock_root, account)
            if lock is None:
                continue
            try:
                owner, token, remaining, refresh = gpu_quota(account, kaggle_bin)
            except Exception:
                lock.close()
                continue
            candidates.append((account, lock, owner, token, remaining, refresh))

        eligible = [item for item in candidates if item[4] >= MIN_GPU_HOURS]
        if eligible:
            eligible.sort(
                key=lambda item: (
                    item[2] == previous_owner,
                    item[0] == preferred,
                    item[4],
                ),
                reverse=True,
            )
            selected = eligible[0]
            for item in candidates:
                if item is not selected:
                    item[1].close()
            account, lock, owner, token, remaining, _ = selected
            return account, lock, owner, token, remaining

        for item in candidates:
            item[1].close()
        print("No unlocked account has >=12 GPU-hours; rechecking in 5 minutes", flush=True)
        time.sleep(300)


def _resume_zip(previous_output: Path, staging: Path) -> Path:
    checkpoints = list(previous_output.rglob("last.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one last.pt under {previous_output}, got {len(checkpoints)}")
    run_dir = checkpoints[0].parent
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / "resume.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in ("last.pt", "metrics.jsonl", "best_train_loss.pt", "best_val_miou.pt"):
            path = run_dir / name
            if path.is_file():
                zf.write(path, path.relative_to(run_dir))
        state_dir = run_dir / "accelerator_state"
        if state_dir.is_dir():
            for path in state_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(run_dir))
    return archive


def create_resume_dataset(
    *,
    model: str,
    part: int,
    owner: str,
    token: str,
    previous_output: Path,
    relay_root: Path,
    kaggle_bin: Path,
) -> str:
    staging = relay_root / f"part-{part:02d}-{uuid.uuid4().hex[:6]}"
    _resume_zip(previous_output, staging)
    slug = f"oem-{model}-resume-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    dataset = f"{owner}/{slug}"
    (staging / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": slug, "id": dataset, "licenses": [{"name": "CC0-1.0"}]},
            indent=2,
        )
        + "\n"
    )
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    subprocess.run(
        [str(kaggle_bin), "datasets", "create", "-p", str(staging), "-q"],
        env=env,
        check=True,
        text=True,
    )
    return dataset


def _token_for_owner(owner: str) -> str:
    for account in range(1, 9):
        token_file = _token_file(account)
        if not token_file.is_file():
            continue
        try:
            account_owner, token = kp._load_account(token_file)
        except Exception:
            continue
        if account_owner == owner:
            return token
    raise RuntimeError(f"no configured Kaggle token found for owner {owner!r}")


def _download_kernel_outputs_with_retry(
    *,
    kaggle_bin: Path,
    kernel: str,
    output_dir: Path,
    token: str,
    attempts: int = 5,
    base_delay: int = 5,
) -> None:
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    command = [
        str(kaggle_bin),
        "kernels",
        "output",
        kernel,
        "-p",
        str(output_dir),
        "-q",
        "--file-pattern",
        r"^oem_outputs/",
    ]
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run(command, env=env, check=True, text=True)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"{kernel}: output download attempt {attempt}/{attempts} failed; "
                f"retrying in {delay}s without forcing already-complete files",
                flush=True,
            )
            time.sleep(delay)


def _load_or_create_wandb_target_id(model_root: Path) -> str:
    state_path = model_root / "fleet-state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        target_id = str(state.get("wandb_target_id") or "")
        if target_id:
            return target_id
    target_id = uuid.uuid4().hex[:8]
    state_path.write_text(json.dumps({"wandb_target_id": target_id}, indent=2) + "\n")
    return target_id


def _recover_existing_part(
    *,
    part_root: Path,
    kaggle_bin: Path,
    wandb_bin: Path,
    wandb_target_id: str,
    wandb_append: bool,
) -> dict[str, object] | None:
    state_path = part_root / "state.json"
    if not state_path.is_file():
        return None

    state = json.loads(state_path.read_text())
    status = str(state.get("status") or "")
    if status not in {"COMPLETE", "DOWNLOADED", "SYNCED"}:
        return None

    kernel = str(state.get("kernel") or "")
    if "/" not in kernel:
        raise RuntimeError(f"invalid kernel in existing part state: {kernel!r}")
    output_dir = Path(str(state.get("output_dir") or (part_root / "output"))).resolve()
    owner = kernel.split("/", 1)[0]

    if status == "COMPLETE":
        token = _token_for_owner(owner)
        print(f"{kernel}: recovering completed Kaggle output download", flush=True)
        _download_kernel_outputs_with_retry(
            kaggle_bin=kaggle_bin,
            kernel=kernel,
            output_dir=output_dir,
            token=token,
        )
        kp._state_update(state_path, state, status="DOWNLOADED")
        status = "DOWNLOADED"

    synced = list(state.get("synced_wandb_runs") or [])
    if status == "DOWNLOADED":
        print(f"{kernel}: syncing recovered offline W&B run(s)", flush=True)
        synced = kp._sync_wandb(
            wandb_bin,
            output_dir,
            target_id=wandb_target_id,
            append=wandb_append,
        )
        kp._state_update(
            state_path,
            state,
            status="SYNCED",
            synced_wandb_runs=synced,
        )

    return {
        "kernel": kernel,
        "output_dir": output_dir,
        "state_path": state_path,
        "synced_wandb_runs": synced,
    }


def run_model(model: str, fleet_root: Path, repo_ref: str, poll_seconds: int) -> int:
    client_dir = Path("/home/duypham/.local/share/oem-kaggle-client")
    kaggle_bin = kp._tool(client_dir, "kaggle")
    wandb_bin = kp._tool(client_dir, "wandb")
    model_root = fleet_root / model
    model_root.mkdir(parents=True, exist_ok=True)
    lock_root = fleet_root / "locks"
    wandb_target_id = _load_or_create_wandb_target_id(model_root)
    previous_kernel = None
    previous_owner = None
    previous_output = None
    base_slug = f"oem-{model}-fleet-{fleet_root.name}"

    args = kp.build_parser().parse_args(["--model", model, "--repo-ref", repo_ref])
    args.repo_ref = repo_ref
    args.client_dir = client_dir
    args.poll_seconds = poll_seconds
    args.smoke = False
    args.model_variant = None

    for part, end_epoch in enumerate(phase_plan(model), start=1):
        part_root = model_root / f"part-{part:02d}"
        recovered = _recover_existing_part(
            part_root=part_root,
            kaggle_bin=kaggle_bin,
            wandb_bin=wandb_bin,
            wandb_target_id=wandb_target_id,
            wandb_append=part > 1,
        )
        if recovered is not None:
            previous_kernel = str(recovered["kernel"])
            previous_owner = previous_kernel.split("/", 1)[0]
            previous_output = Path(recovered["output_dir"])
            print(f"{model}: recovered existing part {part}: {previous_kernel}", flush=True)
            if end_epoch is None:
                break
            markers = list(previous_output.rglob("chunk_state.json"))
            if len(markers) != 1:
                raise RuntimeError(f"{model}: expected one chunk_state.json, got {len(markers)}")
            state = json.loads(markers[0].read_text())
            if bool(state.get("complete")):
                print(f"{model}: completed at epoch {state.get('epoch')}", flush=True)
                break
            continue

        account, lock, owner, token, remaining = choose_account(
            preferred=PREFERRED_ACCOUNT[model],
            previous_owner=previous_owner,
            lock_root=lock_root,
            kaggle_bin=kaggle_bin,
        )
        print(
            f"{model}: part={part} account={account} owner={owner} gpu_remaining={remaining:.2f}h",
            flush=True,
        )
        relay_dataset = None
        same_owner = previous_owner is None or owner == previous_owner
        if previous_owner is not None and not same_owner:
            relay_dataset = create_resume_dataset(
                model=model,
                part=part,
                owner=owner,
                token=token,
                previous_output=previous_output,
                relay_root=model_root / "relay",
                kaggle_bin=kaggle_bin,
            )
            print(f"{model}: relayed resume checkpoint to private dataset {relay_dataset}", flush=True)

        try:
            slug = f"{base_slug}-part{part}" if end_epoch is not None else base_slug
            try:
                result = kp._run_kernel_once(
                    args=args,
                    owner=owner,
                    token=token,
                    kaggle_bin=kaggle_bin,
                    wandb_bin=wandb_bin,
                    slug=slug,
                    run_root=part_root,
                    chunk_end_epoch=end_epoch,
                    previous_kernel=previous_kernel if same_owner else None,
                    resume_dataset=relay_dataset,
                    wandb_target_id=wandb_target_id if end_epoch is not None else None,
                    wandb_append=part > 1,
                )
            except subprocess.CalledProcessError:
                result = _recover_existing_part(
                    part_root=part_root,
                    kaggle_bin=kaggle_bin,
                    wandb_bin=wandb_bin,
                    wandb_target_id=wandb_target_id,
                    wandb_append=part > 1,
                )
                if result is None:
                    raise
        finally:
            lock.close()

        previous_kernel = str(result["kernel"])
        previous_owner = owner
        previous_output = Path(result["output_dir"])
        if end_epoch is None:
            break

        markers = list(previous_output.rglob("chunk_state.json"))
        if len(markers) != 1:
            raise RuntimeError(f"{model}: expected one chunk_state.json, got {len(markers)}")
        state = json.loads(markers[0].read_text())
        if bool(state.get("complete")):
            print(f"{model}: completed at epoch {state.get('epoch')}", flush=True)
            break

    print(f"{model}: DONE", flush=True)
    return 0


def launch_all(fleet_root: Path, repo_ref: str, poll_seconds: int) -> int:
    fleet_root.mkdir(parents=True, exist_ok=True)
    launches = {}
    for model in MODELS:
        log_path = fleet_root / f"{model}.log"
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--model",
                    model,
                    "--fleet-root",
                    str(fleet_root),
                    "--repo-ref",
                    repo_ref,
                    "--poll-seconds",
                    str(poll_seconds),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=Path(__file__).resolve().parents[1],
            )
        launches[model] = {"pid": process.pid, "log": str(log_path)}
    (fleet_root / "launches.json").write_text(json.dumps(launches, indent=2) + "\n")
    for model, item in launches.items():
        print(f"{model}: pid={item['pid']} log={item['log']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--repo-ref", default="main")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--fleet-root", type=Path)
    ns = parser.parse_args()
    if bool(ns.model) == bool(ns.all):
        parser.error("choose exactly one of --model or --all")
    repo_ref = kp.resolve_repo_ref(Path(__file__).resolve().parents[1], ns.repo_ref)
    fleet_root = ns.fleet_root or Path("/home/duypham/.local/state/oem-kaggle") / (
        "fleet-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    if ns.all:
        return launch_all(fleet_root, repo_ref, ns.poll_seconds)
    return run_model(ns.model, fleet_root, repo_ref, ns.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
