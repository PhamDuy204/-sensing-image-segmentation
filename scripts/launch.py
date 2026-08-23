#!/usr/bin/env python3
"""Launch one distributed model or one independent model per GPU."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"


def discover_gpu_ids() -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("NVIDIA GPUs are unavailable (nvidia-smi failed)") from exc
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def parse_gpu_ids(spec: str, available: list[int]) -> list[int]:
    gpu_ids = (
        list(available)
        if spec.strip().lower() == "all"
        else [int(x) for x in spec.split(",") if x.strip()]
    )
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"duplicate GPU IDs: {gpu_ids}")
    missing = [gpu for gpu in gpu_ids if gpu not in available]
    if missing:
        raise ValueError(f"requested GPUs {missing} are unavailable; available GPUs: {available}")
    return gpu_ids


def validate_parallel(gpu_ids: list[int], models: list[str]) -> None:
    if len(gpu_ids) != len(models):
        raise ValueError(
            f"parallel mode requires one model per GPU; GPUs={len(gpu_ids)} models={len(models)}"
        )


def build_distributed_command(gpu_ids: list[int], model: str, train_args: list[str]) -> list[str]:
    if len(gpu_ids) == 1:
        return [sys.executable, str(TRAIN), "--model", model, *train_args]
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes",
        str(len(gpu_ids)),
        str(TRAIN),
        "--model",
        model,
        *train_args,
    ]


def build_parallel_commands(gpu_ids: list[int], models: list[str], train_args: list[str]):
    validate_parallel(gpu_ids, models)
    return [
        (gpu, [sys.executable, str(TRAIN), "--model", model, *train_args])
        for gpu, model in zip(gpu_ids, models, strict=True)
    ]


def _project_models(models: list[str]) -> list[str]:
    sys.path.insert(0, str(ROOT))
    from oemseg.models.registry import available_models, normalize_name

    valid = set(available_models())
    normalized = [normalize_name(model) for model in models]
    invalid = [model for model in normalized if model not in valid]
    if invalid:
        raise ValueError(f"unknown model(s) {invalid}; available models: {sorted(valid)}")
    return normalized


def _forwarded_args(args: list[str]) -> list[str]:
    return args[1:] if args[:1] == ["--"] else args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    distributed = subparsers.add_parser("distributed", help="N GPUs train one model")
    distributed.add_argument("--gpus", default="all", help="comma-separated physical GPU IDs or 'all'")
    distributed.add_argument("--model", required=True)
    distributed.add_argument("--dry-run", action="store_true")
    distributed.add_argument("train_args", nargs=argparse.REMAINDER)

    parallel = subparsers.add_parser("parallel", help="N GPUs train N independent models")
    parallel.add_argument("--gpus", default="all", help="comma-separated physical GPU IDs or 'all'")
    parallel.add_argument("--models", required=True, help="comma-separated models in GPU order")
    parallel.add_argument("--dry-run", action="store_true")
    parallel.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    available = discover_gpu_ids()
    gpu_ids = parse_gpu_ids(args.gpus, available)
    train_args = _forwarded_args(args.train_args)

    if args.mode == "distributed":
        model = _project_models([args.model])[0]
        command = build_distributed_command(gpu_ids, model, train_args)
        print(f"GPUs: {gpu_ids}\n{shlex.join(command)}")
        if args.dry_run:
            return 0
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        return subprocess.run(command, cwd=ROOT, env=env).returncode

    models = _project_models([m.strip() for m in args.models.split(",") if m.strip()])
    validate_parallel(gpu_ids, models)
    commands = build_parallel_commands(gpu_ids, models, train_args)
    for gpu, command in commands:
        print(f"GPU {gpu}: {shlex.join(command)}")
    if args.dry_run:
        return 0

    processes = []
    for gpu, command in commands:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        processes.append(subprocess.Popen(command, cwd=ROOT, env=env))
    exit_code = 0
    for process in processes:
        code = process.wait()
        if code and not exit_code:
            exit_code = code
            for other in processes:
                if other.poll() is None:
                    other.terminate()
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
