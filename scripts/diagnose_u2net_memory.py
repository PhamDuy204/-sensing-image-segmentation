#!/usr/bin/env python3
"""Bounded U2-Net memory probe using the real trainer, loaders and Accelerate.

In a diagnostic checkout, replace train.py with:
    from scripts.diagnose_u2net_memory import main
    main()
This stops after 160 real batches without saving an incomplete epoch.
"""
from __future__ import annotations

import gc
import json
import platform
import weakref
from pathlib import Path

import torch

from oemseg.config import parse_args
from oemseg.engine import trainer

STEPS = {1, 2, 5, 10, 32, 64, 96, 110, 120, 124, 125, 126, 127, 128, 129, 130, 135, 140, 150, 160}


class ProbeComplete(Exception):
    pass


def main():
    args = parse_args()
    if args.model != "u2net" or args.smoke:
        raise ValueError("probe requires U2-Net with the full training configuration")
    original_epoch = trainer.train_one_epoch

    def measured_epoch(**kwargs):
        accelerator = kwargs["accelerator"]
        model = kwargs["model"]
        device = accelerator.device
        rank = accelerator.process_index
        root = args.output_root / args.run_name
        root.mkdir(parents=True, exist_ok=True)
        refs = []
        step = 0
        last_op = {}
        path = root / f"memory-rank{rank}.jsonl"
        log = path.open("w", buffering=1)

        def emit(phase, **extra):
            free, total = torch.cuda.mem_get_info(device)
            row = {
                "rank": rank, "step": step, "phase": phase,
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
                "peak": torch.cuda.max_memory_allocated(device),
                "free": free, "total": total,
                "live_output_tensors": sum(ref() is not None for ref in refs),
                "gc_count": gc.get_count(), **extra,
            }
            line = json.dumps(row)
            log.write(line + "\n")
            print("U2_MEMORY " + line, flush=True)

        def before_forward(module, inputs):
            nonlocal step
            step += 1
            if step in STEPS:
                torch.cuda.reset_peak_memory_stats(device)
                emit("before_forward", shape=list(inputs[0].shape), dtype=str(inputs[0].dtype))

        def outputs_created(module, inputs, outputs):
            refs.extend(weakref.ref(output) for output in outputs)

        backward = accelerator.backward

        def measured_backward(loss, **backward_kwargs):
            handles = []
            if step in STEPS:
                emit("before_backward", loss=float(loss.detach()))
                # Names/shapes only: these hooks never retain gradient tensors.
                seen = set()
                pending = [loss.grad_fn]
                while pending:
                    node = pending.pop()
                    if node is None or node in seen:
                        continue
                    seen.add(node)
                    name = node.name()
                    def entered(grads, name=name):
                        last_op["name"] = name
                        last_op["gradient_shapes"] = [list(g.shape) for g in grads if g is not None]
                    handles.append(node.register_prehook(entered))
                    pending.extend(n for n, _ in node.next_functions)
                del seen, pending, node
            try:
                backward(loss, **backward_kwargs)
            except torch.OutOfMemoryError:
                emit("oom", last_backward_op=last_op)
                (root / f"oom-rank{rank}.txt").write_text(torch.cuda.memory_summary(device))
                raise
            finally:
                for handle in handles:
                    handle.remove()
            if step in STEPS:
                emit("after_backward")

        hooks = [
            model.register_forward_pre_hook(before_forward),
            accelerator.unwrap_model(model).model.register_forward_hook(outputs_created),
        ]
        accelerator.backward = measured_backward
        emit("setup", python=platform.python_version(), torch=torch.__version__,
             cudnn=torch.backends.cudnn.version(), benchmark=torch.backends.cudnn.benchmark,
             mixed_precision=accelerator.mixed_precision, world_size=accelerator.num_processes,
             gpu=torch.cuda.get_device_name(device),
             ddp=model._get_ddp_logging_data() if hasattr(model, "_get_ddp_logging_data") else {})
        try:
            kwargs["max_batches"] = 160
            loss = original_epoch(**kwargs)
            if step != 160:
                raise RuntimeError(f"probe expected 160 batches, completed {step}")
            emit("complete", train_loss=loss)
            print(f"U2_DIAGNOSTIC_PASS rank={rank} batches={step}", flush=True)
        finally:
            accelerator.backward = backward
            for hook in hooks:
                hook.remove()
            log.close()
        raise ProbeComplete

    trainer.train_one_epoch = measured_epoch
    try:
        trainer.run_training(args)
    except ProbeComplete:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    finally:
        trainer.train_one_epoch = original_epoch
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
