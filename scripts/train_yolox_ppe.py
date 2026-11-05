#!/usr/bin/env python3
"""Fine-tune the project YOLOX-PPE experiment without the legacy trainer wrapper."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import cv2

cv2.setNumThreads(0)

import numpy as np
import torch
import psutil

_LAST_COCO_EVAL = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_weights(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"expected a YOLOX checkpoint containing model weights: {path}")
    return checkpoint["model"]


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_ap: float) -> None:
    payload = {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_ap": best_ap,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def metric_value(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    return float(value)


def windows_memory_stats() -> dict[str, float]:
    if sys.platform != "win32":
        return {}

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {}
    mib = 1024 * 1024
    return {
        "commit_available_mib": status.ullAvailPageFile / mib,
        "commit_limit_mib": status.ullTotalPageFile / mib,
    }


def per_class_ap(coco_eval, coco_dataset) -> dict[str, dict[str, float | None]]:
    """Return per-class AP50 and AP50:95 from the standard COCO precision tensor."""
    precision = coco_eval.eval["precision"]
    category_ids = sorted(coco_dataset.cats)
    if precision.shape[2] != len(category_ids):
        raise ValueError(
            f"COCO precision has {precision.shape[2]} classes, expected {len(category_ids)}"
        )

    result = {}
    for index, category_id in enumerate(category_ids):
        class_precision = precision[:, :, index, 0, -1]
        valid_all = class_precision[class_precision > -1]
        valid_50 = class_precision[0][class_precision[0] > -1]
        result[coco_dataset.cats[category_id]["name"]] = {
            "ap50_95": float(valid_all.mean() * 100) if valid_all.size else None,
            "ap50": float(valid_50.mean() * 100) if valid_50.size else None,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", type=Path, default=Path("training/exp/yolox_s_construction_ppe.py"))
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--segment-epochs", type=int, default=None,
                        help="stop cleanly after this many epochs while preserving the full schedule")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training route")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.segment_epochs is not None and args.segment_epochs < 1:
        raise ValueError("segment epochs must be positive")

    from yolox.exp import get_exp
    from yolox.utils import load_ckpt
    from pycocotools.cocoeval import COCOeval as PyCOCOeval
    import yolox.layers

    class CapturedCOCOeval(PyCOCOeval):
        def __init__(self, *values, **options):
            global _LAST_COCO_EVAL
            super().__init__(*values, **options)
            _LAST_COCO_EVAL = self

    # YOLOX's optional fast evaluator JIT-compiles a C++ extension and needs
    # MSVC on Windows. Standard pycocotools is already installed and portable.
    yolox.layers.COCOeval_opt = CapturedCOCOeval

    exp = get_exp(str(args.exp))
    if args.epochs is not None:
        exp.max_epoch = args.epochs
        exp.no_aug_epochs = min(exp.no_aug_epochs, max(1, exp.max_epoch // 5))
    random.seed(exp.seed)
    np.random.seed(exp.seed)
    torch.manual_seed(exp.seed)
    torch.cuda.manual_seed_all(exp.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    exp.output_dir = str(Path(exp.output_dir).resolve())
    output_dir = Path(exp.output_dir) / exp.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest_ckpt.pth"
    best_path = output_dir / "best_ckpt.pth"
    history_path = output_dir / "epoch_metrics.jsonl"
    if not args.resume and history_path.exists():
        history_path.unlink()

    model = exp.get_model().to(device).train()
    optimizer = exp.get_optimizer(args.batch_size)
    for group in optimizer.param_groups:
        # CUDA foreach SGD allocated a large temporary on this GPU; the
        # single-tensor update fits while training the neck plus head.
        group["foreach"] = False
    best_ap = 0.0
    start_epoch = 0
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"cannot resume without {latest_path}")
        resumed = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        start_epoch = int(resumed["epoch"]) + 1
        best_ap = float(resumed.get("best_ap", 0.0))

    stop_epoch = exp.max_epoch
    if args.segment_epochs is not None:
        stop_epoch = min(exp.max_epoch, start_epoch + args.segment_epochs)
    else:
        model = load_ckpt(model, load_checkpoint_weights(args.pretrained))

    train_loader = exp.get_data_loader(
        batch_size=args.batch_size,
        is_distributed=False,
        no_aug=False,
        cache_img=None,
    )
    steps_per_epoch = len(train_loader)
    scheduler = exp.get_lr_scheduler(
        exp.basic_lr_per_img * args.batch_size,
        steps_per_epoch,
    )
    evaluator = exp.get_evaluator(batch_size=1, is_distributed=False)
    pretrained_path = args.pretrained.resolve()
    print(json.dumps({
        "event": "training_started",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": exp.seed,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "train_images": len(train_loader.dataset),
        "steps_per_epoch": steps_per_epoch,
        "epochs": exp.max_epoch,
        "stop_after_epoch": stop_epoch,
        "batch_size": args.batch_size,
        "pretrained_sha256": file_sha256(pretrained_path),
        "output_dir": str(output_dir),
    }, ensure_ascii=False), flush=True)

    global_step = start_epoch * steps_per_epoch
    process = psutil.Process()
    for epoch in range(start_epoch, stop_epoch):
        data_iterator = iter(train_loader)
        if epoch == exp.max_epoch - exp.no_aug_epochs:
            train_loader.close_mosaic()
            model.head.use_l1 = True
        model.train()
        model.backbone.backbone.eval()
        epoch_started = time.perf_counter()
        sums: dict[str, float] = {}

        for step in range(steps_per_epoch):
            batch = next(data_iterator)
            images = batch[0].to(device, dtype=torch.float32, non_blocking=True)
            targets = batch[1].to(device, dtype=torch.float32, non_blocking=True)
            images, targets = exp.preprocess(images, targets, exp.input_size)
            targets.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=False):
                outputs = model(images, targets)
                loss = outputs["total_loss"]
            loss.backward()
            optimizer.step()

            learning_rate = scheduler.update_lr(global_step + 1)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            global_step += 1
            current = {key: metric_value(value) for key, value in outputs.items()}
            for key, value in current.items():
                sums[key] = sums.get(key, 0.0) + value
            if (step + 1) % 100 == 0 or step + 1 == steps_per_epoch:
                averages = {key: value / (step + 1) for key, value in sums.items()}
                memory = process.memory_info()
                full_memory = process.memory_full_info()
                print(json.dumps({
                    "event": "train_progress",
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "steps": steps_per_epoch,
                    "lr": learning_rate,
                    "rss_mib": memory.rss / (1024 * 1024),
                    "vms_mib": memory.vms / (1024 * 1024),
                    "uss_mib": getattr(full_memory, "uss", 0) / (1024 * 1024),
                    "available_ram_mib": psutil.virtual_memory().available / (1024 * 1024),
                    **windows_memory_stats(),
                    **averages,
                }), flush=True)
            # Drop references to the previous autograd graph and host batch
            # before asking the loader and NumPy preprocessing for more memory.
            del batch, images, targets, outputs, loss, current
            if (step + 1) % 100 == 0:
                gc.collect()

        train_metrics = {key: value / steps_per_epoch for key, value in sums.items()}
        record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "learning_rate": learning_rate,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        save_checkpoint(latest_path, model, optimizer, epoch, best_ap)

        if (epoch + 1) % exp.eval_interval == 0 or epoch + 1 == exp.max_epoch:
            model.eval()
            with torch.no_grad():
                (ap50_95, ap50, summary), _ = exp.eval(
                    model,
                    evaluator,
                    is_distributed=False,
                    half=False,
                    return_outputs=True,
                )
            record["validation"] = {
                "coco_ap50_95": float(ap50_95),
                "coco_ap50": float(ap50),
                "summary": str(summary),
                "per_class": per_class_ap(
                    _LAST_COCO_EVAL, evaluator.dataloader.dataset.coco
                ),
            }
            if float(ap50_95) > best_ap:
                best_ap = float(ap50_95)
                save_checkpoint(best_path, model, optimizer, epoch, best_ap)
            save_checkpoint(latest_path, model, optimizer, epoch, best_ap)

        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"event": "epoch_complete", **record, "best_ap50_95": best_ap}, ensure_ascii=False), flush=True)

    if stop_epoch < exp.max_epoch:
        print(json.dumps({
            "event": "training_segment_complete",
            "completed_epoch": stop_epoch,
            "scheduled_epochs": exp.max_epoch,
            "best_ap50_95": best_ap,
            "resume_with": "--resume",
        }), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
