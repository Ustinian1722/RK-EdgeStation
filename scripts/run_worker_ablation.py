#!/usr/bin/env python3
"""Run a matched RK3576 Worker backend/model matrix and preserve raw metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_platform_metadata() -> dict[str, Any]:
    try:
        gst_output = subprocess.run(
            ["gst-launch-1.0", "--version"], capture_output=True, text=True, check=True
        ).stdout.strip().splitlines()
        gst_version = next(
            (line.strip() for line in gst_output if "version" in line.lower()),
            gst_output[0] if gst_output else None,
        )
    except (OSError, subprocess.CalledProcessError, IndexError):
        gst_version = None
    versions: dict[str, str | None] = {}
    for name, package in (("mpp_library", "rockchip_mpp"), ("rga_api", "librga")):
        try:
            versions[name] = subprocess.run(
                ["pkg-config", "--modversion", package],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            versions[name] = None
    try:
        mpp_output = subprocess.run(
            ["gst-inspect-1.0", "mppvideodec"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
        versions["mpp_gstreamer_plugin"] = next(
            (line.split("Version", 1)[-1].strip() for line in mpp_output
             if line.lstrip().startswith("Version")),
            None,
        )
    except (OSError, subprocess.CalledProcessError):
        versions["mpp_gstreamer_plugin"] = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gstreamer_version": gst_version,
        "accelerator_software_versions": versions,
    }


def metric_lines(stdout: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("type") == "metric":
            metrics.append(item)
        elif item.get("type") == "event":
            events.append(item)
    return metrics, events


def run_worker(
    config: dict[str, str],
    repetition: int,
    args: argparse.Namespace,
    video_sha256: str,
    model_hashes: dict[str, str],
    worker_hash: str,
) -> dict[str, Any]:
    run_id = f"{args.experiment}-{config['id']}-r{repetition}"
    state_dir = args.output_dir / "state" / run_id
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    model = args.model_int8 if config["model_key"] == "int8" else args.model_fp

    command = [
        str(args.worker),
        "--model", str(model),
        "--model-family", args.model_family,
        "--video", str(args.video),
        "--state-dir", str(state_dir),
        "--video-backend", config["video_backend"],
        "--preprocess-backend", config["preprocess_backend"],
        "--input-pacing", args.pacing,
        "--queue-capacity", str(args.queue_capacity),
        "--queue-policy", config.get("queue_policy", args.queue_policy),
        "--confidence", str(args.confidence),
    ]

    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    completed = subprocess.run(command, cwd=args.cwd, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = (
        children_after.ru_utime - children_before.ru_utime
        + children_after.ru_stime - children_before.ru_stime
    )
    metrics, events = metric_lines(completed.stdout)
    (raw_dir / f"{run_id}.stdout.ndjson").write_text(
        completed.stdout, encoding="utf-8"
    )
    (raw_dir / f"{run_id}.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    final_metric = next(
        (item for item in reversed(metrics) if item.get("status") in {"completed", "error"}),
        metrics[-1] if metrics else {},
    )
    record: dict[str, Any] = {
        "run_id": run_id,
        "experiment": args.experiment,
        "config": config,
        "repetition": repetition,
        "return_code": completed.returncode,
        "wall_seconds_including_process_startup": elapsed,
        "child_cpu_seconds": cpu_seconds,
        "child_cpu_core_percent": cpu_seconds / elapsed * 100.0 if elapsed else None,
        "child_cpu_system_percent": (
            cpu_seconds / elapsed / os.cpu_count() * 100.0
            if elapsed and os.cpu_count() else None
        ),
        "model_path": str(model),
        "model_family": args.model_family,
        "model_sha256": model_hashes[config["model_key"]],
        "model_bytes": model.stat().st_size,
        "video_path": str(args.video),
        "video_sha256": video_sha256,
        "video_bytes": args.video.stat().st_size,
        "worker_path": str(args.worker),
        "worker_sha256": worker_hash,
        "queue_capacity": args.queue_capacity,
        "queue_policy": config.get("queue_policy", args.queue_policy),
        "input_pacing": args.pacing,
        "confidence": args.confidence,
        "worker_metric_samples": metrics,
        "final_metric": final_metric,
        "event_records": len(events),
    }
    record_path = raw_dir / f"{run_id}.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_id": run_id,
        "return_code": completed.returncode,
        "decoded_frames": final_metric.get("decoded_frames"),
        "processed_frames": final_metric.get("processed_frames"),
        "dropped_frames": final_metric.get("dropped_frames"),
        "throughput_fps": final_metric.get("throughput_fps"),
        "p95_end_to_end_ms": final_metric.get("p95_end_to_end_ms"),
        "rss_peak_kb": final_metric.get("process_peak_rss_kb"),
    }), flush=True)
    return record


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["config"]["id"], []).append(record)
    for config_id, items in groups.items():
        final_metrics = [item.get("final_metric", {}) for item in items]
        fps_values = [
            float(item["final_metric"]["throughput_fps"])
            for item in items if item.get("final_metric", {}).get("throughput_fps") is not None
        ]
        stage_fields = (
            "p50_decode_ms", "p95_decode_ms", "p50_preprocess_ms", "p95_preprocess_ms",
            "p50_rknn_input_set_ms", "p95_rknn_input_set_ms", "p50_rknn_run_ms",
            "p95_rknn_run_ms", "p50_rknn_output_fetch_ms", "p95_rknn_output_fetch_ms",
            "p50_inference_ms", "p95_inference_ms", "p50_postprocess_ms",
            "p95_postprocess_ms", "p50_queue_wait_ms", "p95_queue_wait_ms",
            "p50_end_to_end_ms", "p95_end_to_end_ms", "process_peak_rss_kb",
            "process_cpu_core_percent", "process_cpu_system_percent", "nominal_input_fps",
        )
        stage_medians = {}
        for field in stage_fields:
            values = [
                float(metric[field]) for metric in final_metrics
                if metric.get(field) is not None
            ]
            stage_medians[field] = statistics.median(values) if values else None
        summaries.append({
            "config_id": config_id,
            "successful_runs": sum(item["return_code"] == 0 for item in items),
            "runs": len(items),
            "median_throughput_fps": statistics.median(fps_values) if fps_values else None,
            "throughput_fps_each": fps_values,
            "processed_frames_each": [
                item.get("final_metric", {}).get("processed_frames") for item in items
            ],
            "decoded_frames_each": [
                item.get("final_metric", {}).get("decoded_frames") for item in items
            ],
            "dropped_frames_each": [
                item.get("final_metric", {}).get("dropped_frames") for item in items
            ],
            "drop_rate_percent_each": [
                (
                    100.0 * item["final_metric"].get("dropped_frames", 0) /
                    item["final_metric"].get("decoded_frames", 1)
                )
                for item in items if item.get("final_metric", {}).get("decoded_frames", 0) > 0
            ],
            "p95_end_to_end_ms_each": [
                item.get("final_metric", {}).get("p95_end_to_end_ms") for item in items
            ],
            "p95_rknn_run_ms_each": [
                item.get("final_metric", {}).get("p95_rknn_run_ms") for item in items
            ],
            "p95_preprocess_ms_each": [
                item.get("final_metric", {}).get("p95_preprocess_ms") for item in items
            ],
            "median_stage_metrics": stage_medians,
        })
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model-fp", required=True, type=Path)
    parser.add_argument("--model-int8", required=True, type=Path)
    parser.add_argument("--model-family", choices=("yolov8-coco", "yolox-ppe"),
                        default="yolov8-coco")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment", choices=("offline", "media-clock", "realtime-queue"), required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--expected-frames", type=int, default=150)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--queue-capacity", type=int)
    parser.add_argument("--queue-policy", choices=("latest", "fifo"))
    parser.add_argument("--pacing", choices=("media", "none"))
    args = parser.parse_args()

    args.worker = args.worker.resolve()
    args.cwd = args.cwd.resolve()
    args.video = args.video.resolve()
    args.model_fp = args.model_fp.resolve()
    args.model_int8 = args.model_int8.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.repetitions < 1 or args.expected_frames < 1:
        parser.error("repetitions and expected-frames must be positive")
    if not args.worker.is_file() or not args.video.is_file():
        parser.error("worker and video paths must exist")
    if not args.model_fp.is_file() or not args.model_int8.is_file():
        parser.error("both model paths must exist")

    if args.experiment == "offline":
        args.queue_capacity = args.queue_capacity or args.expected_frames + 1
        args.queue_policy = args.queue_policy or "fifo"
        args.pacing = args.pacing or "none"
        configs = [
            {"id": "A0", "video_backend": "opencv", "preprocess_backend": "cpu", "model_key": "fp"},
            {"id": "A1", "video_backend": "gstreamer-mpp", "preprocess_backend": "cpu", "model_key": "fp"},
            {"id": "A2", "video_backend": "gstreamer-mpp", "preprocess_backend": "rga", "model_key": "fp"},
            {"id": "A3", "video_backend": "gstreamer-mpp", "preprocess_backend": "rga", "model_key": "int8"},
        ]
    elif args.experiment == "media-clock":
        args.queue_capacity = args.queue_capacity or 1
        args.queue_policy = args.queue_policy or "latest"
        args.pacing = args.pacing or "media"
        configs = [
            {"id": "R0", "video_backend": "opencv", "preprocess_backend": "cpu", "model_key": "fp"},
            {"id": "R2", "video_backend": "gstreamer-mpp", "preprocess_backend": "rga", "model_key": "fp"},
            {"id": "R3", "video_backend": "gstreamer-mpp", "preprocess_backend": "rga", "model_key": "int8"},
        ]
    else:
        args.queue_capacity = args.queue_capacity or 1
        args.pacing = args.pacing or "media"
        if args.queue_policy:
            configs = [
                {"id": f"Q-{args.queue_policy}", "video_backend": "gstreamer-mpp",
                 "preprocess_backend": "rga", "model_key": "fp"}
            ]
        else:
            configs = [
                {"id": "Q-latest", "video_backend": "gstreamer-mpp",
                 "preprocess_backend": "rga", "model_key": "fp", "queue_policy": "latest"},
                {"id": "Q-fifo", "video_backend": "gstreamer-mpp",
                 "preprocess_backend": "rga", "model_key": "fp", "queue_policy": "fifo"},
            ]
            args.queue_policy = "latest"

    model_hashes = {
        "fp": sha256_file(args.model_fp),
        "int8": sha256_file(args.model_int8),
    }
    video_hash = sha256_file(args.video)
    worker_hash = sha256_file(args.worker)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_platform_metadata()
    metadata.update({
        "experiment": args.experiment,
        "video_path": str(args.video),
        "video_sha256": video_hash,
        "video_bytes": args.video.stat().st_size,
        "models": {
            "fp": {"path": str(args.model_fp), "sha256": model_hashes["fp"],
                   "bytes": args.model_fp.stat().st_size},
            "int8": {"path": str(args.model_int8), "sha256": model_hashes["int8"],
                     "bytes": args.model_int8.stat().st_size},
        },
        "model_family": args.model_family,
        "expected_frames": args.expected_frames,
        "repetitions": args.repetitions,
        "queue_capacity": args.queue_capacity,
        "queue_policy": args.queue_policy,
        "input_pacing": args.pacing,
        "worker": {"path": str(args.worker), "sha256": worker_hash,
                    "bytes": args.worker.stat().st_size},
        "configs": configs,
        "run_order": [
            {"repetition": repetition, "config_id": config["id"]}
            for repetition in range(1, args.repetitions + 1) for config in configs
        ],
    })
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    records = [
        run_worker(config, repetition, args, video_hash, model_hashes, worker_hash)
        for repetition in range(1, args.repetitions + 1) for config in configs
    ]
    summary = summarize(records)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.experiment == "offline":
        failures = [
            record for record in records
            if record["return_code"] != 0
            or record.get("final_metric", {}).get("status") != "completed"
            or record.get("final_metric", {}).get("decoded_frames") != args.expected_frames
            or record.get("final_metric", {}).get("processed_frames") != args.expected_frames
        ]
    else:
        failures = [
            record for record in records
            if record["return_code"] != 0
            or record.get("final_metric", {}).get("status") != "completed"
            or record.get("final_metric", {}).get("decoded_frames") != args.expected_frames
        ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
