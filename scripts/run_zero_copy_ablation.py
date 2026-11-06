#!/usr/bin/env python3
"""Run matched Z0/Z1/Z2 RKNN input-memory trials on a target board."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONFIGS = (
    ("z0", "copy", "api"),
    ("z1", "copy", "shared"),
    ("z2", "dmabuf", "shared"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def environment_snapshot() -> dict[str, Any]:
    try:
        gst = subprocess.run(["gst-launch-1.0", "--version"], check=True,
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gst = None
    try:
        rknn = subprocess.run(["ldconfig", "-p"], check=True, capture_output=True,
                              text=True, timeout=10).stdout
        rknn = next((line.strip() for line in rknn.splitlines()
                     if "librknnrt.so" in line), None)
    except (OSError, subprocess.SubprocessError):
        rknn = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gstreamer": gst,
        "rknn_runtime_library": rknn,
        "cpu_governor": read_text("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"),
        "cpu_frequency_khz": read_text("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq"),
        "thermal_zone0_millicelsius": read_text("/sys/class/thermal/thermal_zone0/temp"),
        "memory_meminfo": read_text("/proc/meminfo"),
    }


def metric_lines(stdout: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics, events = [], []
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


def json_object_lines(stdout: str) -> list[str]:
    """Return parseable JSONL records; keep library diagnostics in run.raw.log."""
    records = []
    for line in stdout.splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
    return records


def run_one(args: argparse.Namespace, mode: str, video_memory: str, input_mode: str,
            repetition: int, identity: dict[str, Any]) -> dict[str, Any]:
    run_id = f"{args.label}-{mode}-r{repetition}"
    run_dir = args.output_dir / f"{mode}-{repetition}"
    state_dir = run_dir / "state"
    run_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.worker), "--model", str(args.model), "--model-family", "yolox-ppe",
        "--video", str(args.video), "--state-dir", str(state_dir),
        "--video-backend", "gstreamer-mpp", "--preprocess-backend", "rga",
        "--video-memory-mode", video_memory, "--rknn-input-mode", input_mode,
        "--input-pacing", args.pacing, "--queue-capacity", str(args.queue_capacity),
        "--queue-policy", args.queue_policy, "--confidence", str(args.confidence),
        "--dwell-ms", str(args.dwell_ms), "--roi-debounce-frames", str(args.confirm_frames),
        "--ppe-confirm-frames", str(args.confirm_frames),
        "--ppe-cooldown-ms", str(args.ppe_cooldown_ms),
    ]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=args.cwd, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    metrics, events = metric_lines(completed.stdout)
    final = next((metric for metric in reversed(metrics)
                  if metric.get("status") in {"completed", "error"}), {})
    record = {
        **identity,
        "run_id": run_id,
        "mode": mode,
        "video_memory_mode": video_memory,
        "rknn_input_mode": input_mode,
        "queue_capacity": args.queue_capacity,
        "queue_policy": args.queue_policy,
        "input_pacing": args.pacing,
        "confidence": args.confidence,
        "return_code": completed.returncode,
        "wall_seconds_including_startup": elapsed,
        "final_metric": final,
        "metric_records": metrics,
        "event_records": events,
        "command": command,
    }
    (run_dir / "run.raw.log").write_text(completed.stdout, encoding="utf-8")
    jsonl = json_object_lines(completed.stdout)
    (run_dir / "run.jsonl").write_text("".join(line + "\n" for line in jsonl),
                                       encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    print(json.dumps({
        "run_id": run_id,
        "return_code": completed.returncode,
        "decoded_frames": final.get("decoded_frames"),
        "processed_frames": final.get("processed_frames"),
        "dropped_frames": final.get("dropped_frames"),
        "throughput_fps": final.get("throughput_fps"),
        "p95_end_to_end_ms": final.get("p95_end_to_end_ms"),
        "explicit_source_copy_bytes": final.get("source_copy_bytes_total"),
        "events": len(events),
    }), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="zero-copy")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--pacing", choices=("media", "none"), default="media")
    parser.add_argument("--queue-capacity", type=int, default=1)
    parser.add_argument("--queue-policy", choices=("latest", "fifo"), default="latest")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--dwell-ms", type=int, default=60000)
    parser.add_argument("--confirm-frames", type=int, default=1000)
    parser.add_argument("--ppe-cooldown-ms", type=int, default=30000)
    parser.add_argument("--cwd", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or args.queue_capacity < 1:
        parser.error("repeat and queue-capacity must be positive")
    for path in (args.worker, args.model, args.video):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "worker_path": str(args.worker.resolve()),
        "worker_sha256": sha256_file(args.worker),
        "model_path": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "model_bytes": args.model.stat().st_size,
        "video_path": str(args.video.resolve()),
        "video_sha256": sha256_file(args.video),
        "video_bytes": args.video.stat().st_size,
    }
    environment = environment_snapshot()
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "identity.json").write_text(
        json.dumps(identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    records = []
    for repetition in range(1, args.repeat + 1):
        for mode, video_memory, input_mode in CONFIGS:
            records.append(run_one(args, mode, video_memory, input_mode, repetition, identity))
    summary = {
        "environment": environment,
        "identity": identity,
        "repeat_count": args.repeat,
        "configuration_order": [
            {"mode": mode, "video_memory_mode": video_memory, "rknn_input_mode": input_mode}
            for mode, video_memory, input_mode in CONFIGS
        ],
        "runs": records,
    }
    (args.output_dir / "run_matrix.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for mode, _, _ in CONFIGS:
        selected = [row["final_metric"] for row in records if row["mode"] == mode]
        fps = [float(row["throughput_fps"]) for row in selected if row.get("throughput_fps") is not None]
        if fps:
            print(json.dumps({"mode": mode, "runs": len(fps),
                              "median_throughput_fps": statistics.median(fps)}), flush=True)
    if any(row["return_code"] != 0 for row in records):
        raise SystemExit("one or more Worker runs failed; see per-run raw logs")


if __name__ == "__main__":
    main()
