#!/usr/bin/env python3
"""Interleave same-video RK3576 worker runs and capture latency, drops, CPU, and RSS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_sample(pid: int) -> tuple[int, int] | None:
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close = raw_stat.rfind(")")
        fields = raw_stat[close + 2:].split()
        ticks = int(fields[11]) + int(fields[12])
        status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
        rss_kib = next(int(line.split()[1]) for line in status.splitlines()
                       if line.startswith("VmRSS:"))
        return ticks, rss_kib
    except (OSError, ValueError, IndexError, StopIteration):
        return None


class ProcessMonitor:
    def __init__(self, process: subprocess.Popen[str], sample_seconds: float):
        self.process = process
        self.sample_seconds = sample_seconds
        self.stop_requested = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.cpu_percentages: list[float] = []
        self.rss_kib: list[int] = []
        self.clock_ticks = int(os.sysconf("SC_CLK_TCK"))

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_requested.set()
        self.thread.join(timeout=max(2.0, self.sample_seconds * 3))

    def _run(self) -> None:
        previous = None
        previous_time = None
        while not self.stop_requested.is_set() or self.process.poll() is None:
            sampled_at = time.monotonic()
            current = process_sample(self.process.pid)
            if current is not None:
                self.rss_kib.append(current[1])
                if previous is not None and previous_time is not None:
                    elapsed = sampled_at - previous_time
                    if elapsed > 0.0:
                        self.cpu_percentages.append(
                            max(0.0, 100.0 * (current[0] - previous[0]) /
                                self.clock_ticks / elapsed)
                        )
                previous = current
                previous_time = sampled_at
            if self.process.poll() is not None and self.stop_requested.is_set():
                break
            time.sleep(self.sample_seconds)

    def summary(self) -> dict[str, Any]:
        return {
            "cpu_percent_one_core_avg": round(statistics.mean(self.cpu_percentages), 2)
            if self.cpu_percentages else None,
            "cpu_percent_one_core_max": round(max(self.cpu_percentages), 2)
            if self.cpu_percentages else None,
            "rss_kib_avg": round(statistics.mean(self.rss_kib), 1) if self.rss_kib else None,
            "rss_kib_max": max(self.rss_kib) if self.rss_kib else None,
            "resource_samples": len(self.rss_kib),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--reference-model", required=True, type=Path)
    parser.add_argument("--reference-family", choices=("yolov8-coco", "yolox-ppe"), required=True)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--candidate-family", choices=("yolov8-coco", "yolox-ppe"), required=True)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sample-seconds", type=float, default=0.25)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--roi", default="0,0,1,1")
    parser.add_argument("--queue-capacity", type=int, default=1)
    parser.add_argument("--queue-policy", choices=("latest", "fifo"), default="latest")
    parser.add_argument("--dwell-ms", type=int, default=1200)
    parser.add_argument("--roi-debounce-frames", type=int, default=2)
    parser.add_argument("--danger-zone")
    parser.add_argument("--conflict-zone")
    parser.add_argument("--ppe-confirm-frames", type=int, default=3)
    parser.add_argument("--ppe-cooldown-ms", type=int, default=30000)
    parser.add_argument("--conflict-dwell-ms", type=int, default=1000)
    args = parser.parse_args()
    if args.runs < 1 or args.sample_seconds <= 0.0 or args.queue_capacity < 1:
        parser.error("runs, sample interval, and queue capacity must be positive")
    for option in ("worker", "reference_model", "candidate_model", "video"):
        path = getattr(args, option).resolve()
        if not path.is_file():
            parser.error(f"{option.replace('_', ' ')} does not exist: {path}")
        setattr(args, option, path)
    args.output_dir = args.output_dir.resolve()
    return args


def run_one(args: argparse.Namespace, label: str, family: str, model: Path,
            repetition: int) -> dict[str, Any]:
    run_id = f"{label}-{repetition:02d}"
    run_dir = args.output_dir / run_id
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "worker.jsonl"
    command = [
        str(args.worker), "--model", str(model), "--video", str(args.video),
        "--model-family", family, "--state-dir", str(state_dir),
        "--confidence", str(args.confidence), "--roi", args.roi,
        "--queue-capacity", str(args.queue_capacity), "--queue-policy", args.queue_policy,
        "--dwell-ms", str(args.dwell_ms), "--roi-debounce-frames", str(args.roi_debounce_frames),
        "--ppe-confirm-frames", str(args.ppe_confirm_frames),
        "--ppe-cooldown-ms", str(args.ppe_cooldown_ms),
        "--conflict-dwell-ms", str(args.conflict_dwell_ms),
    ]
    if args.danger_zone:
        command.extend(("--danger-zone", args.danger_zone))
    if args.conflict_zone:
        command.extend(("--conflict-zone", args.conflict_zone))

    started_utc = utc_now()
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        monitor = ProcessMonitor(process, args.sample_seconds)
        monitor.start()
        return_code = process.wait()
        monitor.stop()
    elapsed = time.monotonic() - started
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    metrics = [row for row in records if row.get("type") == "metric"]
    if not metrics:
        raise RuntimeError(f"{run_id} produced no worker metrics; inspect {log_path}")
    final = metrics[-1]
    event_counts: dict[str, int] = {}
    for row in records:
        if row.get("type") == "event":
            event_type = str(row.get("event_type", "unknown"))
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
    if return_code != 0 or final.get("status") != "completed":
        raise RuntimeError(f"{run_id} failed with code {return_code}, status={final.get('status')}")
    return {
        "run_id": run_id,
        "family": family,
        "model": model.name,
        "model_sha256": sha256_file(model),
        "started_utc": started_utc,
        "elapsed_seconds": round(elapsed, 3),
        "return_code": return_code,
        "worker_metrics": final,
        "process_resource_samples": monitor.summary(),
        "event_counts": event_counts,
        "worker_log": str(log_path),
        "command": command,
    }


def median_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row["worker_metrics"].get(field) for row in rows]
    values = [float(value) for value in values if isinstance(value, (int, float))]
    return round(statistics.median(values), 3) if values else None


def aggregate(label: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "processed_frames", "decoded_frames", "detections", "dropped_frames", "throughput_fps",
        "last_preprocess_ms", "last_inference_ms", "last_postprocess_ms", "last_queue_wait_ms",
        "last_end_to_end_ms", "p50_end_to_end_ms", "p95_end_to_end_ms", "process_rss_kb",
    )
    result = {field: median_field(runs, field) for field in fields}
    for field in ("cpu_percent_one_core_avg", "cpu_percent_one_core_max", "rss_kib_avg", "rss_kib_max"):
        values = [row["process_resource_samples"].get(field) for row in runs]
        values = [float(value) for value in values if isinstance(value, (int, float))]
        result[field] = round(statistics.median(values), 2) if values else None
    result["run_count"] = len(runs)
    result["label"] = label
    return result


def main() -> int:
    args = parse_args()
    if args.output_dir == args.worker or args.output_dir in args.worker.parents:
        raise ValueError("output directory must not contain or replace the worker directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {args.output_dir}")

    reference_runs: list[dict[str, Any]] = []
    candidate_runs: list[dict[str, Any]] = []
    for repetition in range(1, args.runs + 1):
        pair = (
            ("reference", args.reference_family, args.reference_model),
            ("candidate", args.candidate_family, args.candidate_model),
        )
        if repetition % 2 == 0:
            pair = tuple(reversed(pair))
        for label, family, model in pair:
            run = run_one(args, label, family, model, repetition)
            if label == "reference":
                reference_runs.append(run)
            else:
                candidate_runs.append(run)

    uname = os.uname()
    report = {
        "started_utc": reference_runs[0]["started_utc"],
        "finished_utc": utc_now(),
        "board_platform": {
            "sysname": uname.sysname,
            "nodename": uname.nodename,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "worker_sha256": sha256_file(args.worker),
        "video": args.video.name,
        "video_sha256": sha256_file(args.video),
        "same_input": True,
        "confidence": args.confidence,
        "roi": args.roi,
        "queue_capacity": args.queue_capacity,
        "queue_policy": args.queue_policy,
        "reference": {
            "model": args.reference_model.name,
            "model_sha256": sha256_file(args.reference_model),
            "family": args.reference_family,
            "runs": reference_runs,
            "median": aggregate("reference", reference_runs),
        },
        "candidate": {
            "model": args.candidate_model.name,
            "model_sha256": sha256_file(args.candidate_model),
            "family": args.candidate_family,
            "runs": candidate_runs,
            "median": aggregate("candidate", candidate_runs),
        },
        "limits": (
            "Same source file and worker settings; frame decisions can differ because queue=latest "
            "drops depend on timing. PPE and COCO detectors have different class sets, so detection "
            "counts are workload observations, not accuracy or equivalence metrics. End-to-end latency "
            "is decode-to-result age; process CPU is sampled from /proc on the board."
        ),
    }
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(output_path),
        "reference_median": report["reference"]["median"],
        "candidate_median": report["candidate"]["median"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
