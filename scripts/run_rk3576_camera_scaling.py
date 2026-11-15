#!/usr/bin/env python3
"""Run aligned 1/2/4 independent Worker replays with RK3576 telemetry.

This deliberately measures process-level replay concurrency. It does not
implement or claim single-process multi-camera scheduling.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


NPU_DEVFREQ = Path("/sys/class/devfreq/27700000.npu")
CPUFREQ_ROOT = Path("/sys/devices/system/cpu/cpufreq")
THERMAL_ROOT = Path("/sys/class/thermal")
COOLING_ROOT = THERMAL_ROOT


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None


def read_number(path: Path) -> int | None:
    value = read_text(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def cpu_counters() -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    try:
        lines = Path("/proc/stat").read_text(encoding="ascii").splitlines()
    except OSError:
        return result
    for line in lines:
        fields = line.split()
        if not fields or not re.fullmatch(r"cpu(?:\d+)?", fields[0]):
            continue
        try:
            result[fields[0]] = [int(value) for value in fields[1:]]
        except ValueError:
            continue
    return result


def cpu_percentages(previous: dict[str, list[int]], current: dict[str, list[int]]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name, now in current.items():
        before = previous.get(name)
        if before is None or len(now) < 4 or len(before) < 4:
            values[name] = None
            continue
        total_now, total_before = sum(now), sum(before)
        idle_now = now[3] + (now[4] if len(now) > 4 else 0)
        idle_before = before[3] + (before[4] if len(before) > 4 else 0)
        delta_total = total_now - total_before
        delta_idle = idle_now - idle_before
        values[name] = (100.0 * (delta_total - delta_idle) / delta_total
                        if delta_total > 0 else None)
    return values


def process_cpu_ticks(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except OSError:
        return None
    right = raw.rfind(")")
    if right < 0:
        return None
    fields = raw[right + 1 :].split()  # begins with proc field 3 (state)
    try:
        return int(fields[11]) + int(fields[12])  # utime + stime, fields 14 and 15
    except (IndexError, ValueError):
        return None


def process_rss_kb(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def memory_status() -> dict[str, int | None]:
    values: dict[str, int | None] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    except OSError:
        return values
    for line in lines:
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached", "SwapTotal", "SwapFree"}:
            try:
                values[key + "_kB"] = int(rest.split()[0])
            except (ValueError, IndexError):
                values[key + "_kB"] = None
    if values.get("MemTotal_kB") is not None and values.get("MemAvailable_kB") is not None:
        values["MemUsed_kB"] = int(values["MemTotal_kB"] - values["MemAvailable_kB"])
    return values


def npu_load_raw() -> str | None:
    return read_text(NPU_DEVFREQ / "load")


def npu_load_is_reliable(raw: str | None, runtime_status: str | None) -> bool:
    # The initial idle probe returned 100@300000000Hz while runtime_status was
    # suspended. Treat this driver's node as untrustworthy until that sanity
    # failure is absent; never publish it as NPU utilization.
    match = re.fullmatch(r"(\d+)@(\d+)Hz", raw or "")
    if match is None or runtime_status == "suspended":
        return False
    return 0 <= int(match.group(1)) <= 100


def frequencies() -> dict[str, Any]:
    result: dict[str, Any] = {"npu_hz": read_number(NPU_DEVFREQ / "cur_freq")}
    policies: dict[str, Any] = {}
    if CPUFREQ_ROOT.exists():
        for path in sorted(CPUFREQ_ROOT.glob("policy*")):
            policies[path.name] = {
                "affected_cpus": read_text(path / "affected_cpus"),
                "current_khz": read_number(path / "scaling_cur_freq"),
                "min_khz": read_number(path / "scaling_min_freq"),
                "max_khz": read_number(path / "scaling_max_freq"),
                "cpuinfo_max_khz": read_number(path / "cpuinfo_max_freq"),
            }
    result["cpu_policies"] = policies
    return result


def temperatures() -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for path in sorted(THERMAL_ROOT.glob("thermal_zone*")):
        name = read_text(path / "type")
        value = read_number(path / "temp")
        if name is not None:
            result[name] = value
    return result


def cooling_devices() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted(COOLING_ROOT.glob("cooling_device*")):
        kind = read_text(path / "type")
        if kind is not None:
            result[path.name + ":" + kind] = {
                "current_state": read_number(path / "cur_state"),
                "max_state": read_number(path / "max_state"),
            }
    return result


def worker_metrics(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    metrics: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("type") == "metric":
            metrics.append(item)
        elif item.get("event_type") or item.get("type") == "event":
            events.append(item)
    completed = next((item for item in reversed(metrics) if item.get("status") == "completed"), None)
    return metrics, events, completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4))
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-period-s", type=float, default=1.0)
    parser.add_argument("--model-family", default="constructelligence-ppe")
    parser.add_argument("--track-logs", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="optional standalone command after --, monitored as one process")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_period_s <= 0:
        raise SystemExit("--sample-period-s must be positive")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    command_mode = bool(command)
    if command_mode:
        if args.workers not in (None, 1):
            raise SystemExit("standalone command monitoring supports one process")
        args.workers = 1
        for path in (args.model, args.video):
            if path is not None and not path.is_file():
                raise SystemExit(f"required file is missing: {path}")
    else:
        if args.workers not in (1, 2, 4) or args.worker is None or args.model is None or args.video is None:
            raise SystemExit("Worker mode requires --workers, --worker, --model and --video")
        for path in (args.worker, args.model, args.video):
            if not path.is_file():
                raise SystemExit(f"required file is missing: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    idle_load = npu_load_raw()
    idle_runtime = read_text(Path("/sys/devices/platform/27700000.npu/power/runtime_status"))
    npu_utilization_reliable = npu_load_is_reliable(idle_load, idle_runtime)
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    cpu_count = os.cpu_count() or 1
    worker_processes: list[dict[str, Any]] = []
    handles: list[Any] = []
    run_start_wall = utc_now()
    run_start_ns = time.monotonic_ns()
    previous_cpu = cpu_counters()
    previous_proc: dict[int, tuple[int, int]] = {}
    maximum_return_code = 0

    try:
        for index in range(args.workers):
            worker_dir = args.output_dir / f"worker-{index + 1:02d}"
            worker_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = worker_dir / "stdout.jsonl"
            stderr_path = worker_dir / "stderr.log"
            stdout_handle = stdout_path.open("w", encoding="utf-8", buffering=1)
            stderr_handle = stderr_path.open("w", encoding="utf-8", buffering=1)
            handles.extend((stdout_handle, stderr_handle))
            if command_mode:
                process_command = command
            else:
                process_command = [
                    str(args.worker), "--model", str(args.model), "--video", str(args.video),
                    "--state-dir", str(worker_dir / "state"),
                    "--video-backend", "gstreamer-mpp", "--preprocess-backend", "rga",
                    "--video-memory-mode", "dmabuf", "--rknn-input-mode", "shared",
                    "--input-pacing", "media", "--model-family", args.model_family,
                    "--queue-policy", "latest", "--queue-capacity", "1", "--confidence", "0.25",
                ]
                if args.track_logs:
                    process_command.extend(["--track-log", str(worker_dir / "tracks.jsonl")])
            spawn_ns = time.monotonic_ns()
            proc = subprocess.Popen(process_command, stdout=stdout_handle, stderr=stderr_handle)
            worker_processes.append({
                "index": index + 1,
                "pid": proc.pid,
                "spawn_monotonic_ns": spawn_ns,
                "spawn_delta_ms": (spawn_ns - run_start_ns) / 1e6,
                "process": proc,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "worker_dir": worker_dir,
            })

        sample_path = args.output_dir / "monitor.jsonl"
        next_sample = time.monotonic()
        with sample_path.open("w", encoding="utf-8", buffering=1) as stream:
            while any(item["process"].poll() is None for item in worker_processes):
                now = time.monotonic()
                if now < next_sample:
                    time.sleep(min(next_sample - now, 0.1))
                    continue
                current_cpu = cpu_counters()
                cpu = cpu_percentages(previous_cpu, current_cpu)
                interval_s = args.sample_period_s
                workers: dict[str, Any] = {}
                for item in worker_processes:
                    pid = item["pid"]
                    ticks = process_cpu_ticks(pid)
                    rss = process_rss_kb(pid)
                    old = previous_proc.get(pid)
                    process_core_pct = None
                    if ticks is not None and old is not None:
                        dt_s = max(1e-9, (time.monotonic_ns() - old[1]) / 1e9)
                        process_core_pct = max(0.0, (ticks - old[0]) / ticks_per_second / dt_s * 100.0)
                    if ticks is not None:
                        previous_proc[pid] = (ticks, time.monotonic_ns())
                    workers[str(item["index"])] = {
                        "pid": pid,
                        "alive": item["process"].poll() is None,
                        "cpu_one_core_percent": process_core_pct,
                        "cpu_system_percent": process_core_pct / cpu_count if process_core_pct is not None else None,
                        "rss_kb": rss,
                    }
                runtime_status = read_text(Path("/sys/devices/platform/27700000.npu/power/runtime_status"))
                load_raw = npu_load_raw()
                reliable_now = npu_utilization_reliable and npu_load_is_reliable(load_raw, runtime_status)
                record = {
                    "elapsed_s": (time.monotonic_ns() - run_start_ns) / 1e9,
                    "sampled_at_utc": utc_now(),
                    "cpu_utilization_percent": cpu,
                    "system_memory": memory_status(),
                    "workers": workers,
                    "npu": {
                        "utilization_percent": None,
                        "utilization_status": "available" if reliable_now else "unavailable",
                        "devfreq_load_raw": load_raw,
                        "cur_freq_hz": read_number(NPU_DEVFREQ / "cur_freq"),
                        "runtime_status": runtime_status,
                    },
                    "frequencies": frequencies(),
                    "temperatures_millicelsius": temperatures(),
                    "cooling_devices": cooling_devices(),
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                previous_cpu = current_cpu
                next_sample += args.sample_period_s
                if next_sample < time.monotonic() - args.sample_period_s:
                    next_sample = time.monotonic() + args.sample_period_s

        for item in worker_processes:
            item["return_code"] = item["process"].wait()
            maximum_return_code = max(maximum_return_code, int(item["return_code"]))
    except KeyboardInterrupt:
        for item in worker_processes:
            if item["process"].poll() is None:
                item["process"].send_signal(signal.SIGINT)
        for item in worker_processes:
            try:
                item["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                item["process"].terminate()
        raise
    finally:
        for handle in handles:
            handle.close()

    run_end_ns = time.monotonic_ns()
    summaries = []
    for item in worker_processes:
        metrics, events, completed = worker_metrics(item["stdout_path"])
        summaries.append({
            "worker": item["index"],
            "pid": item["pid"],
            "return_code": item.get("return_code"),
            "command": command if command_mode else None,
            "metrics_records": len(metrics),
            "inference_calls": completed.get("processed_frames") if completed else None,
            "decoded_frames": completed.get("decoded_frames") if completed else None,
            "fps": completed.get("throughput_fps") if completed else None,
            "queue_dropped_frames": completed.get("dropped_frames") if completed else None,
            "backend_dropped_frames": completed.get("backend_dropped_frames") if completed else None,
            "reconnect_stale_frames_dropped": completed.get("reconnect_stale_frames_dropped") if completed else None,
            "drop_rate_percent": (
                100.0 * (completed.get("dropped_frames", 0) + completed.get("backend_dropped_frames", 0) +
                         completed.get("reconnect_stale_frames_dropped", 0)) /
                max(1, completed.get("decoded_frames", 0) + completed.get("backend_dropped_frames", 0))
            ) if completed else None,
            "p50_rknn_run_ms": completed.get("p50_rknn_run_ms") if completed else None,
            "p95_rknn_run_ms": completed.get("p95_rknn_run_ms") if completed else None,
            "p50_end_to_end_ms": completed.get("p50_end_to_end_ms") if completed else None,
            "p95_end_to_end_ms": completed.get("p95_end_to_end_ms") if completed else None,
            "peak_rss_kb_reported_by_worker": completed.get("process_peak_rss_kb") if completed else None,
            "events": events,
            "worker_log": str(item["stdout_path"]),
            "stderr_log": str(item["stderr_path"]),
            "track_log": str(item["worker_dir"] / "tracks.jsonl") if args.track_logs else None,
        })

    manifest = {
        "experiment": "standalone_command_resource_monitor" if command_mode else "independent_worker_media_replay",
        "single_process_multicamera": False,
        "worker_count": args.workers,
        "model_family": args.model_family,
        "video_backend": None if command_mode else "gstreamer-mpp",
        "preprocess_backend": None if command_mode else "rga",
        "video_memory_mode": None if command_mode else "dmabuf",
        "rknn_input_mode": "rknn_inputs_set" if command_mode else "shared",
        "queue_policy": None if command_mode else "latest",
        "queue_capacity": None if command_mode else 1,
        "confidence_threshold": None if command_mode else 0.25,
        "input_pacing": "not applicable; offline image manifest" if command_mode else "source media rate",
        "model_path": str(args.model) if args.model else None,
        "model_sha256": sha256(args.model) if args.model else None,
        "video_path": str(args.video) if args.video else None,
        "video_sha256": sha256(args.video) if args.video else None,
        "input_fps_policy": "unpaced offline image manifest" if command_mode else
                            "media paced; file declares source frame rate",
        "sample_period_s": args.sample_period_s,
        "npu_utilization": "available" if npu_utilization_reliable else "unavailable",
        "npu_load_idle_raw": idle_load,
        "npu_runtime_idle": idle_runtime,
        "npu_utilization_note": (
            "devfreq load node passed idle sanity check" if npu_utilization_reliable else
            "devfreq load node failed idle sanity check; raw value is retained but not called utilization"
        ),
        "run_start_utc": run_start_wall,
        "run_end_utc": utc_now(),
        "elapsed_s": (run_end_ns - run_start_ns) / 1e9,
        "worker_spawn_delta_ms": [round(item["spawn_delta_ms"], 3) for item in worker_processes],
        "worker_summaries": summaries,
        "total_fps": round(sum(item["fps"] or 0.0 for item in summaries), 4) if not command_mode else None,
        "total_inference_calls": sum(item["inference_calls"] or 0 for item in summaries) if not command_mode else None,
        "command_mode": command_mode,
        "monitor_path": str(args.output_dir / "monitor.jsonl"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if maximum_return_code == 0 else maximum_return_code


if __name__ == "__main__":
    raise SystemExit(main())
