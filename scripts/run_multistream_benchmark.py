#!/usr/bin/env python3
"""Run one RK-EdgeStation Worker with board-side system telemetry.

Designed for short 1/2/4/8/16/32-source RTSP scaling runs on RK3576. It records
the exact Worker JSON output and aligned /proc, sysfs, frequency, and thermal
samples. NPU utilization stays unavailable because this RK3576 driver's load
node failed an idle sanity check (100@300MHz while the runtime was suspended).
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
import threading
import time
from pathlib import Path
from typing import Any


NPU_ROOT = Path("/sys/class/devfreq/27700000.npu")
CPUFREQ_ROOT = Path("/sys/devices/system/cpu/cpufreq")
THERMAL_ROOT = Path("/sys/class/thermal")
KNOWN_UNRELIABLE_NPU_LOAD = "prior idle sanity check returned 100@300MHz while runtime was suspended"


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None


def read_int(path: Path) -> int | None:
    value = read_text(path)
    try:
        return int(value) if value is not None else None
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
        if fields and re.fullmatch(r"cpu(?:\d+)?", fields[0]):
            try:
                result[fields[0]] = [int(value) for value in fields[1:]]
            except ValueError:
                pass
    return result


def cpu_percentages(before: dict[str, list[int]], after: dict[str, list[int]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, current in after.items():
        previous = before.get(name)
        if previous is None or len(current) < 4 or len(previous) < 4:
            result[name] = None
            continue
        total_delta = sum(current) - sum(previous)
        idle_now = current[3] + (current[4] if len(current) > 4 else 0)
        idle_before = previous[3] + (previous[4] if len(previous) > 4 else 0)
        idle_delta = idle_now - idle_before
        result[name] = 100.0 * (total_delta - idle_delta) / total_delta if total_delta > 0 else None
    return result


def process_ticks(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = raw[raw.rfind(")") + 1 :].split()
        return int(fields[11]) + int(fields[12])
    except (OSError, ValueError, IndexError):
        return None


def process_rss_kb(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def memory_info() -> dict[str, int | None]:
    keys = {"MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached", "SwapTotal", "SwapFree"}
    result: dict[str, int | None] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    except OSError:
        return result
    for line in lines:
        key, _, rest = line.partition(":")
        if key in keys:
            try:
                result[key + "_kB"] = int(rest.split()[0])
            except (ValueError, IndexError):
                result[key + "_kB"] = None
    if result.get("MemTotal_kB") is not None and result.get("MemAvailable_kB") is not None:
        result["MemUsed_kB"] = int(result["MemTotal_kB"] - result["MemAvailable_kB"])
    return result


def frequencies() -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for path in sorted(CPUFREQ_ROOT.glob("policy*")):
        policies[path.name] = {
            "affected_cpus": read_text(path / "affected_cpus"),
            "current_khz": read_int(path / "scaling_cur_freq"),
            "min_khz": read_int(path / "scaling_min_freq"),
            "max_khz": read_int(path / "scaling_max_freq"),
        }
    return {"npu_hz": read_int(NPU_ROOT / "cur_freq"), "cpu_policies": policies}


def npu_counter_status() -> dict[str, Any]:
    runtime_candidates = [
        NPU_ROOT / "device" / "power" / "runtime_status",
        NPU_ROOT / "device" / "rknpu" / "runtime_status",
        NPU_ROOT / "rknpu" / "runtime_status",
    ]
    runtime = next((read_text(path) for path in runtime_candidates if read_text(path) is not None), None)
    return {
        "utilization": "unavailable",
        "reason": KNOWN_UNRELIABLE_NPU_LOAD,
        "raw_load": read_text(NPU_ROOT / "load"),
        "runtime_status": runtime,
        "counter_paths": [str(path) for path in [NPU_ROOT / "load", *runtime_candidates]
                          if path.exists()],
    }


def temperatures() -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for path in sorted(THERMAL_ROOT.glob("thermal_zone*")):
        kind = read_text(path / "type")
        value = read_int(path / "temp")
        if kind is not None:
            result[kind] = value
    return result


def cooling_states() -> dict[str, dict[str, int | None]]:
    result: dict[str, dict[str, int | None]] = {}
    for path in sorted(THERMAL_ROOT.glob("cooling_device*")):
        kind = read_text(path / "type")
        if kind is not None:
            result[path.name + ":" + kind] = {
                "current_state": read_int(path / "cur_state"),
                "max_state": read_int(path / "max_state"),
            }
    return result


def parse_worker_metrics(log_path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    cameras: dict[str, dict[str, Any]] = {}
    all_metrics: list[dict[str, Any]] = []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return cameras, all_metrics
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("type") == "camera_metric" and isinstance(item.get("camera_id"), str):
            cameras[item["camera_id"]] = item
            all_metrics.append(item)
    return cameras, all_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--sample-period-s", type=float, default=1.0)
    parser.add_argument("--model-family", default="constructelligence-ppe")
    parser.add_argument("--video-backend", default="gstreamer-mpp")
    parser.add_argument("--preprocess-backend", default="rga")
    parser.add_argument("--video-memory-mode", default="copy")
    parser.add_argument("--rknn-input-mode", default="shared")
    parser.add_argument("--input-pacing", default="media")
    parser.add_argument("--queue-policy", choices=("latest", "fifo"), default="latest")
    parser.add_argument("--loop-video", action="store_true")
    parser.add_argument("--settle-s", type=float, default=2.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for path in (args.worker, args.model, args.config):
        if not path.is_file():
            raise SystemExit(f"required input is missing: {path}")
    if args.duration_s <= 0 or args.sample_period_s <= 0 or args.settle_s < 0:
        raise SystemExit("duration and sample period must be positive; settle time cannot be negative")
    config_data = json.loads(args.config.read_text(encoding="utf-8"))
    cameras = config_data.get("cameras") if isinstance(config_data, dict) else None
    if not isinstance(cameras, list) or not cameras:
        raise SystemExit("camera config must contain a non-empty cameras array")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "worker.jsonl"
    telemetry_path = args.output_dir / "telemetry.jsonl"
    summary_path = args.output_dir / "summary.json"
    command = [
        str(args.worker), "--model", str(args.model), "--model-family", args.model_family,
        "--camera-config", str(args.config), "--video-backend", args.video_backend,
        "--preprocess-backend", args.preprocess_backend, "--video-memory-mode", args.video_memory_mode,
        "--rknn-input-mode", args.rknn_input_mode, "--input-pacing", args.input_pacing,
        "--queue-policy", args.queue_policy, "--queue-capacity", "1", "--state-dir",
        str(args.output_dir / "worker-state"),
    ]
    if args.loop_video:
        command.append("--loop-video")

    if args.settle_s:
        time.sleep(args.settle_s)
    wall_start = utc_now()
    monotonic_start = time.monotonic()
    baseline_ticks = os.sysconf("SC_CLK_TCK")
    npu_status = npu_counter_status()
    with raw_path.open("w", encoding="utf-8", buffering=1) as raw_log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        assert process.stdout is not None
        reader_error: list[str] = []

        def copy_worker_log() -> None:
            try:
                for line in process.stdout:
                    raw_log.write(line)
            except Exception as error:  # preserve an unexpected reader failure for the summary
                reader_error.append(f"{type(error).__name__}: {error}")

        reader = threading.Thread(target=copy_worker_log, name="worker-log-reader", daemon=True)
        reader.start()
        samples: list[dict[str, Any]] = []
        previous_counters = cpu_counters()
        previous_ticks = process_ticks(process.pid)
        previous_time = time.monotonic()
        deadline = monotonic_start + args.duration_s
        next_sample = previous_time
        while time.monotonic() < deadline and process.poll() is None:
            next_sample += args.sample_period_s
            time.sleep(max(0.0, min(next_sample - time.monotonic(), max(deadline - time.monotonic(), 0.0))))
            now = time.monotonic()
            current_counters = cpu_counters()
            interval = max(now - previous_time, 1e-6)
            current_ticks = process_ticks(process.pid)
            process_cpu = None
            if current_ticks is not None and previous_ticks is not None:
                process_cpu = 100.0 * (current_ticks - previous_ticks) / baseline_ticks / interval
            sample = {
                "timestamp_utc": utc_now(),
                "elapsed_s": round(now - monotonic_start, 3),
                "cpu_percent": cpu_percentages(previous_counters, current_counters),
                "worker_pid": process.pid,
                "worker_cpu_core_percent": process_cpu,
                "worker_rss_kb": process_rss_kb(process.pid),
                "memory": memory_info(),
                "frequencies": frequencies(),
                "temperatures_milli_c": temperatures(),
                "cooling_devices": cooling_states(),
                "npu": npu_counter_status(),
            }
            samples.append(sample)
            with telemetry_path.open("a", encoding="utf-8") as telemetry:
                telemetry.write(json.dumps(sample, separators=(",", ":")) + "\n")
            previous_counters = current_counters
            previous_ticks = current_ticks
            previous_time = now
        natural_exit = process.poll() is not None
        if not natural_exit:
            process.send_signal(signal.SIGTERM)
        try:
            exit_code = process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            process.kill()
            exit_code = process.wait(timeout=5)
        reader.join(timeout=3)
        raw_log.flush()

    wall_end = utc_now()
    run_seconds = max(time.monotonic() - monotonic_start, 1e-6)
    camera_metrics, metric_history = parse_worker_metrics(raw_path)
    telemetry_cpu: dict[str, list[float]] = {}
    process_cpu_values: list[float] = []
    rss_values: list[int] = []
    temp_values: dict[str, list[int]] = {}
    npu_freq_values: list[int] = []
    cpu_freq_values: dict[str, list[int]] = {}
    throttle_active = False
    for sample in samples:
        for cpu, value in sample["cpu_percent"].items():
            if value is not None:
                telemetry_cpu.setdefault(cpu, []).append(float(value))
        if sample["worker_cpu_core_percent"] is not None:
            process_cpu_values.append(float(sample["worker_cpu_core_percent"]))
        if sample["worker_rss_kb"] is not None:
            rss_values.append(int(sample["worker_rss_kb"]))
        for name, value in sample["temperatures_milli_c"].items():
            if value is not None:
                temp_values.setdefault(name, []).append(int(value))
        freq = sample["frequencies"]
        if freq.get("npu_hz") is not None:
            npu_freq_values.append(int(freq["npu_hz"]))
        for name, policy in freq.get("cpu_policies", {}).items():
            if policy.get("current_khz") is not None:
                cpu_freq_values.setdefault(name, []).append(int(policy["current_khz"]))
        if any((device.get("current_state") or 0) > 0 for device in sample["cooling_devices"].values()):
            throttle_active = True

    processed = sum(int(item.get("processed_frames", 0)) for item in camera_metrics.values())
    decoded = sum(int(item.get("decoded_frames", 0)) for item in camera_metrics.values())
    dropped = sum(int(item.get("dropped_frames", 0)) for item in camera_metrics.values())
    summary = {
        "schema_version": 1,
        "run_id": args.output_dir.name,
        "started_utc": wall_start,
        "ended_utc": wall_end,
        "duration_s": round(run_seconds, 3),
        "sample_period_s": args.sample_period_s,
        "camera_count": len(cameras),
        "source_kind": "synthetic moving H.264 RTSP test patterns; not PPE accuracy data",
        "camera_config_sha256": sha256(args.config),
        "model_sha256": sha256(args.model),
        "worker_sha256": sha256(args.worker),
        "worker_command": command,
        "worker_exit_code": exit_code,
        "worker_natural_exit_before_duration": natural_exit,
        "worker_log_reader_error": reader_error or None,
        "input_frames_decoded": decoded,
        "inference_calls": sum(int(item.get("inference_calls", 0)) for item in camera_metrics.values()),
        "processed_frames": processed,
        "dropped_frames": dropped,
        "aggregate_processed_fps": round(processed / run_seconds, 4),
        "aggregate_decoded_fps": round(decoded / run_seconds, 4),
        "drop_rate_percent": round(100 * dropped / max(decoded, 1), 3),
        "npu_utilization": "unavailable",
        "npu_counter_diagnostic": npu_status,
        "system_cpu_percent_mean": {
            cpu: round(sum(values) / len(values), 2) for cpu, values in telemetry_cpu.items() if values
        },
        "worker_cpu_core_percent_mean": round(sum(process_cpu_values) / len(process_cpu_values), 2) if process_cpu_values else None,
        "worker_cpu_core_percent_peak": round(max(process_cpu_values), 2) if process_cpu_values else None,
        "worker_rss_kb_mean": round(sum(rss_values) / len(rss_values)) if rss_values else None,
        "worker_rss_kb_peak": max(rss_values) if rss_values else None,
        "system_memory_last": samples[-1]["memory"] if samples else memory_info(),
        "temperature_max_milli_c": {name: max(values) for name, values in temp_values.items() if values},
        "cpu_frequency_khz_minmax": {name: {"min": min(values), "max": max(values)}
                                      for name, values in cpu_freq_values.items() if values},
        "npu_frequency_hz_minmax": {"min": min(npu_freq_values), "max": max(npu_freq_values)} if npu_freq_values else None,
        "cooling_state_nonzero_observed": throttle_active,
        "telemetry_samples": len(samples),
        "camera_metrics": camera_metrics,
        "raw_metric_records": len(metric_history),
        "artifacts": {"worker_log": raw_path.name, "telemetry": telemetry_path.name},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_id": summary["run_id"], "camera_count": len(cameras),
        "duration_s": summary["duration_s"], "aggregate_processed_fps": summary["aggregate_processed_fps"],
        "processed_frames": processed, "dropped_frames": dropped, "worker_exit_code": exit_code,
        "worker_rss_kb_peak": summary["worker_rss_kb_peak"],
        "worker_cpu_core_percent_mean": summary["worker_cpu_core_percent_mean"],
        "npu_utilization": "unavailable", "telemetry_samples": len(samples),
        "summary": str(summary_path),
    }, separators=(",", ":")), flush=True)
    return 0 if exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
