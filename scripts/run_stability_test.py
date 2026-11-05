#!/usr/bin/env python3
"""Run a bounded, monitored RK-EdgeStation board soak against a looped local clip."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_health(port: int, timeout: float = 5.0) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/health", headers={"Cache-Control": "no-cache"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def read_cpu_counters() -> tuple[int, int] | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()
        values = [int(value) for value in fields[1:]]
    except (OSError, ValueError, IndexError):
        return None
    if len(values) < 4:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def cpu_percent(previous: tuple[int, int] | None,
                current: tuple[int, int] | None) -> float | None:
    if previous is None or current is None:
        return None
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return None
    return round(100.0 * (total_delta - idle_delta) / total_delta, 2)


def thermal_temperatures() -> dict[str, float]:
    values: dict[str, float] = {}
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            raw = float(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        values[path.parent.name] = round(raw / 1000.0 if abs(raw) > 1000 else raw, 2)
    return values


def process_counters(pid: int | None) -> tuple[int, int] | None:
    if pid is None:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close = stat.rfind(")")
        fields = stat[close + 2:].split()
        if close < 0 or len(fields) < 13:
            return None
        cpu_ticks = int(fields[11]) + int(fields[12])
        status = Path(f"/proc/{pid}/status").read_text(encoding="ascii")
        rss_kib = next(int(line.split()[1]) for line in status.splitlines()
                       if line.startswith("VmRSS:"))
        return cpu_ticks, rss_kib
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def process_usage(previous: tuple[int, int] | None,
                  current: tuple[int, int] | None,
                  interval_seconds: float,
                  clock_ticks: int) -> dict[str, float | int] | None:
    if current is None:
        return None
    usage: dict[str, float | int] = {"rss_kib": current[1]}
    if previous is not None and interval_seconds > 0 and clock_ticks > 0:
        usage["cpu_percent"] = round(
            100.0 * (current[0] - previous[0]) / clock_ticks / interval_seconds, 2
        )
    else:
        usage["cpu_percent"] = 0.0
    return usage


def npu_load_sample() -> str | None:
    for path in (
        Path("/sys/kernel/debug/rknpu/load"),
        Path("/sys/class/devfreq/fdab0000.npu/load"),
    ):
        try:
            return path.read_text(encoding="ascii").strip()
        except OSError:
            pass
    return None


def available_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def tree_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).stat().st_size
            except OSError:
                continue
    return total


def inspect_database(state_dir: Path) -> dict[str, Any]:
    database = state_dir / "events.sqlite3"
    if not database.is_file():
        return {"integrity_check": "missing", "event_count": None,
                "missing_evidence": None}
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5.0)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            count = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            missing_evidence = 0
            evidence_root = (state_dir / "evidence").resolve()
            for (payload_json,) in connection.execute("SELECT payload_json FROM events"):
                try:
                    event = json.loads(payload_json)
                    evidence = Path(event.get("evidence", ""))
                    candidate = (state_dir / evidence).resolve()
                    if evidence_root not in candidate.parents or not candidate.is_file():
                        missing_evidence += 1
                except (TypeError, ValueError, OSError):
                    missing_evidence += 1
            return {"integrity_check": integrity, "event_count": count,
                    "missing_evidence": missing_evidence}
        finally:
            connection.close()
    except sqlite3.Error as error:
        return {"integrity_check": f"error: {error}", "event_count": None,
                "missing_evidence": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True,
                        help="new, isolated output directory; existing non-empty paths are refused")
    parser.add_argument("--worker", type=Path, default=ROOT / "build/bin/rk_edge_worker")
    parser.add_argument("--duration-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--sample-seconds", type=int, default=60)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--max-state-gib", type=float, default=4.0)
    parser.add_argument("--min-mem-available-mib", type=float, default=512.0)
    parser.add_argument("--max-temp-c", type=float, default=80.0)
    parser.add_argument("--max-worker-rss-mib", type=float, default=2048.0)
    parser.add_argument("--roi", default="0.40,0.25,0.30,0.40")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--dwell-ms", type=int, default=1200)
    parser.add_argument("--roi-debounce-frames", type=int, default=2)
    parser.add_argument("--queue-capacity", type=int, default=1)
    parser.add_argument("--queue-policy", choices=("latest", "fifo"), default="latest")
    parser.add_argument("--source-commit", default="unknown")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    video = args.video.resolve()
    worker = args.worker.resolve()
    state_dir = args.state_dir.resolve()
    if not model.is_file() or not video.is_file() or not worker.is_file():
        raise SystemExit("model, video, and worker must all exist")
    if args.duration_seconds <= 0 or args.sample_seconds <= 0:
        raise SystemExit("duration and sample interval must be positive")
    if not (1 <= args.port <= 65535):
        raise SystemExit("port must be 1..65535")
    if state_dir.exists() and any(state_dir.iterdir()):
        raise SystemExit(f"refusing non-empty state directory: {state_dir}")
    state_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = state_dir / "monitor"
    monitor_dir.mkdir()
    monitor_path = monitor_dir / "samples.jsonl"
    server_log_path = monitor_dir / "server.log"
    summary_path = monitor_dir / "summary.json"
    free_limit = int(args.min_free_gib * 1024**3)
    state_limit = int(args.max_state_gib * 1024**3)
    memory_limit = int(args.min_mem_available_mib * 1024**2)
    rss_limit = int(args.max_worker_rss_mib * 1024**2)

    command = [
        sys.executable, "-m", "edge_station.server",
        "--worker", str(worker), "--model", str(model), "--video", str(video),
        "--state-dir", str(state_dir), "--port", str(args.port), "--loop-video",
        "--roi", args.roi, "--confidence", str(args.confidence),
        "--dwell-ms", str(args.dwell_ms),
        "--roi-debounce-frames", str(args.roi_debounce_frames),
        "--queue-capacity", str(args.queue_capacity),
        "--queue-policy", args.queue_policy,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + environment.get("PYTHONPATH", "")
    started_utc = utc_now()
    started_monotonic = time.monotonic()
    metadata = {
        "started_utc": started_utc,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "source_commit": args.source_commit,
        "model": str(model),
        "model_sha256": sha256_file(model),
        "video": str(video),
        "video_sha256": sha256_file(video),
        "worker": str(worker),
        "worker_sha256": sha256_file(worker),
        "command": command,
        "duration_seconds": args.duration_seconds,
        "sample_seconds": args.sample_seconds,
        "safety_limits": {
            "min_free_bytes": free_limit,
            "max_state_bytes": state_limit,
            "min_mem_available_bytes": memory_limit,
            "max_worker_rss_bytes": rss_limit,
            "max_temperature_c": args.max_temp_c,
        },
    }
    (monitor_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    old_handlers: dict[int, Any] = {}
    interrupted = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        print(f"received signal {signum}; stopping soak", file=sys.stderr, flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, request_stop)

    result = "failed"
    error = ""
    samples = 0
    worker_pid: int | None = None
    worker_pid_changes = 0
    previous_cpu: tuple[int, int] | None = None
    previous_process: tuple[int, int] | None = None
    previous_sample_at: float | None = None
    previous_worker_pid: int | None = None
    last_health: dict[str, Any] = {}
    server_exit_code: int | None = None

    try:
        with server_log_path.open("ab", buffering=0) as server_log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment,
                stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            deadline = started_monotonic + args.duration_seconds
            startup_deadline = min(deadline, time.monotonic() + 60)
            while time.monotonic() < startup_deadline and not interrupted:
                if process.poll() is not None:
                    raise RuntimeError(f"service exited during startup with code {process.returncode}")
                try:
                    last_health = request_health(args.port)
                    if last_health.get("status") == "running":
                        break
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("service/worker did not reach running state within 60 seconds")

            with monitor_path.open("w", encoding="utf-8", buffering=1) as monitor:
                while time.monotonic() < deadline and not interrupted:
                    if process.poll() is not None:
                        raise RuntimeError(f"service exited with code {process.returncode}")
                    try:
                        health = request_health(args.port)
                    except (OSError, ValueError, urllib.error.URLError) as failure:
                        raise RuntimeError(f"health API request failed: {failure}") from failure
                    last_health = health
                    if health.get("status") != "running":
                        raise RuntimeError(f"worker left running state: {health.get('status')}")
                    if health.get("storage_status") != "available":
                        raise RuntimeError(f"storage became unavailable: {health.get('storage_last_error')}")
                    if health.get("storage_failures", 0):
                        raise RuntimeError("storage operation-failure counter increased")
                    if health.get("event_write_failures", 0) or health.get("metric_write_failures", 0):
                        raise RuntimeError("storage write-failure counter increased")

                    worker_pid = health.get("worker_pid")
                    if previous_worker_pid is not None and worker_pid != previous_worker_pid:
                        worker_pid_changes += 1
                        raise RuntimeError("worker PID changed during the continuous run")
                    previous_worker_pid = worker_pid
                    metrics = health.get("metrics") or {}
                    sample_at = time.monotonic()
                    process_now = process_counters(worker_pid)
                    usage = process_usage(
                        previous_process, process_now,
                        sample_at - previous_sample_at if previous_sample_at is not None else 0.0,
                        int(os.sysconf("SC_CLK_TCK")),
                    )
                    previous_process = process_now
                    previous_sample_at = sample_at
                    temperatures = thermal_temperatures()
                    disk = shutil.disk_usage(state_dir)
                    memory_available = available_memory_bytes()
                    now_cpu = read_cpu_counters()
                    row = {
                        "timestamp_utc": utc_now(),
                        "elapsed_seconds": round(time.monotonic() - started_monotonic, 2),
                        "worker_pid": worker_pid,
                        "worker_status": health.get("status"),
                        "processed_frames": metrics.get("processed_frames"),
                        "decoded_frames": metrics.get("decoded_frames"),
                        "throughput_fps": metrics.get("throughput_fps"),
                        "last_inference_ms": metrics.get("last_inference_ms"),
                        "p95_end_to_end_ms": metrics.get("p95_end_to_end_ms"),
                        "dropped_frames": metrics.get("dropped_frames"),
                        "event_count": health.get("event_count"),
                        "storage_failures": health.get("storage_failures"),
                        "event_write_failures": health.get("event_write_failures"),
                        "metric_write_failures": health.get("metric_write_failures"),
                        "process_usage": usage,
                        "system_cpu_percent": cpu_percent(previous_cpu, now_cpu),
                        "memory_available_bytes": memory_available,
                        "thermal_c": temperatures,
                        "npu_load_raw": npu_load_sample(),
                        "disk_free_bytes": disk.free,
                    }
                    previous_cpu = now_cpu
                    monitor.write(json.dumps(row, separators=(",", ":")) + "\n")
                    monitor.flush()
                    samples += 1

                    if disk.free < free_limit:
                        raise RuntimeError(f"free disk below safety limit: {disk.free} bytes")
                    if memory_available is not None and memory_available < memory_limit:
                        raise RuntimeError(f"MemAvailable below safety limit: {memory_available} bytes")
                    if usage is not None and usage["rss_kib"] * 1024 > rss_limit:
                        raise RuntimeError(f"worker RSS above safety limit: {usage['rss_kib']} KiB")
                    hot_zones = {name: value for name, value in temperatures.items()
                                 if value >= args.max_temp_c}
                    if hot_zones:
                        raise RuntimeError(f"thermal safety limit reached: {hot_zones}")
                    if samples % 10 == 0 and tree_size(state_dir) > state_limit:
                        raise RuntimeError("soak output directory exceeded its size safety limit")

                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(args.sample_seconds, remaining))
            if not interrupted and time.monotonic() >= deadline:
                result = "passed"
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        print(error, file=sys.stderr, flush=True)
    finally:
        if "process" in locals() and process.poll() is None:
            try:
                process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server_exit_code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                server_exit_code = process.wait(timeout=5)
                if not error:
                    error = "service did not stop within 20 seconds and was killed"
                    result = "failed"
        elif "process" in locals():
            server_exit_code = process.returncode
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)

    elapsed = round(time.monotonic() - started_monotonic, 2)
    database_result = inspect_database(state_dir)
    if result == "passed":
        if server_exit_code != 0:
            result = "failed"
            error = f"service exited with code {server_exit_code} after the timed run"
        elif database_result.get("integrity_check") != "ok":
            result = "failed"
            error = f"SQLite integrity check failed: {database_result}"
        elif database_result.get("missing_evidence") != 0:
            result = "failed"
            error = f"event evidence references missing files: {database_result}"
        elif (last_health.get("event_count") is not None and
              database_result.get("event_count") < last_health.get("event_count")):
            result = "failed"
            error = "final SQLite event count is below the last health API count"
    summary = {
        "result": result,
        "error": error or None,
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "requested_duration_seconds": args.duration_seconds,
        "sample_count": samples,
        "worker_pid_changes": worker_pid_changes,
        "service_exit_code": server_exit_code,
        "last_health": last_health,
        "database": database_result,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if result == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
