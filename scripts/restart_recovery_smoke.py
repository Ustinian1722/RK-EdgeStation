#!/usr/bin/env python3
"""Verify graceful RK-EdgeStation service restart and SQLite state recovery on Linux."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
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


def request_health(port: int, timeout: float = 3.0) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/health",
        headers={"Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def proc_snapshot(pid: int) -> dict[str, Any]:
    root = Path(f"/proc/{pid}")
    if not root.is_dir():
        return {"pid": pid, "alive": False}
    result: dict[str, Any] = {"pid": pid, "alive": True}
    try:
        status = (root / "status").read_text(encoding="ascii").splitlines()
        result["rss_kib"] = next(
            int(line.split()[1]) for line in status if line.startswith("VmRSS:")
        )
        result["threads"] = int(
            next(line.split()[1] for line in status if line.startswith("Threads:"))
        )
        result["fd_count"] = len(list((root / "fd").iterdir()))
    except (OSError, StopIteration, ValueError, IndexError):
        result["details_available"] = False
    return result


def inspect_database(state_dir: Path) -> dict[str, Any]:
    database = state_dir / "events.sqlite3"
    if not database.is_file():
        return {"integrity_check": "missing", "event_count": None, "missing_evidence": None}
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5.0)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            count = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            evidence_root = (state_dir / "evidence").resolve()
            missing_evidence = 0
            for (payload_json,) in connection.execute("SELECT payload_json FROM events"):
                try:
                    event = json.loads(payload_json)
                    evidence = Path(event.get("evidence", ""))
                    candidate = (state_dir / evidence).resolve()
                    if evidence_root not in candidate.parents or not candidate.is_file():
                        missing_evidence += 1
                except (TypeError, ValueError, OSError):
                    missing_evidence += 1
            return {
                "integrity_check": integrity,
                "event_count": count,
                "missing_evidence": missing_evidence,
            }
        finally:
            connection.close()
    except sqlite3.Error as error:
        return {
            "integrity_check": f"error: {error}",
            "event_count": None,
            "missing_evidence": None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True,
                        help="new state directory; existing paths are refused")
    parser.add_argument("--worker", type=Path, default=ROOT / "build/bin/rk_edge_worker")
    parser.add_argument("--port", type=int, default=18084)
    parser.add_argument("--observe-seconds", type=int, default=15)
    parser.add_argument("--startup-timeout", type=int, default=60)
    parser.add_argument("--source-commit", default="unknown")
    return parser.parse_args()


def check_health(health: dict[str, Any]) -> None:
    if health.get("status") != "running":
        raise RuntimeError(f"service worker status is {health.get('status')!r}")
    if health.get("worker_pid") is None:
        raise RuntimeError("health API did not report a worker PID")
    if health.get("storage_status") != "available":
        raise RuntimeError(f"storage status is {health.get('storage_status')!r}")
    if health.get("storage_failures", 0):
        raise RuntimeError("storage operation-failure counter is nonzero")
    if health.get("event_write_failures", 0) or health.get("metric_write_failures", 0):
        raise RuntimeError("storage write-failure counter is nonzero")


def start_service(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    port: int,
    timeout: int,
) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited during startup with code {process.returncode}")
        try:
            health = request_health(port)
            check_health(health)
            return process, health
        except (OSError, ValueError, urllib.error.URLError, RuntimeError) as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(f"service did not become healthy in {timeout}s: {last_error}")


def observe(
    process: subprocess.Popen[bytes],
    port: int,
    seconds: int,
    phase: str,
    sample_stream: Any,
) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    latest: dict[str, Any] = {}
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"service exited during {phase} with code {process.returncode}")
        try:
            latest = request_health(port)
        except (OSError, ValueError, urllib.error.URLError) as error:
            raise RuntimeError(f"health API failed during {phase}: {error}") from error
        check_health(latest)
        metrics = latest.get("metrics") or {}
        sample_stream.write(json.dumps({
            "timestamp_utc": utc_now(),
            "phase": phase,
            "worker_pid": latest.get("worker_pid"),
            "processed_frames": metrics.get("processed_frames"),
            "decoded_frames": metrics.get("decoded_frames"),
            "dropped_frames": metrics.get("dropped_frames"),
            "event_count": latest.get("event_count"),
            "storage_status": latest.get("storage_status"),
            "storage_failures": latest.get("storage_failures"),
        }, separators=(",", ":")) + "\n")
        sample_stream.flush()
        if time.monotonic() >= deadline:
            return latest
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def stop_service(process: subprocess.Popen[bytes]) -> int:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            return process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError("service required SIGKILL after the 20-second graceful-stop window")
    return int(process.returncode)


def wait_pid_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return True
        time.sleep(0.1)
    return not Path(f"/proc/{pid}").exists()


def main() -> int:
    args = parse_args()
    model = args.model.resolve()
    video = args.video.resolve()
    worker = args.worker.resolve()
    state_dir = args.state_dir.resolve()
    if not model.is_file() or not video.is_file() or not worker.is_file():
        raise SystemExit("model, video, and worker must all exist")
    if args.observe_seconds <= 0 or args.startup_timeout <= 0:
        raise SystemExit("observe duration and startup timeout must be positive")
    if not (1 <= args.port <= 65535):
        raise SystemExit("port must be 1..65535")
    if state_dir.exists():
        raise SystemExit(f"refusing existing state path: {state_dir}")
    state_dir.mkdir(parents=True)
    monitor_dir = state_dir / "monitor"
    monitor_dir.mkdir()
    samples_path = monitor_dir / "samples.jsonl"
    summary_path = monitor_dir / "summary.json"

    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as error:
            raise SystemExit(f"loopback port {args.port} is not available: {error}") from error

    command = [
        sys.executable, "-m", "edge_station.server",
        "--worker", str(worker), "--model", str(model), "--video", str(video),
        "--state-dir", str(state_dir), "--port", str(args.port), "--loop-video",
        "--queue-capacity", "1", "--queue-policy", "latest",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + environment.get("PYTHONPATH", "")
    started_utc = utc_now()
    started_monotonic = time.monotonic()
    processes: list[subprocess.Popen[bytes]] = []
    phase_records: list[dict[str, Any]] = []
    result = "failed"
    error = ""

    try:
        with samples_path.open("w", encoding="utf-8", buffering=1) as samples:
            first_process, first_health = start_service(
                command, environment, monitor_dir / "service-1.log",
                args.port, args.startup_timeout,
            )
            processes.append(first_process)
            first_pid = int(first_health["worker_pid"])
            first_health = observe(
                first_process, args.port, args.observe_seconds, "before_restart", samples
            )
            first_resources = {
                "service": proc_snapshot(first_process.pid),
                "worker": proc_snapshot(first_pid),
            }
            if not first_health.get("event_count"):
                raise RuntimeError("no event was persisted before the restart")
            first_exit = stop_service(first_process)
            first_worker_stopped = wait_pid_gone(first_pid)
            if first_exit != 0 or not first_worker_stopped:
                raise RuntimeError(
                    f"first shutdown failed: service_exit={first_exit}, worker_stopped={first_worker_stopped}"
                )
            first_db = inspect_database(state_dir)
            if first_db["integrity_check"] != "ok" or first_db["missing_evidence"] != 0:
                raise RuntimeError(f"database/evidence check failed after first stop: {first_db}")
            phase_records.append({
                "phase": "before_restart",
                "service_pid": first_process.pid,
                "worker_pid": first_pid,
                "health": first_health,
                "resources_before_stop": first_resources,
                "service_exit_code": first_exit,
                "worker_stopped": first_worker_stopped,
                "database_after_stop": first_db,
            })

            second_process, second_health = start_service(
                command, environment, monitor_dir / "service-2.log",
                args.port, args.startup_timeout,
            )
            processes.append(second_process)
            second_pid = int(second_health["worker_pid"])
            if second_pid == first_pid:
                raise RuntimeError("worker PID did not change across the service restart")
            count_after_restart = second_health.get("event_count")
            count_before_restart = first_db.get("event_count")
            if count_after_restart is None or count_after_restart < count_before_restart:
                raise RuntimeError(
                    f"persisted event count regressed: before={count_before_restart}, "
                    f"after={count_after_restart}"
                )
            second_health = observe(
                second_process, args.port, args.observe_seconds, "after_restart", samples
            )
            second_resources = {
                "service": proc_snapshot(second_process.pid),
                "worker": proc_snapshot(second_pid),
            }
            second_exit = stop_service(second_process)
            second_worker_stopped = wait_pid_gone(second_pid)
            second_db = inspect_database(state_dir)
            if second_exit != 0 or not second_worker_stopped:
                raise RuntimeError(
                    f"second shutdown failed: service_exit={second_exit}, "
                    f"worker_stopped={second_worker_stopped}"
                )
            if second_db["integrity_check"] != "ok" or second_db["missing_evidence"] != 0:
                raise RuntimeError(f"database/evidence check failed after restart: {second_db}")
            if second_db["event_count"] < first_db["event_count"]:
                raise RuntimeError("final persisted event count is below the pre-restart count")
            phase_records.append({
                "phase": "after_restart",
                "service_pid": second_process.pid,
                "worker_pid": second_pid,
                "health": second_health,
                "resources_before_stop": second_resources,
                "service_exit_code": second_exit,
                "worker_stopped": second_worker_stopped,
                "database_after_stop": second_db,
                "retained_event_count": count_after_restart,
            })
        result = "passed"
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    stop_service(process)
                except Exception as failure:
                    if not error:
                        error = f"cleanup failure: {type(failure).__name__}: {failure}"
        if error:
            result = "failed"
        elapsed = round(time.monotonic() - started_monotonic, 2)
        final_db = inspect_database(state_dir)
        summary = {
            "result": result,
            "error": error or None,
            "started_utc": started_utc,
            "finished_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "source_commit": args.source_commit,
            "model": str(model),
            "model_sha256": sha256_file(model),
            "video": str(video),
            "video_sha256": sha256_file(video),
            "worker": str(worker),
            "worker_sha256": sha256_file(worker),
            "port": args.port,
            "observe_seconds_per_run": args.observe_seconds,
            "phases": phase_records,
            "final_database": final_db,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if result == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
