from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .media_source import display_source, is_rtsp_source, resolve_worker_source
from .storage import EventStore


ROOT = Path(__file__).resolve().parents[2]


class StorageUnavailable(RuntimeError):
    pass


class AppState:
    def __init__(
        self,
        store: EventStore | None,
        evidence_dir: Path,
        store_path: Path | None = None,
        storage_error: str | None = None,
    ):
        self.store = store
        self.store_path = Path(store_path) if store_path is not None else None
        self.evidence_dir = evidence_dir.resolve()
        self.started_at = time.monotonic()
        self.lock = threading.Lock()
        self.storage_init_lock = threading.Lock()
        self.storage_status = "available" if store is not None else "unavailable"
        self.storage_error = storage_error
        self.storage_failures = 1 if storage_error else 0
        self.event_write_failures = 0
        self.metric_write_failures = 0
        if storage_error:
            print(f"SQLite initialization failed: {storage_error}", file=sys.stderr)
        self.worker_status = "idle"
        self.worker_pid: int | None = None
        self.worker_exit_code: int | None = None
        self.worker_failure = threading.Event()
        self.metrics: dict[str, Any] = {}
        self.model_name = ""
        self.video_name = ""
        self.process: subprocess.Popen[str] | None = None
        self.reader: threading.Thread | None = None

    def _mark_storage_failure(self, operation: str, error: Exception) -> str:
        detail = f"{operation}: {type(error).__name__}: {error}"
        with self.lock:
            should_log = self.storage_status != "unavailable"
            self.storage_status = "unavailable"
            self.storage_error = detail
            self.storage_failures += 1
            if operation in ("write event", "acknowledge event"):
                self.event_write_failures += 1
            elif operation == "write metric":
                self.metric_write_failures += 1
        if should_log:
            print(f"SQLite unavailable: {detail}", file=sys.stderr)
        return detail

    def _mark_storage_available(self) -> None:
        with self.lock:
            self.storage_status = "available"

    def _get_store(self) -> EventStore:
        with self.storage_init_lock:
            if self.store is not None:
                return self.store
            if self.store_path is None:
                error = StorageUnavailable(self.storage_error or "database path is not configured")
                self._mark_storage_failure("initialize", error)
                raise error
            try:
                self.store = EventStore(self.store_path)
            except (OSError, sqlite3.Error) as error:
                detail = self._mark_storage_failure("initialize", error)
                raise StorageUnavailable(detail) from error
            self._mark_storage_available()
            return self.store

    def read_events(self, limit: int) -> list[dict[str, Any]]:
        try:
            events = self._get_store().events(limit)
        except StorageUnavailable:
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError) as error:
            detail = self._mark_storage_failure("read events", error)
            raise StorageUnavailable(detail) from error
        self._mark_storage_available()
        return events

    def acknowledge_event(self, event_id: str, acknowledged_by: str) -> bool:
        try:
            found = self._get_store().acknowledge_event(event_id, acknowledged_by)
        except StorageUnavailable:
            raise
        except (OSError, sqlite3.Error, ValueError, TypeError) as error:
            detail = self._mark_storage_failure("acknowledge event", error)
            raise StorageUnavailable(detail) from error
        self._mark_storage_available()
        return found

    def start_worker(self, command: list[str], model: Path, video: Path | str) -> None:
        self.model_name = model.name
        self.video_name = display_source(str(video))
        self.worker_status = "starting"
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self.worker_pid = self.process.pid
        self.worker_status = "running"
        self.reader = threading.Thread(target=self._read_worker, daemon=True)
        self.reader.start()

    def _read_worker(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    print(f"worker emitted invalid JSON: {line.rstrip()}", file=sys.stderr)
                    continue
                if message.get("type") == "event":
                    try:
                        self._get_store().add_event(message)
                    except StorageUnavailable as error:
                        self._mark_storage_failure("write event", error)
                    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError) as error:
                        self._mark_storage_failure("write event", error)
                    else:
                        self._mark_storage_available()
                elif message.get("type") == "source_status":
                    # Keep reconnect state visible between frames without storing every retry.
                    with self.lock:
                        self.metrics.update(message)
                elif message.get("type") == "metric":
                    with self.lock:
                        self.metrics = message
                    try:
                        self._get_store().add_metric(message)
                    except StorageUnavailable as error:
                        self._mark_storage_failure("write metric", error)
                    except (OSError, sqlite3.Error, TypeError, ValueError) as error:
                        self._mark_storage_failure("write metric", error)
                    else:
                        self._mark_storage_available()
        finally:
            code = self.process.wait()
            with self.lock:
                self.worker_exit_code = code
                self.worker_status = "completed" if code == 0 else "error"
            if code != 0:
                self.worker_failure.set()

    def snapshot(self) -> dict[str, Any]:
        event_count: int | None = None
        try:
            event_count = self._get_store().event_count()
        except (StorageUnavailable, OSError, sqlite3.Error, ValueError, TypeError) as error:
            if not isinstance(error, StorageUnavailable):
                self._mark_storage_failure("read event count", error)
        else:
            self._mark_storage_available()
        with self.lock:
            metrics = dict(self.metrics)
            status = self.worker_status
            pid = self.worker_pid
            exit_code = self.worker_exit_code
            storage_status = self.storage_status
            storage_error = self.storage_error
            storage_failures = self.storage_failures
            event_write_failures = self.event_write_failures
            metric_write_failures = self.metric_write_failures
        return {
            "status": status,
            "worker_pid": pid,
            "worker_exit_code": exit_code,
            "model": self.model_name,
            "source": self.video_name,
            "uptime_seconds": round(time.monotonic() - self.started_at, 3),
            "event_count": event_count,
            "storage_status": storage_status,
            "storage_last_error": storage_error,
            "storage_failures": storage_failures,
            "event_write_failures": event_write_failures,
            "metric_write_failures": metric_write_failures,
            "metrics": metrics,
        }

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if self.reader is not None:
            self.reader.join(timeout=2)


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RK-EdgeStation/0.1"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: Any) -> None:
            body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            if parsed.path == "/":
                page = (ROOT / "web" / "index.html").read_bytes()
                self._send(200, page, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/health":
                self._json(200, state.snapshot())
                return
            if parsed.path == "/api/metrics":
                self._json(200, state.snapshot().get("metrics", {}))
                return
            if parsed.path == "/api/events":
                limit = parse_qs(parsed.query).get("limit", ["100"])[0]
                try:
                    events = state.read_events(int(limit))
                except ValueError:
                    self._json(400, {"error": "limit must be an integer"})
                    return
                except StorageUnavailable as error:
                    self._json(503, {"error": "event storage unavailable", "detail": str(error)})
                    return
                self._json(200, {"events": events})
                return
            if parsed.path.startswith("/evidence/"):
                filename = parsed.path.removeprefix("/evidence/")
                candidate = (state.evidence_dir / filename).resolve()
                if Path(filename).name != filename or state.evidence_dir not in candidate.parents:
                    self._json(400, {"error": "invalid evidence path"})
                    return
                if not candidate.is_file():
                    self._json(404, {"error": "evidence not found"})
                    return
                self._send(200, candidate.read_bytes(), "image/jpeg")
                return
            self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            prefix = "/api/events/"
            suffix = "/ack"
            if not parsed.path.startswith(prefix) or not parsed.path.endswith(suffix):
                self._json(404, {"error": "not found"})
                return
            event_id = unquote(parsed.path[len(prefix):-len(suffix)])
            if not event_id or "/" in event_id:
                self._json(400, {"error": "invalid event id"})
                return
            try:
                body_size = int(self.headers.get("Content-Length", "0"))
                if body_size < 0 or body_size > 8192:
                    raise ValueError("request body must be at most 8192 bytes")
                payload = json.loads(self.rfile.read(body_size) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                acknowledged_by = payload.get("acknowledged_by", "operator")
                if not isinstance(acknowledged_by, str):
                    raise ValueError("acknowledged_by must be a string")
                found = state.acknowledge_event(event_id, acknowledged_by)
            except (ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
                return
            except StorageUnavailable as error:
                self._json(503, {"error": "event storage unavailable", "detail": str(error)})
                return
            if not found:
                self._json(404, {"error": "event not found"})
                return
            self._json(200, {"event_id": event_id, "acknowledged": True})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"http {self.address_string()} {fmt % args}", file=sys.stderr)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK-EdgeStation local dashboard")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-family", choices=("yolov8-coco", "yolox-ppe"), default="yolov8-coco")
    parser.add_argument("--video")
    parser.add_argument("--worker", type=Path, default=ROOT / "build" / "bin" / "rk_edge_worker")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "state")
    parser.add_argument("--loop-video", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--queue-capacity", type=int, default=1)
    parser.add_argument("--queue-policy", choices=("latest", "fifo"), default="latest")
    parser.add_argument("--roi", default="0,0,1,1")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--dwell-ms", type=int, default=5000)
    parser.add_argument("--roi-debounce-frames", type=int, default=2)
    parser.add_argument("--danger-zone", help="normalized x,y,width,height for persistent person/forklift entry")
    parser.add_argument("--conflict-zone", help="normalized x,y,width,height for person/forklift co-occurrence")
    parser.add_argument("--ppe-confirm-frames", type=int, default=3)
    parser.add_argument("--ppe-cooldown-ms", type=int, default=30000)
    parser.add_argument("--conflict-dwell-ms", type=int, default=1000)
    return parser.parse_args()


def build_worker_command(args: argparse.Namespace, worker: Path, model: Path,
                         video: Path | str, state_dir: Path) -> list[str]:
    command = [
        str(worker), "--model", str(model), "--video", str(video),
        "--model-family", args.model_family,
        "--state-dir", str(state_dir), "--roi", args.roi,
        "--queue-capacity", str(args.queue_capacity), "--queue-policy", args.queue_policy,
        "--confidence", str(args.confidence), "--dwell-ms", str(args.dwell_ms),
        "--roi-debounce-frames", str(args.roi_debounce_frames),
        "--ppe-confirm-frames", str(args.ppe_confirm_frames),
        "--ppe-cooldown-ms", str(args.ppe_cooldown_ms),
        "--conflict-dwell-ms", str(args.conflict_dwell_ms),
    ]
    if args.danger_zone:
        command.extend(("--danger-zone", args.danger_zone))
    if args.conflict_zone:
        command.extend(("--conflict-zone", args.conflict_zone))
    if args.loop_video:
        command.append("--loop-video")
    return command


def main() -> int:
    args = parse_args()
    if (args.model is None) != (args.video is None):
        print("--model and --video must be supplied together", file=sys.stderr)
        return 2
    if not (1 <= args.port <= 65535):
        print("port must be between 1 and 65535", file=sys.stderr)
        return 2
    if args.dwell_ms <= 0 or not (1 <= args.roi_debounce_frames <= 10000):
        print("dwell-ms must be positive and ROI debounce frames must be 1..10000", file=sys.stderr)
        return 2
    if not (1 <= args.ppe_confirm_frames <= 10000) or args.ppe_cooldown_ms < 0 or args.conflict_dwell_ms < 0:
        print("PPE confirmation, cooldown, and conflict dwell settings are invalid", file=sys.stderr)
        return 2
    for option_name in ("danger_zone", "conflict_zone"):
        value = getattr(args, option_name)
        if value is None:
            continue
        try:
            values = [float(part.strip()) for part in value.split(",")]
            x, y, width, height = values
        except (ValueError, TypeError):
            print(f"--{option_name.replace('_', '-')} must be x,y,width,height", file=sys.stderr)
            return 2
        if (x < 0 or y < 0 or width <= 0 or height <= 0 or
                x + width > 1 or y + height > 1):
            print(f"--{option_name.replace('_', '-')} must lie within normalized frame coordinates", file=sys.stderr)
            return 2

    state_dir = args.state_dir if args.state_dir.is_absolute() else ROOT / args.state_dir
    state_dir = state_dir.resolve()
    evidence_dir = state_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    database = state_dir / "events.sqlite3"
    try:
        store = EventStore(database)
        storage_error = None
    except (OSError, sqlite3.Error) as error:
        store = None
        storage_error = f"{type(error).__name__}: {error}"
    state = AppState(store, evidence_dir, database, storage_error)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    server.daemon_threads = True
    server.timeout = 0.5
    if args.model is not None and args.video is not None:
        model = args.model.resolve()
        video = resolve_worker_source(args.video)
        worker = args.worker.resolve()
        if (not model.is_file() or not worker.is_file() or
                (not is_rtsp_source(video) and not Path(video).is_file())):
            print("model, video, and worker must all exist", file=sys.stderr)
            return 2
        command = build_worker_command(args, worker, model, video, state_dir)
        state.start_worker(command, model, video)

    print(f"RK-EdgeStation UI: http://127.0.0.1:{args.port}/", flush=True)

    def request_shutdown(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    try:
        while not state.worker_failure.is_set():
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.stop()
    return 1 if state.worker_failure.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
