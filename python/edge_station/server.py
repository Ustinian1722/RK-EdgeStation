from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .storage import EventStore


ROOT = Path(__file__).resolve().parents[2]


class AppState:
    def __init__(self, store: EventStore, evidence_dir: Path):
        self.store = store
        self.evidence_dir = evidence_dir.resolve()
        self.started_at = time.monotonic()
        self.lock = threading.Lock()
        self.worker_status = "idle"
        self.worker_pid: int | None = None
        self.worker_exit_code: int | None = None
        self.metrics: dict[str, Any] = {}
        self.model_name = ""
        self.video_name = ""
        self.process: subprocess.Popen[str] | None = None
        self.reader: threading.Thread | None = None

    def start_worker(self, command: list[str], model: Path, video: Path) -> None:
        self.model_name = model.name
        self.video_name = video.name
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
                        self.store.add_event(message)
                    except (KeyError, TypeError, ValueError) as error:
                        print(f"event persistence failed: {error}", file=sys.stderr)
                elif message.get("type") == "metric":
                    with self.lock:
                        self.metrics = message
                    try:
                        self.store.add_metric(message)
                    except Exception as error:
                        print(f"metric persistence failed: {error}", file=sys.stderr)
        finally:
            code = self.process.wait()
            with self.lock:
                self.worker_exit_code = code
                self.worker_status = "completed" if code == 0 else "error"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            metrics = dict(self.metrics)
            status = self.worker_status
            pid = self.worker_pid
            exit_code = self.worker_exit_code
        return {
            "status": status,
            "worker_pid": pid,
            "worker_exit_code": exit_code,
            "model": self.model_name,
            "source": self.video_name,
            "uptime_seconds": round(time.monotonic() - self.started_at, 3),
            "event_count": self.store.event_count(),
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
                    events = state.store.events(int(limit))
                except ValueError:
                    self._json(400, {"error": "limit must be an integer"})
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

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"http {self.address_string()} {fmt % args}", file=sys.stderr)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK-EdgeStation local dashboard")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--worker", type=Path, default=ROOT / "build" / "bin" / "rk_edge_worker")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "state")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--queue-capacity", type=int, default=1)
    parser.add_argument("--queue-policy", choices=("latest", "fifo"), default="latest")
    parser.add_argument("--roi", default="0,0,1,1")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--dwell-ms", type=int, default=5000)
    parser.add_argument("--roi-debounce-frames", type=int, default=2)
    return parser.parse_args()


def build_worker_command(args: argparse.Namespace, worker: Path, model: Path,
                         video: Path, state_dir: Path) -> list[str]:
    return [
        str(worker), "--model", str(model), "--video", str(video),
        "--state-dir", str(state_dir), "--roi", args.roi,
        "--queue-capacity", str(args.queue_capacity), "--queue-policy", args.queue_policy,
        "--confidence", str(args.confidence), "--dwell-ms", str(args.dwell_ms),
        "--roi-debounce-frames", str(args.roi_debounce_frames),
    ]


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

    state_dir = args.state_dir if args.state_dir.is_absolute() else ROOT / args.state_dir
    state_dir = state_dir.resolve()
    evidence_dir = state_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    state = AppState(EventStore(state_dir / "events.sqlite3"), evidence_dir)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    server.daemon_threads = True
    if args.model is not None and args.video is not None:
        model = args.model.resolve()
        video = args.video.resolve()
        worker = args.worker.resolve()
        if not model.is_file() or not video.is_file() or not worker.is_file():
            print("model, video, and worker must all exist", file=sys.stderr)
            return 2
        command = build_worker_command(args, worker, model, video, state_dir)
        state.start_worker(command, model, video)

    print(f"RK-EdgeStation UI: http://127.0.0.1:{args.port}/", flush=True)

    def request_shutdown(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
