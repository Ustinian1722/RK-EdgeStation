from __future__ import annotations

import argparse
import json
import math
import os
import re
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
        camera_config_path: Path | None = None,
    ):
        self.store = store
        self.store_path = Path(store_path) if store_path is not None else None
        self.evidence_dir = evidence_dir.resolve()
        self.started_at = time.monotonic()
        self.lock = threading.Lock()
        self.storage_init_lock = threading.Lock()
        self.camera_config_path = Path(camera_config_path).resolve() if camera_config_path else None
        self.camera_config_lock = threading.RLock()
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
        self.camera_metrics: dict[str, dict[str, Any]] = {}
        self.camera_statuses: dict[str, dict[str, Any]] = {}
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

    def read_events(
        self, limit: int, camera_id: str | None = None, algorithm_id: str | None = None
    ) -> list[dict[str, Any]]:
        try:
            events = self._get_store().events(limit, camera_id=camera_id, algorithm_id=algorithm_id)
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

    def start_worker(self, command: list[str], model: Path, video: Path | str | None) -> None:
        self.model_name = model.name
        self.video_name = display_source(str(video)) if video is not None else "configured cameras"
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
                elif message.get("type") == "camera_status":
                    camera_id = str(message.get("camera_id", ""))
                    if camera_id:
                        if isinstance(message.get("source"), str):
                            message["source"] = display_source(message["source"])
                        with self.lock:
                            self.camera_statuses[camera_id] = dict(message)
                elif message.get("type") == "camera_metric":
                    camera_id = str(message.get("camera_id", ""))
                    if camera_id:
                        if isinstance(message.get("source"), str):
                            message["source"] = display_source(message["source"])
                        with self.lock:
                            self.camera_metrics[camera_id] = dict(message)
                            self.metrics["camera_count"] = len(self.camera_metrics)
                            self.metrics["throughput_fps"] = sum(
                                float(item.get("throughput_fps", 0.0))
                                for item in self.camera_metrics.values()
                            )
                            self.metrics["processed_frames"] = sum(
                                int(item.get("processed_frames", 0))
                                for item in self.camera_metrics.values()
                            )
                            self.metrics["decoded_frames"] = sum(
                                int(item.get("decoded_frames", 0))
                                for item in self.camera_metrics.values()
                            )
                            self.metrics["dropped_frames"] = sum(
                                int(item.get("dropped_frames", 0))
                                for item in self.camera_metrics.values()
                            )
                            self.metrics["p95_end_to_end_ms"] = max(
                                (float(item.get("end_to_end_p95_ms", 0.0))
                                 for item in self.camera_metrics.values()), default=0.0
                            )
                        try:
                            self._get_store().add_metric(message)
                        except StorageUnavailable as error:
                            self._mark_storage_failure("write camera metric", error)
                        except (OSError, sqlite3.Error, TypeError, ValueError) as error:
                            self._mark_storage_failure("write camera metric", error)
                        else:
                            self._mark_storage_available()
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
            camera_ids = set(self.camera_statuses) | set(self.camera_metrics)
            cameras = {}
            for camera_id in camera_ids:
                camera = dict(self.camera_statuses.get(camera_id, {}))
                camera.update(self.camera_metrics.get(camera_id, {}))
                cameras[camera_id] = camera
            camera_metrics = list(self.camera_metrics.values())
            if camera_metrics:
                metrics["camera_count"] = len(camera_metrics)
                for field in ("processed_frames", "decoded_frames", "dropped_frames"):
                    metrics[field] = sum(int(item.get(field, 0)) for item in camera_metrics)
                metrics["throughput_fps"] = sum(
                    float(item.get("throughput_fps", 0.0)) for item in camera_metrics
                )
                metrics["p95_end_to_end_ms"] = max(
                    (float(item.get("end_to_end_p95_ms", 0.0)) for item in camera_metrics),
                    default=0.0,
                )
            status = self.worker_status
            pid = self.worker_pid
            exit_code = self.worker_exit_code
            storage_status = self.storage_status
            storage_error = self.storage_error
            storage_failures = self.storage_failures
            event_write_failures = self.event_write_failures
            metric_write_failures = self.metric_write_failures
        if self.camera_config_path is not None:
            try:
                config = json.loads(self.camera_config_path.read_text(encoding="utf-8"))
                entries = config.get("cameras", []) if isinstance(config, dict) else []
                for entry in entries:
                    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                        continue
                    camera_id = entry["id"]
                    configured = {
                        "camera_id": camera_id,
                        "source": display_source(str(entry.get("source", ""))),
                        "enabled": bool(entry.get("enabled", True)),
                        "tasks": entry.get("tasks", ["ppe", "roi"]),
                        "weight": entry.get("weight", 1),
                        "max_fps": entry.get("max_fps", 0),
                    }
                    configured.update(cameras.get(camera_id, {}))
                    if not configured.get("source_status"):
                        configured["source_status"] = "connecting" if configured["enabled"] else "disabled"
                    cameras[camera_id] = configured
            except (OSError, json.JSONDecodeError, TypeError):
                pass
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
            "camera_config": str(self.camera_config_path) if self.camera_config_path else None,
            "cameras": [cameras[key] for key in sorted(cameras)],
        }

    def update_camera_config(self, config: dict[str, Any]) -> int:
        path = self.camera_config_path
        if path is None:
            raise RuntimeError("camera configuration is not enabled for this service")
        cameras = config.get("cameras") if isinstance(config, dict) else None
        if not isinstance(cameras, list) or not (1 <= len(cameras) <= 32):
            raise ValueError("camera config must contain 1..32 camera entries")
        identifiers: set[str] = set()
        enabled_count = 0
        for camera in cameras:
            if not isinstance(camera, dict):
                raise ValueError("each camera must be a JSON object")
            camera_id = camera.get("id")
            source = camera.get("source")
            if (not isinstance(camera_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", camera_id)
                    or camera_id in (".", "..")):
                raise ValueError("camera id must contain 1..64 safe ASCII characters")
            if camera_id in identifiers:
                raise ValueError("camera ids must be unique")
            identifiers.add(camera_id)
            if not isinstance(source, str) or not source or len(source) > 2048:
                raise ValueError("camera source must contain 1..2048 characters")
            enabled = camera.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError("camera enabled must be a boolean")
            enabled_count += int(enabled)
            tasks = camera.get("tasks", ["ppe", "roi"])
            if (not isinstance(tasks, list) or not tasks or
                    any(task not in ("ppe", "roi") for task in tasks) or
                    len(set(tasks)) != len(tasks)):
                raise ValueError("camera tasks must contain ppe and/or roi")
            max_fps = camera.get("max_fps", 0)
            if isinstance(max_fps, bool) or not isinstance(max_fps, (int, float)) or not math.isfinite(max_fps) or not 0 <= max_fps <= 120:
                raise ValueError("max_fps must be between 0 and 120")
            weight = camera.get("weight", 1)
            if isinstance(weight, bool) or not isinstance(weight, int) or not 1 <= weight <= 64:
                raise ValueError("weight must be an integer between 1 and 64")
            capacity = camera.get("queue_capacity", 1)
            if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 8:
                raise ValueError("queue_capacity must be an integer between 1 and 8")
            roi = camera.get("roi")
            if roi is not None:
                if (not isinstance(roi, list) or len(roi) != 4 or
                        any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                            not math.isfinite(value) for value in roi)):
                    raise ValueError("roi must contain four finite numbers")
                x, y, width, height = roi
                if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
                    raise ValueError("roi must be a normalized rectangle inside the frame")
        if enabled_count == 0:
            raise ValueError("at least one camera must be enabled")
        payload = json.dumps(config, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.camera_config_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            try:
                temporary.write_text(payload, encoding="utf-8")
                os.chmod(temporary, 0o600)
                os.replace(temporary, path)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            process = self.process
            if process is not None and process.poll() is None:
                reload_signal = getattr(signal, "SIGUSR1", None)
                if reload_signal is None:
                    raise RuntimeError("this platform cannot signal the worker to reload configuration")
                process.send_signal(reload_signal)
        return len(cameras)

    def set_camera_enabled(self, camera_id: str, enabled: bool) -> int:
        path = self.camera_config_path
        if path is None:
            raise RuntimeError("camera configuration is not enabled for this service")
        with self.camera_config_lock:
            try:
                config = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError("camera configuration could not be read") from error
            cameras = config.get("cameras") if isinstance(config, dict) else None
            if not isinstance(cameras, list):
                raise RuntimeError("camera configuration is invalid")
            for camera in cameras:
                if isinstance(camera, dict) and camera.get("id") == camera_id:
                    camera["enabled"] = enabled
                    return self.update_camera_config(config)
            raise KeyError(camera_id)

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
            if parsed.path == "/api/cameras":
                self._json(200, {"cameras": state.snapshot().get("cameras", [])})
                return
            if parsed.path == "/api/algorithms":
                self._json(200, {"algorithms": [
                    {"algorithm_id": "roi", "status": "ready", "model_required": False},
                    {"algorithm_id": "ppe", "status": "ready", "model_required": True},
                    {"algorithm_id": "fire_smoke", "status": "not_ready",
                     "reason": "no license- and accuracy-validated RKNN model is integrated"},
                ]})
                return
            if parsed.path == "/api/events":
                query = parse_qs(parsed.query)
                limit = query.get("limit", ["100"])[0]
                camera_id = query.get("camera_id", [None])[0]
                algorithm_id = query.get("algorithm_id", [None])[0]
                if camera_id is not None and (not camera_id or len(camera_id) > 64):
                    self._json(400, {"error": "camera_id must contain 1..64 characters"})
                    return
                if algorithm_id is not None and (not algorithm_id or len(algorithm_id) > 64):
                    self._json(400, {"error": "algorithm_id must contain 1..64 characters"})
                    return
                try:
                    events = state.read_events(int(limit), camera_id, algorithm_id)
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
            camera_prefix = "/api/cameras/"
            camera_suffix = "/enabled"
            if parsed.path.startswith(camera_prefix) and parsed.path.endswith(camera_suffix):
                camera_id = unquote(parsed.path[len(camera_prefix):-len(camera_suffix)])
                if not camera_id or "/" in camera_id:
                    self._json(400, {"error": "invalid camera id"})
                    return
                try:
                    body_size = int(self.headers.get("Content-Length", "0"))
                    if body_size < 0 or body_size > 8192:
                        raise ValueError("request body must be at most 8192 bytes")
                    payload = json.loads(self.rfile.read(body_size) or b"{}")
                    if not isinstance(payload, dict) or not isinstance(payload.get("enabled"), bool):
                        raise ValueError("enabled must be a boolean")
                    count = state.set_camera_enabled(camera_id, payload["enabled"])
                except KeyError:
                    self._json(404, {"error": "camera not found"})
                    return
                except (ValueError, json.JSONDecodeError) as error:
                    self._json(400, {"error": str(error)})
                    return
                except RuntimeError as error:
                    self._json(409, {"error": str(error)})
                    return
                self._json(200, {"camera_id": camera_id, "enabled": payload["enabled"],
                                 "configured_camera_count": count, "reload_requested": True})
                return
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

        def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            if parsed.path != "/api/cameras/config":
                self._json(404, {"error": "not found"})
                return
            try:
                body_size = int(self.headers.get("Content-Length", "0"))
                if body_size < 0 or body_size > 65536:
                    raise ValueError("request body must be at most 65536 bytes")
                payload = json.loads(self.rfile.read(body_size) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("camera config must be a JSON object")
                count = state.update_camera_config(payload)
            except (ValueError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)})
                return
            except RuntimeError as error:
                self._json(409, {"error": str(error)})
                return
            self._json(200, {"configured_camera_count": count, "reload_requested": True})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"http {self.address_string()} {fmt % args}", file=sys.stderr)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK-EdgeStation local dashboard")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-family", choices=("yolov8-coco", "yolox-ppe", "constructelligence-ppe"), default="yolov8-coco")
    parser.add_argument("--video")
    parser.add_argument("--camera-config", type=Path,
                        help="local JSON camera list for the single-process shared inference worker")
    parser.add_argument("--video-backend", choices=("opencv", "gstreamer-mpp"), default="opencv")
    parser.add_argument("--preprocess-backend", choices=("cpu", "rga"), default="cpu")
    parser.add_argument("--input-pacing", choices=("media", "none"), default="media")
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
                         video: Path | str | None, state_dir: Path) -> list[str]:
    command = [
        str(worker), "--model", str(model),
        "--model-family", args.model_family,
        "--video-backend", args.video_backend,
        "--preprocess-backend", args.preprocess_backend,
        "--input-pacing", args.input_pacing,
        "--state-dir", str(state_dir), "--roi", args.roi,
        "--queue-capacity", str(args.queue_capacity), "--queue-policy", args.queue_policy,
        "--confidence", str(args.confidence), "--dwell-ms", str(args.dwell_ms),
        "--roi-debounce-frames", str(args.roi_debounce_frames),
        "--ppe-confirm-frames", str(args.ppe_confirm_frames),
        "--ppe-cooldown-ms", str(args.ppe_cooldown_ms),
        "--conflict-dwell-ms", str(args.conflict_dwell_ms),
    ]
    if args.camera_config is not None:
        command.extend(("--camera-config", str(args.camera_config)))
    elif video is not None:
        command.extend(("--video", str(video)))
    else:
        raise ValueError("either a video source or camera config is required")
    if args.danger_zone:
        command.extend(("--danger-zone", args.danger_zone))
    if args.conflict_zone:
        command.extend(("--conflict-zone", args.conflict_zone))
    if args.loop_video:
        command.append("--loop-video")
    return command


def main() -> int:
    args = parse_args()
    single_source = args.video is not None
    multi_source = args.camera_config is not None
    if (args.model is None and (single_source or multi_source)) or (args.model is not None and single_source == multi_source):
        print("--model must be supplied with exactly one of --video or --camera-config", file=sys.stderr)
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
    camera_config = args.camera_config.resolve() if args.camera_config is not None else None
    if camera_config is not None and not camera_config.is_file():
        print("camera config file must exist", file=sys.stderr)
        return 2
    state = AppState(store, evidence_dir, database, storage_error, camera_config)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    server.daemon_threads = True
    server.timeout = 0.5
    if args.model is not None:
        model = args.model.resolve()
        video = resolve_worker_source(args.video) if single_source else None
        worker = args.worker.resolve()
        if (not model.is_file() or not worker.is_file() or
                (single_source and not is_rtsp_source(str(video)) and not Path(str(video)).is_file())):
            print("model, selected video source, and worker must all exist", file=sys.stderr)
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
