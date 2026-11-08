import io
import json
import signal
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    import resource
except ImportError:  # Windows host tests skip the POSIX file-size-limit probe.
    resource = None

from edge_station.server import AppState, build_worker_command, make_handler, parse_args
from edge_station.storage import EventStore
from edge_station.media_source import display_source, resolve_worker_source


class LocalApiTests(unittest.TestCase):
    def test_loopback_health_events_and_dashboard(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            store = EventStore(state_dir / "events.sqlite3")
            store.add_event(
                {
                    "event_id": "evt-api-1",
                    "timestamp_utc": "2026-11-05T00:00:00.000Z",
                    "event_type": "leave",
                    "frame_sequence": 1,
                    "track_id": 3,
                    "label": "person",
                    "confidence": 0.88,
                    "state": "closed",
                    "transition_reason": "roi_exit",
                    "roi_occupancy_count": 0,
                    "cumulative_roi_entries": 2,
                    "evidence": "evidence/frame-1.jpg",
                }
            )
            app_state = AppState(store, state_dir / "evidence")
            app_state.camera_statuses["cam-a"] = {
                "camera_id": "cam-a", "source_status": "connected",
                "source": "rtsp://camera-a:554",
            }
            app_state.camera_metrics["cam-a"] = {
                "camera_id": "cam-a", "status": "running", "throughput_fps": 2.5,
                "end_to_end_p95_ms": 400.0,
            }
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), make_handler(app_state)
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/health", timeout=3) as response:
                    health = json.load(response)
                self.assertEqual(health["status"], "idle")
                self.assertEqual(health["event_count"], 1)
                self.assertEqual(health["cameras"][0]["camera_id"], "cam-a")
                self.assertEqual(health["metrics"]["throughput_fps"], 2.5)

                with urlopen(base + "/api/cameras", timeout=3) as response:
                    cameras = json.load(response)["cameras"]
                self.assertEqual(cameras[0]["source_status"], "connected")
                with urlopen(base + "/api/algorithms", timeout=3) as response:
                    algorithms = json.load(response)["algorithms"]
                self.assertEqual(next(a for a in algorithms if a["algorithm_id"] == "fire_smoke")["status"],
                                 "not_ready")

                with urlopen(base + "/api/events?limit=5", timeout=3) as response:
                    events = json.load(response)["events"]
                self.assertEqual(events[0]["track_id"], 3)
                self.assertEqual(events[0]["event_type"], "leave")
                self.assertEqual(events[0]["state"], "closed")
                self.assertEqual(events[0]["transition_reason"], "roi_exit")
                self.assertEqual(events[0]["roi_occupancy_count"], 0)
                self.assertEqual(events[0]["cumulative_roi_entries"], 2)

                request = Request(
                    base + "/api/events/evt-api-1/ack",
                    data=json.dumps({"acknowledged_by": "reviewer-1"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=3) as response:
                    self.assertTrue(json.load(response)["acknowledged"])
                with urlopen(base + "/api/events?limit=5", timeout=3) as response:
                    acknowledged = json.load(response)["events"][0]
                self.assertEqual(acknowledged["acknowledged_by"], "reviewer-1")
                self.assertTrue(acknowledged["acknowledged_at"])

                with urlopen(base + "/", timeout=3) as response:
                    page = response.read().decode("utf-8")
                self.assertIn("RK-EdgeStation", page)
                self.assertIn("ROI occupancy", page)
                self.assertIn("reconnect_attempts", page)
                self.assertIn("source_status", page)
                self.assertIn("event.state", page)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


class CameraConfigApiTests(unittest.TestCase):
    def test_camera_toggle_is_persisted_and_source_credentials_are_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "cameras.json"
            config_path.write_text(json.dumps({"version": 1, "cameras": [
                {"id": "cam-a", "source": "rtsp://user:secret@camera-a:554/live", "enabled": True,
                 "tasks": ["ppe", "roi"]},
                {"id": "cam-b", "source": "rtsp://camera-b:554/live", "enabled": True,
                 "tasks": ["ppe"]},
            ]}), encoding="utf-8")
            state = AppState(EventStore(root / "events.sqlite3"), root / "evidence",
                             camera_config_path=config_path)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                request = Request(base + "/api/cameras/cam-a/enabled",
                                  data=json.dumps({"enabled": False}).encode("utf-8"),
                                  headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=3) as response:
                    self.assertFalse(json.load(response)["enabled"])
                stored = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertFalse(stored["cameras"][0]["enabled"])

                with urlopen(base + "/api/cameras", timeout=3) as response:
                    cameras = json.load(response)["cameras"]
                cam_a = next(camera for camera in cameras if camera["camera_id"] == "cam-a")
                self.assertEqual(cam_a["source"], "rtsp://camera-a")
                self.assertEqual(cam_a["source_status"], "disabled")
                self.assertNotIn("secret", json.dumps(cameras))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


class ServerQueuePolicyTests(unittest.TestCase):
    def test_default_and_fifo_policy_reach_worker_command(self):
        with patch("sys.argv", ["edge_station.server"]):
            default_args = parse_args()
        self.assertEqual(default_args.queue_policy, "latest")
        self.assertEqual(default_args.video_backend, "opencv")
        self.assertEqual(default_args.preprocess_backend, "cpu")
        self.assertEqual(default_args.input_pacing, "media")
        self.assertFalse(default_args.loop_video)
        self.assertIsNone(default_args.camera_config)

        with patch("sys.argv", ["edge_station.server", "--queue-policy", "fifo"]):
            fifo_args = parse_args()
        command = build_worker_command(
            fifo_args,
            Path("worker"),
            Path("model.rknn"),
            Path("video.mp4"),
            Path("state"),
        )
        self.assertEqual(command[command.index("--queue-policy") + 1], "fifo")

        with patch("sys.argv", ["edge_station.server", "--loop-video"]):
            loop_args = parse_args()
        loop_command = build_worker_command(
            loop_args,
            Path("worker"),
            Path("model.rknn"),
            Path("video.mp4"),
            Path("state"),
        )
        self.assertIn("--loop-video", loop_command)

    def test_video_and_preprocess_backends_reach_worker_command(self):
        with patch("sys.argv", ["edge_station.server", "--video-backend", "gstreamer-mpp",
                                "--preprocess-backend", "rga", "--input-pacing", "none"]):
            args = parse_args()
        command = build_worker_command(
            args, Path("worker"), Path("model.rknn"), Path("video.mp4"), Path("state"))
        self.assertEqual(command[command.index("--video-backend") + 1], "gstreamer-mpp")
        self.assertEqual(command[command.index("--preprocess-backend") + 1], "rga")
        self.assertEqual(command[command.index("--input-pacing") + 1], "none")

    def test_rtsp_source_preserves_uri_and_hides_credentials(self):
        source = "rtsp://u:p@camera.example:8554/live?x=1"
        with patch("sys.argv", ["edge_station.server", "--video", source]):
            args = parse_args()
        self.assertEqual(args.video, source)
        self.assertEqual(resolve_worker_source(args.video), source)
        self.assertEqual(display_source(args.video), "rtsp://camera.example:8554")

        command = build_worker_command(
            args,
            Path("worker"),
            Path("model.rknn"),
            args.video,
            Path("state"),
        )
        self.assertEqual(command[command.index("--video") + 1], source)

    def test_camera_config_reaches_single_process_worker_command(self):
        with patch("sys.argv", ["edge_station.server", "--camera-config", "deploy/cameras.json",
                                "--model-family", "constructelligence-ppe"]):
            args = parse_args()
        command = build_worker_command(
            args, Path("worker"), Path("model.rknn"), None, Path("state"))
        self.assertEqual(command[command.index("--camera-config") + 1], str(Path("deploy/cameras.json")))
        self.assertNotIn("--video", command)

    def test_invalid_rtsp_uri_requires_host(self):
        with self.assertRaises(ValueError):
            resolve_worker_source("rtsp:///stream")


class SourceStatusTests(unittest.TestCase):
    def test_reconnect_state_updates_health_without_persisting_each_retry(self):
        class WorkerOutput:
            def __init__(self):
                messages = [
                    {"type": "metric", "status": "running", "processed_frames": 10},
                    {"type": "source_status", "source_type": "rtsp",
                     "source_status": "reconnecting", "reconnect_attempts": 3,
                     "stream_epoch": 0},
                ]
                self.stdout = io.StringIO("".join(json.dumps(message) + "\n" for message in messages))

            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            store = EventStore(state_dir / "events.sqlite3")
            state = AppState(store, state_dir / "evidence")
            state.process = WorkerOutput()
            state._read_worker()

            metrics = state.snapshot()["metrics"]
            self.assertEqual(metrics["processed_frames"], 10)
            self.assertEqual(metrics["source_status"], "reconnecting")
            self.assertEqual(metrics["reconnect_attempts"], 3)
            self.assertEqual(metrics["stream_epoch"], 0)
            self.assertEqual(store.latest_metric()["processed_frames"], 10)
            self.assertFalse(state.worker_failure.is_set())

    def test_worker_failure_is_signaled_for_service_manager_restart(self):
        class FailedWorker:
            stdout = io.StringIO("")

            def wait(self):
                return 2

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            state = AppState(EventStore(state_dir / "events.sqlite3"), state_dir / "evidence")
            state.process = FailedWorker()
            state._read_worker()
            self.assertTrue(state.worker_failure.is_set())
            self.assertEqual(state.worker_status, "error")


class StorageRecoveryTests(unittest.TestCase):
    def test_unavailable_sqlite_is_reported_and_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            database = state_dir / "events.sqlite3"
            database.mkdir()
            state = AppState(
                None,
                state_dir / "evidence",
                database,
                "OperationalError: simulated unavailable database",
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/health", timeout=3) as response:
                    health = json.load(response)
                self.assertEqual(health["storage_status"], "unavailable")
                self.assertIsNone(health["event_count"])

                with self.assertRaises(HTTPError) as unavailable:
                    urlopen(base + "/api/events", timeout=3)
                self.assertEqual(unavailable.exception.code, 503)
                self.assertEqual(json.load(unavailable.exception)["error"], "event storage unavailable")

                database.rmdir()
                with urlopen(base + "/api/events", timeout=3) as response:
                    events = json.load(response)
                self.assertEqual(events["events"], [])

                with urlopen(base + "/api/health", timeout=3) as response:
                    recovered = json.load(response)
                self.assertEqual(recovered["storage_status"], "available")
                self.assertEqual(recovered["event_count"], 0)
                self.assertIsNotNone(recovered["storage_last_error"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

    @unittest.skipIf(
        resource is None or not hasattr(signal, "SIGXFSZ"),
        "requires POSIX RLIMIT_FSIZE and SIGXFSZ",
    )
    def test_sqlite_write_failure_is_counted_and_recovers(self):
        class WorkerOutput:
            def __init__(self, event):
                self.stdout = io.StringIO(json.dumps({"type": "event", **event}) + "\n")

            def wait(self):
                return 0

        def event(sequence, evidence):
            return {
                "event_id": f"evt-limit-{sequence}",
                "timestamp_utc": "2026-11-05T00:00:00.000Z",
                "event_type": "enter",
                "frame_sequence": sequence,
                "track_id": 1,
                "label": "person",
                "confidence": 0.9,
                "evidence": evidence,
            }

        with tempfile.TemporaryDirectory() as directory:
            state_dir = Path(directory)
            database = state_dir / "events.sqlite3"
            state = AppState(EventStore(database), state_dir / "evidence")
            original_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
            original_handler = signal.getsignal(signal.SIGXFSZ)
            try:
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
                resource.setrlimit(
                    resource.RLIMIT_FSIZE,
                    (database.stat().st_size, original_limit[1]),
                )
                state.process = WorkerOutput(event(1, "x" * 262144))
                state._read_worker()
                with state.lock:
                    self.assertEqual(state.storage_status, "unavailable")
                    self.assertEqual(state.event_write_failures, 1)
            finally:
                resource.setrlimit(resource.RLIMIT_FSIZE, original_limit)
                signal.signal(signal.SIGXFSZ, original_handler)

            state.process = WorkerOutput(event(2, "frame-2.jpg"))
            state._read_worker()
            recovered = state.snapshot()
            self.assertEqual(recovered["storage_status"], "available")
            self.assertEqual(recovered["event_count"], 1)
            self.assertEqual(recovered["event_write_failures"], 1)
            self.assertIn("write event", recovered["storage_last_error"])


if __name__ == "__main__":
    unittest.main()
