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
from urllib.request import urlopen

try:
    import resource
except ImportError:  # Windows host tests skip the POSIX file-size-limit probe.
    resource = None

from edge_station.server import AppState, build_worker_command, make_handler, parse_args
from edge_station.storage import EventStore


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
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), make_handler(AppState(store, state_dir / "evidence"))
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/health", timeout=3) as response:
                    health = json.load(response)
                self.assertEqual(health["status"], "idle")
                self.assertEqual(health["event_count"], 1)

                with urlopen(base + "/api/events?limit=5", timeout=3) as response:
                    events = json.load(response)["events"]
                self.assertEqual(events[0]["track_id"], 3)
                self.assertEqual(events[0]["event_type"], "leave")
                self.assertEqual(events[0]["state"], "closed")
                self.assertEqual(events[0]["transition_reason"], "roi_exit")
                self.assertEqual(events[0]["roi_occupancy_count"], 0)
                self.assertEqual(events[0]["cumulative_roi_entries"], 2)

                with urlopen(base + "/", timeout=3) as response:
                    page = response.read().decode("utf-8")
                self.assertIn("RK-EdgeStation", page)
                self.assertIn("ROI occupancy", page)
                self.assertIn("event.state", page)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


class ServerQueuePolicyTests(unittest.TestCase):
    def test_default_and_fifo_policy_reach_worker_command(self):
        with patch("sys.argv", ["edge_station.server"]):
            default_args = parse_args()
        self.assertEqual(default_args.queue_policy, "latest")
        self.assertFalse(default_args.loop_video)

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
