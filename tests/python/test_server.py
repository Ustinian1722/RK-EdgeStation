import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

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


if __name__ == "__main__":
    unittest.main()
