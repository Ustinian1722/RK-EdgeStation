import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

from edge_station.server import AppState, make_handler
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
                    "event_type": "enter",
                    "frame_sequence": 1,
                    "track_id": 3,
                    "label": "person",
                    "confidence": 0.88,
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

                with urlopen(base + "/", timeout=3) as response:
                    page = response.read().decode("utf-8")
                self.assertIn("RK-EdgeStation", page)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
