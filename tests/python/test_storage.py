import tempfile
import unittest
from pathlib import Path

from edge_station.storage import EventStore


class EventStoreTests(unittest.TestCase):
    def test_event_and_metric_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EventStore(Path(directory) / "run" / "events.sqlite3")
            event = {
                "event_id": "evt-test-1",
                "timestamp_utc": "2026-11-05T00:00:00.000Z",
                "event_type": "enter",
                "camera_id": "line-a-cam-1",
                "algorithm_id": "roi",
                "frame_sequence": 4,
                "track_id": 2,
                "label": "person",
                "confidence": 0.91,
                "state": "active",
                "transition_reason": "roi_enter",
                "roi_occupancy_count": 1,
                "cumulative_roi_entries": 1,
                "evidence": "state/evidence/frame-4.jpg",
            }
            store.add_event(event)
            store.add_event(event)
            store.add_metric({"status": "completed", "processed_frames": 4})

            self.assertEqual(store.event_count(), 1)
            self.assertEqual(store.events(10), [event])
            self.assertEqual(store.events(10, camera_id="line-a-cam-1", algorithm_id="roi"), [event])
            self.assertEqual(store.events(10, camera_id="missing"), [])
            self.assertEqual(store.latest_metric()["processed_frames"], 4)


if __name__ == "__main__":
    unittest.main()
