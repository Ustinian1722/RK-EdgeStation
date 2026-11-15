from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class EventStore:
    def __init__(self, database: Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    camera_id TEXT NOT NULL DEFAULT 'camera-0',
                    algorithm_id TEXT NOT NULL DEFAULT 'unknown',
                    frame_sequence INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp_utc DESC);
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    payload_json TEXT NOT NULL
                );
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(events)")}
            if "camera_id" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN camera_id TEXT NOT NULL DEFAULT 'camera-0'")
            if "algorithm_id" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN algorithm_id TEXT NOT NULL DEFAULT 'unknown'")
            if "acknowledged_at" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN acknowledged_at TEXT")
            if "acknowledged_by" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN acknowledged_by TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events(camera_id, timestamp_utc DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_camera_algorithm_time "
                "ON events(camera_id, algorithm_id, timestamp_utc DESC)"
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add_event(self, event: dict[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO events
                   (event_id, timestamp_utc, event_type, camera_id, algorithm_id,
                    frame_sequence, track_id, label, confidence, evidence, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["event_id"],
                    event["timestamp_utc"],
                    event["event_type"],
                    str(event.get("camera_id", "camera-0")),
                    str(event.get("algorithm_id", "unknown")),
                    int(event["frame_sequence"]),
                    int(event["track_id"]),
                    event["label"],
                    float(event["confidence"]),
                    event["evidence"],
                    json.dumps(event, separators=(",", ":"), ensure_ascii=False),
                ),
            )

    def add_metric(self, metric: dict[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO metrics(payload_json) VALUES (?)",
                (json.dumps(metric, separators=(",", ":")),),
            )

    def events(
        self,
        limit: int = 100,
        camera_id: str | None = None,
        algorithm_id: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        filters: list[str] = []
        parameters: list[Any] = []
        if camera_id:
            filters.append("camera_id = ?")
            parameters.append(camera_id)
        if algorithm_id:
            filters.append("algorithm_id = ?")
            parameters.append(algorithm_id)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload_json, acknowledged_at, acknowledged_by FROM events "
                + where + " ORDER BY timestamp_utc DESC LIMIT ?", parameters
            ).fetchall()
        events = []
        for row in rows:
            event = json.loads(row["payload_json"])
            if row["acknowledged_at"] is not None:
                event["acknowledged_at"] = row["acknowledged_at"]
                event["acknowledged_by"] = row["acknowledged_by"]
            events.append(event)
        return events

    def acknowledge_event(self, event_id: str, acknowledged_by: str = "operator") -> bool:
        event_id = str(event_id).strip()
        acknowledged_by = str(acknowledged_by).strip()
        if not event_id or len(event_id) > 200:
            raise ValueError("event_id must contain 1..200 characters")
        if not acknowledged_by or len(acknowledged_by) > 128:
            raise ValueError("acknowledged_by must contain 1..128 characters")
        with self._connection() as connection:
            cursor = connection.execute(
                """UPDATE events
                   SET acknowledged_at = COALESCE(acknowledged_at,
                                                  strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                       acknowledged_by = COALESCE(acknowledged_by, ?)
                   WHERE event_id = ?""",
                (acknowledged_by, event_id),
            )
            return cursor.rowcount > 0

    def latest_metric(self) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM metrics ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def event_count(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
