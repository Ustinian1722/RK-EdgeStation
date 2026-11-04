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
                    frame_sequence INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp_utc DESC);
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    payload_json TEXT NOT NULL
                );
                """
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
                   (event_id, timestamp_utc, event_type, frame_sequence, track_id,
                    label, confidence, evidence, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["event_id"],
                    event["timestamp_utc"],
                    event["event_type"],
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

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM events ORDER BY timestamp_utc DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def latest_metric(self) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM metrics ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def event_count(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
