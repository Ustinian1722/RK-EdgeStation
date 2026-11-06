#!/usr/bin/env python3
"""Summarize Worker event JSONL and optional per-frame tracker JSONL."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


PPE_LABELS = {
    0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
    5: "Person", 6: "no_helmet", 7: "no_gloves", 8: "no_boots",
    9: "no_goggles", 10: "forklift", 11: "no_vest",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.lstrip().startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def summarize(stdout_path: Path, track_path: Path | None) -> dict[str, Any]:
    output = read_jsonl(stdout_path)
    metrics = [row for row in output if row.get("type") == "metric"]
    events = [row for row in output if row.get("type") == "event"]
    final = next((row for row in reversed(metrics) if row.get("status") in {"completed", "error"}), {})
    event_types = collections.Counter(row.get("event_type", "unknown") for row in events)
    event_reasons = collections.Counter(row.get("transition_reason", "unknown") for row in events)
    event_labels = collections.Counter(row.get("label", "unknown") for row in events)
    ppe_events = [row for row in events if row.get("event_type") == "ppe_violation"]
    result: dict[str, Any] = {
        "final_metric": final,
        "event_records": len(events),
        "events_by_type": dict(sorted(event_types.items())),
        "events_by_reason": dict(sorted(event_reasons.items())),
        "events_by_label": dict(sorted(event_labels.items())),
        "ppe_violation_events": ppe_events,
    }
    if track_path is not None:
        frames = read_jsonl(track_path)
        by_track: dict[int, dict[str, Any]] = {}
        class_counts: collections.Counter[int] = collections.Counter()
        for frame in frames:
            sequence = int(frame["frame_sequence"])
            pts = float(frame["source_pts_ms"])
            for detection in frame.get("detections", []):
                track_id = int(detection["track_id"])
                class_id = int(detection["class_id"])
                class_counts[class_id] += 1
                state = by_track.setdefault(track_id, {
                    "track_id": track_id,
                    "class_id": class_id,
                    "label": PPE_LABELS.get(class_id, str(class_id)),
                    "first_frame_sequence": sequence,
                    "last_frame_sequence": sequence,
                    "first_source_pts_ms": pts,
                    "last_source_pts_ms": pts,
                    "observations": 0,
                    "new_track_observations": 0,
                    "max_observed_frame_gap": 0,
                    "_last_sequence": sequence,
                })
                if state["class_id"] != class_id:
                    raise ValueError(f"track {track_id} changed class from {state['class_id']} to {class_id}")
                state["max_observed_frame_gap"] = max(
                    state["max_observed_frame_gap"], sequence - state["_last_sequence"])
                state["last_frame_sequence"] = sequence
                state["last_source_pts_ms"] = pts
                state["_last_sequence"] = sequence
                state["observations"] += 1
                state["new_track_observations"] += int(bool(detection.get("new_track", False)))
        tracks = []
        for state in by_track.values():
            state["observed_span_ms"] = round(
                state["last_source_pts_ms"] - state["first_source_pts_ms"], 3)
            state.pop("_last_sequence")
            tracks.append(state)
        tracks.sort(key=lambda row: (row["class_id"], row["track_id"]))
        result["tracker"] = {
            "frames_with_inference": len(frames),
            "frames_with_person_detection": sum(
                any(int(d.get("class_id", -1)) == 5 for d in frame.get("detections", []))
                for frame in frames),
            "detections_by_class": {
                PPE_LABELS.get(class_id, str(class_id)): count
                for class_id, count in sorted(class_counts.items())
            },
            "unique_tracks_by_class": dict(sorted(collections.Counter(
                state["label"] for state in tracks).items())),
            "tracks": tracks,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stdout_jsonl", type=Path)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.stdout_jsonl, args.tracks)
    encoded = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
