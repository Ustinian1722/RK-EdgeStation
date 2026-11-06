#!/usr/bin/env python3
"""Recompute clip-level PPE event-presence metrics from labels and RK3576 evidence.

The input clips repeat one labeled validation image. This evaluator measures only
whether a PPE-violation event was present in each short replay; it does not score
temporal boundaries or estimate deployment accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # The label and event-count audit can still run without media decoding.
    cv2 = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = (
    ROOT / ".work/construction-ppe/extracted/construction-ppe/annotations/instances_val.json"
)
DEFAULT_IMAGES = ROOT / ".work/construction-ppe/extracted/construction-ppe/images/val"
DEFAULT_VIDEOS = ROOT / ".work/ppe-eval-20261106"
DEFAULT_SMOKE = ROOT / "evidence/ppe/rk3576-ppe-api-smoke.json"
DEFAULT_AUDIT = ROOT / "evidence/ppe/construction-ppe-audit.json"
DEFAULT_OUTPUT = ROOT / "evidence/ppe/event-level-replay-evaluation.json"
VIOLATION_CLASSES = {"no_helmet", "no_vest"}
POSITIVE_EQUIPMENT_CLASSES = {"helmet", "vest"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside_person_center(item: dict[str, Any], person: dict[str, Any]) -> bool:
    px, py, pw, ph = person["bbox"]
    x, y, width, height = item["bbox"]
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    return px <= center_x <= px + pw and py <= center_y <= py + ph


def metric(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def inspect_replay_video(video_path: Path, source_image_path: Path) -> dict[str, Any] | None:
    if cv2 is None:
        return None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot decode replay video {video_path}")
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    first_frame = None
    frame_hashes: set[str] = set()
    frame_count = 0
    maximum_mean_difference = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if first_frame is None:
                first_frame = frame.copy()
            else:
                difference = cv2.absdiff(first_frame, frame)
                maximum_mean_difference = max(maximum_mean_difference, float(difference.mean()))
            frame_hashes.add(hashlib.sha256(frame.tobytes()).hexdigest())
            frame_count += 1
    finally:
        capture.release()
    if first_frame is None:
        raise ValueError(f"replay video contains no decodable frames: {video_path}")
    source_image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise ValueError(f"cannot decode labeled source image {source_image_path}")
    if source_image.shape[:2] != first_frame.shape[:2]:
        source_image = cv2.resize(source_image, (first_frame.shape[1], first_frame.shape[0]))
    source_mae = float(cv2.absdiff(first_frame, source_image).mean())
    return {
        "declared_frames": declared_frames,
        "decoded_frames": frame_count,
        "fps": fps,
        "duration_seconds": frame_count / fps if fps > 0 else None,
        "width": int(first_frame.shape[1]),
        "height": int(first_frame.shape[0]),
        "unique_decoded_frame_hashes": len(frame_hashes),
        "maximum_mean_pixel_difference_from_first_frame": maximum_mean_difference,
        "first_frame_mean_absolute_difference_from_labeled_source": source_mae,
    }


def evaluate(annotations_path: Path, images_dir: Path, videos_dir: Path,
             smoke_path: Path, audit_path: Path) -> dict[str, Any]:
    annotations = load_json(annotations_path)
    smoke = load_json(smoke_path)
    audit = load_json(audit_path)
    category_names = {int(row["id"]): row["name"] for row in annotations["categories"]}
    images = {int(row["id"]): row for row in annotations["images"]}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in annotations["annotations"]:
        grouped.setdefault(int(item["image_id"]), []).append(item)

    scenarios = smoke.get("scenarios", {})
    if set(scenarios) != {"positive", "negative"}:
        raise ValueError("the board smoke record must contain positive and negative scenarios")

    rows: list[dict[str, Any]] = []
    confusion = {"true_positive": 0, "false_positive": 0,
                 "false_negative": 0, "true_negative": 0}
    status_correct = 0
    status_count = 0
    media_verified = True

    for name in ("positive", "negative"):
        scenario = scenarios[name]
        image_id = int(scenario["source_validation_image_id"])
        if image_id not in images:
            raise ValueError(f"validation image id {image_id} is absent from COCO annotations")
        image = images[image_id]
        image_annotations = grouped.get(image_id, [])
        persons = [item for item in image_annotations
                   if category_names[int(item["category_id"])] == "Person"]
        if not persons:
            raise ValueError(f"validation image {image_id} has no labeled Person box")

        labels = Counter(category_names[int(item["category_id"])]
                         for item in image_annotations)
        linked_labels: set[str] = set()
        for person in persons:
            for item in image_annotations:
                label = category_names[int(item["category_id"])]
                if label in VIOLATION_CLASSES | POSITIVE_EQUIPMENT_CLASSES and inside_person_center(item, person):
                    linked_labels.add(label)

        ground_truth_event = bool(linked_labels & VIOLATION_CLASSES)
        if ground_truth_event:
            ground_truth_status = "violation"
        elif POSITIVE_EQUIPMENT_CLASSES <= linked_labels:
            ground_truth_status = "compliant"
        else:
            ground_truth_status = "unknown"

        predicted_event_count = int(scenario["ppe_violation_count"])
        predicted_event = predicted_event_count > 0
        predicted_status_counts = scenario["ppe_person_status"]
        predicted_status = next(
            (state for state in ("violation", "compliant", "unknown")
             if int(predicted_status_counts.get(state, 0)) > 0),
            "no_person",
        )

        if ground_truth_event and predicted_event:
            confusion["true_positive"] += 1
        elif ground_truth_event:
            confusion["false_negative"] += 1
        elif predicted_event:
            confusion["false_positive"] += 1
        else:
            confusion["true_negative"] += 1

        status_count += len(persons)
        if ground_truth_status == predicted_status:
            status_correct += 1

        image_path = images_dir / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"labeled source image is missing: {image_path}")

        video_path = videos_dir / scenario["video"]
        video_hash_record = scenario["video_sha256"]
        video_hash_local = None
        video_analysis = None
        if video_path.is_file():
            video_hash_local = sha256_file(video_path)
            if video_hash_local != video_hash_record:
                raise ValueError(f"video hash mismatch for {video_path}")
            video_analysis = inspect_replay_video(video_path, image_path)
            if video_analysis is not None and video_analysis["decoded_frames"] != int(scenario["decoded_frames"]):
                raise ValueError(f"decoded video frame count does not match board smoke record: {video_path}")
        else:
            media_verified = False

        rows.append({
            "scenario": name,
            "source_split": "val (development split)",
            "source_image_id": image_id,
            "source_image": image["file_name"],
            "source_image_sha256": sha256_file(image_path),
            "source_image_labels": dict(sorted(labels.items())),
            "person_boxes": len(persons),
            "person_linked_ppe_labels_by_center_containment": sorted(linked_labels),
            "ground_truth_event_present": ground_truth_event,
            "ground_truth_person_status": ground_truth_status,
            "replay_video": scenario["video"],
            "replay_video_sha256_recorded": video_hash_record,
            "replay_video_sha256_local": video_hash_local,
            "replay_video_hash_verified_locally": video_hash_local == video_hash_record,
            "replay_video_content_check": video_analysis,
            "decoded_frames": int(scenario["decoded_frames"]),
            "predicted_event_count": predicted_event_count,
            "predicted_event_present": predicted_event,
            "predicted_person_status_counts": predicted_status_counts,
            "predicted_person_status": predicted_status,
            "status_match": ground_truth_status == predicted_status,
            "worker_exit_code": int(scenario["worker_exit_code"]),
            "health_status": scenario["health_status"],
        })

    tp = confusion["true_positive"]
    fp = confusion["false_positive"]
    fn = confusion["false_negative"]
    tn = confusion["true_negative"]
    total = tp + fp + fn + tn
    return {
        "evaluation_name": "RK3576 PPE fixed-scene replay event-presence check",
        "date": smoke.get("date"),
        "board": smoke.get("board"),
        "model": smoke.get("model"),
        "unit": "one 150-frame replay clip; binary presence of at least one PPE-violation event",
        "label_rule": "positive if a source COCO no_helmet/no_vest box is linked to a labeled person by center containment",
        "status_rule": "person status follows the production association rule: violation evidence wins; compliant requires linked helmet and vest; otherwise unknown",
        "dataset_archive_sha256": audit.get("archive", {}).get("sha256"),
        "dataset_release": audit.get("asset_release"),
        "dataset_license": audit.get("license"),
        "dataset_documentation_url": audit.get("source_documentation_url"),
        "checkpoint_selection_split": "The same 200-image validation split used for model selection; not an independent test split.",
        "clip_count": total,
        "media_hashes_verified_locally": media_verified,
        "confusion_matrix_clip_presence": confusion,
        "clip_presence_metrics": {
            "accuracy": metric(tp + tn, total),
            "precision": metric(tp, tp + fp),
            "recall": metric(tp, tp + fn),
            "specificity": metric(tn, tn + fp),
            "f1": metric(2 * tp, 2 * tp + fp + fn),
        },
        "person_status": {
            "persons_scored": status_count,
            "correct": status_correct,
            "accuracy": metric(status_correct, status_count),
            "unknown_status_examples": sum(row["ground_truth_person_status"] == "unknown" for row in rows),
        },
        "scenarios": rows,
        "scope_limits": [
            "The two clips each repeat one still image; this is clip-level event presence, not temporal event-boundary matching.",
            "Both images come from the development validation split used for checkpoint selection; results are not independent generalization evidence.",
            "The sample has one positive and one compliant negative clip; the scores are a narrow workflow check, not a safety performance estimate.",
            "Danger-zone, dwell, channel-occupancy, and forklift/person events were observed in the board smoke run but have no independent temporal ground-truth labels here.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--smoke-evidence", type=Path, default=DEFAULT_SMOKE)
    parser.add_argument("--dataset-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = evaluate(args.annotations, args.images_dir, args.videos_dir,
                      args.smoke_evidence, args.dataset_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "clip_count": result["clip_count"],
        "confusion_matrix": result["confusion_matrix_clip_presence"],
        "metrics": result["clip_presence_metrics"],
        "person_status": result["person_status"],
        "media_hashes_verified_locally": result["media_hashes_verified_locally"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
