#!/usr/bin/env python3
"""Score board RKNN PPE detections against the dataset's COCO annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


MODEL_CLASSES = [
    "helmet", "gloves", "vest", "boots", "goggles", "Person",
    "no_helmet", "no_goggle", "no_gloves", "no_boots", "forklift", "no_vest",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions(
    predictions: list[dict[str, Any]],
    ground_truth: dict[int, list[list[float]]],
    threshold: float,
) -> tuple[list[int], list[int]]:
    matched: dict[int, set[int]] = defaultdict(set)
    true_positives: list[int] = []
    false_positives: list[int] = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        image_id = prediction["image_id"]
        boxes = ground_truth.get(image_id, [])
        candidates = [
            (iou(prediction["bbox_xyxy"], box), index)
            for index, box in enumerate(boxes) if index not in matched[image_id]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= threshold:
            matched[image_id].add(best_index)
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)
    return true_positives, false_positives


def average_precision(true_positives: list[int], false_positives: list[int], total_gt: int) -> float:
    if total_gt == 0:
        return math.nan
    cumulative_tp = cumulative_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for tp, fp in zip(true_positives, false_positives):
        cumulative_tp += tp
        cumulative_fp += fp
        recalls.append(cumulative_tp / total_gt)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
    return sum(
        max((precision for recall, precision in zip(recalls, precisions) if recall >= target),
            default=0.0)
        for target in (step / 100 for step in range(101))
    ) / 101


def evaluate(annotation_path: Path, predictions_path: Path, confidence_threshold: float) -> dict[str, Any]:
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images_by_name = {image["file_name"]: image["id"] for image in coco["images"]}
    categories_by_id = {category["id"]: category["name"] for category in coco["categories"]}
    category_by_name = {name: category_id for category_id, name in categories_by_id.items()}
    if any(name not in category_by_name for name in MODEL_CLASSES):
        raise ValueError("COCO category names do not match the fixed YOLOX-PPE class order")
    class_to_category = {class_id: category_by_name[name]
                         for class_id, name in enumerate(MODEL_CLASSES)}

    gt_by_category: dict[int, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for annotation in coco["annotations"]:
        category_id = annotation["category_id"]
        x, y, width, height = annotation["bbox"]
        gt_by_category[category_id][annotation["image_id"]].append(
            [x, y, x + width, y + height]
        )

    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != len(images_by_name):
        raise ValueError(f"predictions cover {len(rows)} images, expected {len(images_by_name)}")
    seen: set[int] = set()
    by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_id = images_by_name.get(row["file_name"])
        if image_id is None or image_id in seen:
            raise ValueError(f"unknown or duplicate prediction image: {row['file_name']}")
        seen.add(image_id)
        for detection in row["detections"]:
            if detection["class_id"] not in class_to_category:
                raise ValueError(f"invalid PPE class id: {detection['class_id']}")
            by_category[class_to_category[detection["class_id"]]].append({
                "image_id": image_id,
                "score": float(detection["score"]),
                "bbox_xyxy": [float(value) for value in detection["bbox_xyxy"]],
            })

    thresholds = [round(0.50 + 0.05 * index, 2) for index in range(10)]
    per_class: dict[str, Any] = {}
    aps_50_95: list[float] = []
    aps_50: list[float] = []
    operating_counts: dict[str, Any] = {}
    for class_id, name in enumerate(MODEL_CLASSES):
        category_id = class_to_category[class_id]
        gt = gt_by_category.get(category_id, {})
        total_gt = sum(len(boxes) for boxes in gt.values())
        predictions = by_category.get(category_id, [])
        class_aps: list[float] = []
        for threshold in thresholds:
            tp, fp = match_predictions(predictions, gt, threshold)
            class_aps.append(average_precision(tp, fp, total_gt))
        ap50 = class_aps[0]
        ap50_95 = sum(class_aps) / len(class_aps) if total_gt else math.nan
        if total_gt:
            aps_50.append(ap50)
            aps_50_95.append(ap50_95)

        operating_predictions = [item for item in predictions
                                if item["score"] >= confidence_threshold]
        tp, fp = match_predictions(operating_predictions, gt, 0.50)
        true_count = sum(tp)
        false_count = sum(fp)
        operating_counts[name] = {
            "ground_truth": total_gt,
            "predictions_at_confidence_threshold": len(operating_predictions),
            "true_positives_at_iou_0_50": true_count,
            "false_positives_at_iou_0_50": false_count,
            "false_negatives_at_iou_0_50": total_gt - true_count,
            "recall_at_confidence_threshold": true_count / total_gt if total_gt else None,
        }
        per_class[name] = {
            "ground_truth": total_gt,
            "predictions_at_low_score_threshold": len(predictions),
            "AP50": ap50 if math.isfinite(ap50) else None,
            "AP50_95": ap50_95 if math.isfinite(ap50_95) else None,
        }

    return {
        "dataset_split": "development validation; the FP epoch selection split, not an independent test set",
        "annotation_sha256": sha256(annotation_path),
        "predictions_sha256": sha256(predictions_path),
        "image_count": len(rows),
        "confidence_threshold_for_operating_recall": confidence_threshold,
        "AP50": sum(aps_50) / len(aps_50) if aps_50 else None,
        "AP50_95": sum(aps_50_95) / len(aps_50_95) if aps_50_95 else None,
        "iou_thresholds": thresholds,
        "per_class": per_class,
        "operating_point_iou_0_50": operating_counts,
        "required_violation_class_recall": {
            name: operating_counts[name]["recall_at_confidence_threshold"]
            for name in ("no_helmet", "no_vest")
        },
        "metric_definition": "101-point interpolated AP per class, averaged over classes with GT; AP50:95 averages IoU 0.50 to 0.95 in 0.05 steps",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.annotations, args.predictions, args.confidence_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("image_count", "AP50", "AP50_95", "required_violation_class_recall")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
