#!/usr/bin/env python3
"""Score RKNN predictions against a Roboflow YOLO-format Test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml


def digest_files(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def parse_class_map(raw: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for pair in raw.split(","):
        source, target = pair.split(":", maxsplit=1)
        result[int(source)] = int(target)
    if not result:
        raise ValueError("empty source-to-dataset class map")
    return result


def iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def match(predictions: list[dict[str, Any]], gt_by_image: dict[str, list[list[float]]], threshold: float):
    matched: dict[str, set[int]] = defaultdict(set)
    tp: list[int] = []
    fp: list[int] = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        image_name = prediction["file_name"]
        boxes = gt_by_image.get(image_name, [])
        candidates = [
            (iou(prediction["bbox_xyxy"], box), index)
            for index, box in enumerate(boxes) if index not in matched[image_name]
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= threshold:
            matched[image_name].add(best_index)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    return tp, fp


def ap101(tp: list[int], fp: list[int], total_gt: int) -> float | None:
    if total_gt == 0:
        return None
    cumulative_tp = cumulative_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for true_positive, false_positive in zip(tp, fp):
        cumulative_tp += true_positive
        cumulative_fp += false_positive
        recalls.append(cumulative_tp / total_gt)
        precisions.append(cumulative_tp / max(1, cumulative_tp + cumulative_fp))
    return sum(
        max((p for r, p in zip(recalls, precisions) if r >= target), default=0.0)
        for target in (step / 100 for step in range(101))
    ) / 101


def class_names_from_yaml(path: Path) -> list[str]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = value["names"]
    if isinstance(names, dict):
        names = [names.get(index, names.get(str(index))) for index in range(len(names))]
    if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
        raise ValueError("Roboflow data.yaml must contain a list or index map of class names")
    return names


def load_ground_truth(images_dir: Path, labels_dir: Path, class_count: int):
    images = sorted(path for path in images_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        raise ValueError(f"no images found under {images_dir}")
    by_class: dict[int, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    counts: Counter[int] = Counter()
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image {image_path}")
        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"missing YOLO label file for {image_path.name}")
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"malformed label {label_path}:{line_number}")
            class_id = int(values[0])
            if not 0 <= class_id < class_count:
                raise ValueError(f"label class id out of range in {label_path}:{line_number}")
            cx, cy, box_width, box_height = map(float, values[1:])
            box = [
                (cx - box_width / 2.0) * width,
                (cy - box_height / 2.0) * height,
                (cx + box_width / 2.0) * width,
                (cy + box_height / 2.0) * height,
            ]
            by_class[class_id][image_path.name].append(box)
            counts[class_id] += 1
    return images, by_class, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Roboflow YOLO export Test directory with images/, labels/, data.yaml")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="RKNN output JSONL with class_id mapped to native dataset class IDs")
    parser.add_argument("--model-class-map", required=True,
                        help="source model class ID to Roboflow native class ID, e.g. 0:9,1:4")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = class_names_from_yaml(args.dataset_root / "data.yaml")
    class_map = parse_class_map(args.model_class_map)
    if any(target < 0 or target >= len(names) for target in class_map.values()):
        raise ValueError("class map target is outside Roboflow native class order")
    images, gt_by_class, counts = load_ground_truth(
        args.dataset_root / "images", args.dataset_root / "labels", len(names))
    predictions_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    image_names = {path.name for path in images}
    if len(rows) != len(images):
        raise ValueError(f"RKNN predictions cover {len(rows)} images, expected {len(images)}")
    seen: set[str] = set()
    for row in rows:
        file_name = row["file_name"]
        if file_name not in image_names or file_name in seen:
            raise ValueError(f"unknown or duplicate prediction image: {file_name}")
        seen.add(file_name)
        for detection in row["detections"]:
            class_id = int(detection["class_id"])
            if class_id not in set(class_map.values()):
                raise ValueError(f"prediction class id {class_id} is not in native class map")
            predictions_by_class[class_id].append({
                "file_name": file_name,
                "score": float(detection["score"]),
                "bbox_xyxy": [float(value) for value in detection["bbox_xyxy"]],
            })

    iou_thresholds = [round(0.5 + index * 0.05, 2) for index in range(10)]
    per_class: dict[str, Any] = {}
    class_ap50: list[float] = []
    class_ap50_95: list[float] = []
    operating: dict[str, Any] = {}
    for class_id, name in enumerate(names):
        gt = gt_by_class.get(class_id, {})
        total_gt = counts[class_id]
        predictions = predictions_by_class.get(class_id, [])
        class_aps = []
        for threshold in iou_thresholds:
            tp, fp = match(predictions, gt, threshold)
            ap = ap101(tp, fp, total_gt)
            if ap is not None:
                class_aps.append(ap)
        ap50 = class_aps[0] if class_aps else None
        ap50_95 = sum(class_aps) / len(class_aps) if class_aps else None
        if ap50 is not None and ap50_95 is not None:
            class_ap50.append(ap50)
            class_ap50_95.append(ap50_95)
        selected = [item for item in predictions if item["score"] >= args.confidence_threshold]
        tp, fp = match(selected, gt, 0.50)
        true_count, false_count = sum(tp), sum(fp)
        operating[name] = {
            "ground_truth": total_gt,
            "predictions_at_confidence_threshold": len(selected),
            "true_positives_iou_0_50": true_count,
            "false_positives_iou_0_50": false_count,
            "false_negatives_iou_0_50": total_gt - true_count,
            "recall_at_confidence_threshold": true_count / total_gt if total_gt else None,
            "precision_at_confidence_threshold": true_count / len(selected) if selected else None,
        }
        per_class[name] = {
            "ground_truth": total_gt,
            "predictions_at_low_score_floor": len(predictions),
            "AP50": ap50,
            "AP50_95": ap50_95,
        }

    required_aliases = {
        "Person": {"person"},
        "Helmet": {"helmet", "hardhat", "hard hat"},
        "Vest": {"vest", "safety vest"},
        "No-Helmet": {"no helmet", "no hardhat", "no hard hat"},
        "No-Vest": {"no vest", "no safety vest"},
    }
    def normalize_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    focus: dict[str, Any] = {}
    for label, aliases in required_aliases.items():
        target_id = next((i for i, name in enumerate(names)
                          if normalize_name(name) in aliases), None)
        focus[label] = per_class[names[target_id]] if target_id is not None else {
            "status": "class absent from this dataset version",
            "ground_truth": 0,
            "recall_at_confidence_threshold": None,
        }

    unmapped_gt = {names[i]: counts[i] for i in range(len(names))
                   if counts[i] > 0 and i not in set(class_map.values())}
    result = {
        "dataset": "Roboflow Universe construction-site-safety v1 Test",
        "image_count": len(images),
        "class_names_native_order": names,
        "ground_truth_counts": {names[i]: counts[i] for i in range(len(names))},
        "class_map_model_source_to_roboflow_native": {
            str(source): {"id": target, "name": names[target]}
            for source, target in sorted(class_map.items())
        },
        "confidence_floor_for_AP": 0.001,
        "nms_iou": 0.7,
        "confidence_threshold_for_operating_recall": args.confidence_threshold,
        "iou_thresholds": iou_thresholds,
        "AP50": sum(class_ap50) / len(class_ap50) if class_ap50 else None,
        "AP50_95": sum(class_ap50_95) / len(class_ap50_95) if class_ap50_95 else None,
        "ap_class_count_with_ground_truth": len(class_ap50),
        "per_class": per_class,
        "focus_classes": focus,
        "operating_point_iou_0_50": operating,
        "unmapped_annotated_classes": unmapped_gt,
        "annotation_sha256": digest_files(sorted((args.dataset_root / "labels").glob("*.txt")),
                                          args.dataset_root),
        "image_manifest_sha256": digest_files(images, args.dataset_root),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "metric_definition": "101-point interpolated AP per class; mean over native classes with GT; AP50:95 averages IoU 0.50..0.95 at 0.05 steps",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("image_count", "AP50", "AP50_95", "focus_classes", "unmapped_annotated_classes")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
