#!/usr/bin/env python3
"""Convert the released YOLO labels to COCO JSON for YOLOX training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2


LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
DATASET_DOCUMENTATION_URL = "https://docs.ultralytics.com/zh/datasets/detect/construction-ppe"
ATTRIBUTION = "Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0"


def load_class_names(path: Path) -> list[str]:
    names: dict[int, str] = {}
    in_names = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names:
            match = re.match(r"\s+(\d+)\s*:\s*(.*?)\s*$", line)
            if match:
                names[int(match.group(1))] = match.group(2).strip("'\"")
            elif line and not line[0].isspace():
                break
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError(f"invalid or non-contiguous class list in {path}")
    return [names[index] for index in range(len(names))]


def convert_split(root: Path, split: str, names: list[str]) -> tuple[dict[str, Any], int]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    clipped_boxes = 0

    for image_id, image_path in enumerate(sorted(image_dir.glob("*.jpg")), 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"cannot decode image {image_path}")
        height, width = image.shape[:2]
        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"missing label file {label_path}")
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"expected 5 YOLO fields at {label_path}:{line_number}")
            class_id = int(values[0])
            center_x, center_y, norm_width, norm_height = map(float, values[1:])
            if not 0 <= class_id < len(names):
                raise ValueError(f"class id outside names at {label_path}:{line_number}")
            if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and
                    0 < norm_width <= 1 and 0 < norm_height <= 1):
                raise ValueError(f"invalid normalized box at {label_path}:{line_number}")

            x1 = (center_x - norm_width / 2.0) * width
            y1 = (center_y - norm_height / 2.0) * height
            x2 = (center_x + norm_width / 2.0) * width
            y2 = (center_y + norm_height / 2.0) * height
            clipped = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
            if clipped != (x1, y1, x2, y2):
                normalized_excess = max(
                    abs(x1 - clipped[0]) / width,
                    abs(y1 - clipped[1]) / height,
                    abs(x2 - clipped[2]) / width,
                    abs(y2 - clipped[3]) / height,
                )
                if normalized_excess > 1e-4:
                    raise ValueError(f"box exceeds image beyond rounding tolerance at {label_path}:{line_number}")
                clipped_boxes += 1
            x1, y1, x2, y2 = clipped
            box_width, box_height = x2 - x1, y2 - y1
            if box_width <= 0 or box_height <= 0:
                raise ValueError(f"empty box after clipping at {label_path}:{line_number}")
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [x1, y1, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            })

    categories = [
        {"id": class_id + 1, "name": name, "supercategory": "safety"}
        for class_id, name in enumerate(names)
    ]
    payload = {
        "info": {
            "description": "Ultralytics Construction-PPE converted from YOLO to COCO format",
            "source_attribution": ATTRIBUTION,
            "source_license": "CC BY 4.0",
            "source_license_url": LICENSE_URL,
            "source_url": DATASET_URL,
            "source_documentation_url": DATASET_DOCUMENTATION_URL,
            "source_release": "ultralytics/assets v0.0.0",
        },
        "licenses": [{"id": 1, "name": "CC BY 4.0", "url": LICENSE_URL}],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return payload, clipped_boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    names = load_class_names(root / "data.yaml")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"dataset": root.name, "classes": names, "splits": {}}
    for split in ("train", "val"):
        payload, clipped = convert_split(root, split, names)
        output_name = f"instances_{split}.json"
        output_path = args.output_dir / output_name
        output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        report["splits"][split] = {
            "images": len(payload["images"]),
            "annotations": len(payload["annotations"]),
            "rounding_boundary_boxes_clipped": clipped,
            "annotation_file": output_name,
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
