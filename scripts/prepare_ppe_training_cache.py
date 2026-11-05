#!/usr/bin/env python3
"""Create an ignored, max-side resized COCO cache for PPE training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


def prepare_split(source_root: Path, output_root: Path, split: str, max_side: int) -> dict[str, Any]:
    annotation_path = source_root / "annotations" / f"instances_{split}.json"
    source = json.loads(annotation_path.read_text(encoding="utf-8"))
    input_images = source_root / "images" / split
    output_images = output_root / "images" / split
    output_images.mkdir(parents=True, exist_ok=True)

    scale_by_id: dict[int, tuple[float, float]] = {}
    images: list[dict[str, Any]] = []
    for image_info in source["images"]:
        image_path = input_images / image_info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {image_path}")
        old_height, old_width = image.shape[:2]
        scale = min(1.0, max_side / max(old_width, old_height))
        new_width = max(1, int(round(old_width * scale)))
        new_height = max(1, int(round(old_height * scale)))
        if (new_width, new_height) != (old_width, old_height):
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(output_images / image_info["file_name"]), image,
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"cannot write resized image {image_path}")
        scale_x = new_width / old_width
        scale_y = new_height / old_height
        scale_by_id[int(image_info["id"])] = (scale_x, scale_y)
        copied_info = dict(image_info)
        copied_info["width"] = new_width
        copied_info["height"] = new_height
        images.append(copied_info)

    annotations: list[dict[str, Any]] = []
    for annotation in source["annotations"]:
        copied = dict(annotation)
        scale_x, scale_y = scale_by_id[int(annotation["image_id"])]
        x, y, width, height = annotation["bbox"]
        copied["bbox"] = [x * scale_x, y * scale_y, width * scale_x, height * scale_y]
        copied["area"] = copied["bbox"][2] * copied["bbox"][3]
        annotations.append(copied)

    payload = dict(source)
    payload["images"] = images
    payload["annotations"] = annotations
    payload.setdefault("info", {})["max_side_training_cache"] = max_side
    out_annotations = output_root / "annotations"
    out_annotations.mkdir(parents=True, exist_ok=True)
    (out_annotations / annotation_path.name).write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    return {
        "split": split,
        "images": len(images),
        "annotations": len(annotations),
        "max_side": max(max(image["width"], image["height"]) for image in images),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="dataset root with images/ and annotations/")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--max-side", type=int, default=640)
    args = parser.parse_args()
    if args.max_side < 32 or args.max_side % 32:
        raise ValueError("max-side must be a positive multiple of 32")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    report = {
        "source": str(source_root),
        "output": str(output_root),
        "license": "CC BY 4.0",
        "attribution": "Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0",
        "splits": [
            prepare_split(source_root, output_root, split, args.max_side)
            for split in ("train", "val")
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
