#!/usr/bin/env python3
"""Select deterministic, class-aware PPE INT8 calibration images from COCO train."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="root containing annotations/instances_train.json and images/train")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3576)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Toolkit2 text manifest, one absolute image path per line")
    parser.add_argument("--summary", type=Path, required=True,
                        help="JSON record with selected filenames, hashes, and class coverage")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")

    annotation_path = args.dataset_root / "annotations" / "instances_train.json"
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in document["categories"]}
    images = {item["id"]: item for item in document["images"]}
    labels: dict[int, set[int]] = defaultdict(set)
    for annotation in document["annotations"]:
        labels[annotation["image_id"]].add(annotation["category_id"])

    candidates: dict[int, Path] = {}
    for image_id, image in images.items():
        path = args.dataset_root / "images" / "train" / image["file_name"]
        if path.is_file():
            candidates[image_id] = path.resolve()
    missing = len(images) - len(candidates)
    if missing:
        raise FileNotFoundError(f"{missing} COCO training images are missing")
    if args.count > len(candidates):
        parser.error(f"count {args.count} exceeds {len(candidates)} available training images")

    image_frequency = Counter(category for ids in labels.values() for category in ids)
    tie_break = random.Random(args.seed)
    tie_rank = {image_id: tie_break.random() for image_id in candidates}
    selected: list[int] = []
    selected_per_class: Counter[int] = Counter()
    remaining = set(candidates)
    while len(selected) < args.count:
        def score(image_id: int) -> tuple[float, float, int]:
            class_ids = labels.get(image_id, set())
            rarity_gain = sum(
                1.0 / (math.sqrt(image_frequency[class_id]) * (1 + selected_per_class[class_id]))
                for class_id in class_ids
            )
            return rarity_gain, tie_rank[image_id], -image_id

        image_id = max(remaining, key=score)
        selected.append(image_id)
        selected_per_class.update(labels.get(image_id, set()))
        remaining.remove(image_id)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "\n".join(str(candidates[image_id]) for image_id in selected) + "\n",
        encoding="utf-8",
    )
    class_counts = Counter()
    for image_id in selected:
        for category_id in labels.get(image_id, set()):
            class_counts[categories[category_id]] += 1
    summary = {
        "selector": "greedy inverse-square-root class-image frequency; seeded tie break",
        "seed": args.seed,
        "count": len(selected),
        "source_split": "train",
        "annotation_file_sha256": sha256(annotation_path),
        "selected_images": [
            {
                "image_id": image_id,
                "file_name": images[image_id]["file_name"],
                "sha256": sha256(candidates[image_id]),
                "classes": sorted(categories[item] for item in labels.get(image_id, set())),
            }
            for image_id in selected
        ],
        "selected_image_class_coverage": dict(sorted(class_counts.items())),
        "class_image_frequency_in_train": {
            categories[class_id]: image_frequency[class_id]
            for class_id in sorted(categories)
        },
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps({
        "count": len(selected),
        "class_coverage": dict(sorted(class_counts.items())),
        "manifest": str(args.manifest),
        "summary": str(args.summary),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
