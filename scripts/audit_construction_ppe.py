#!/usr/bin/env python3
"""Audit the official Construction-PPE archive without modifying its contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
DATASET_DOCUMENTATION_URL = "https://docs.ultralytics.com/zh/datasets/detect/construction-ppe"
DATASET_YAML_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/construction-ppe.yaml"
ASSET_RELEASE = "ultralytics/assets v0.0.0"
KNOWN_PROVENANCE_KEYS = {
    "source", "source_type", "synthetic", "generated", "origin", "image_type",
    "human_collected", "is_synthetic",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_class_names(data_yaml: Path) -> list[str]:
    names: dict[int, str] = {}
    in_names = False
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
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
        raise ValueError(f"invalid or non-contiguous class list in {data_yaml}")
    return [names[index] for index in range(len(names))]


def audit(root: Path, archive: Path | None = None) -> dict[str, Any]:
    readme_path = root / "README.md"
    license_path = root / "LICENSE.txt"
    manifest_path = root / "manifest.jsonl"
    yaml_path = root / "data.yaml"
    readme = readme_path.read_text(encoding="utf-8")
    license_text = license_path.read_text(encoding="utf-8") if license_path.is_file() else ""
    license_header = next((line.strip() for line in license_text.splitlines() if line.strip()), "")
    license_evidence = {
        "embedded_license_file": license_path.name,
        "present": license_path.is_file(),
        "header": license_header,
        "readme_mentions_cc_by_4": "CC BY 4.0" in readme,
        "sha256": sha256_file(license_path) if license_path.is_file() else None,
        "matches_cc_by_4_declaration": (
            license_header == "Attribution 4.0 International"
            and "CC BY 4.0" in readme
            and "creativecommons.org/licenses/by/4.0" in readme.lower()
        ),
    }
    names = load_class_names(yaml_path)
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_by_file = {row["file"]: row for row in manifest_rows}
    provenance_fields = sorted(
        {key for row in manifest_rows for key in row}
        & KNOWN_PROVENANCE_KEYS
    )
    provenance_counts: Counter[str] = Counter()
    if provenance_fields:
        for row in manifest_rows:
            value = next((row[key] for key in provenance_fields if key in row), None)
            provenance_counts[str(value)] += 1

    split_stats: dict[str, Any] = {}
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    malformed: list[str] = []
    out_of_bounds: list[str] = []
    rounding_boundary_boxes = 0
    max_edge_excess = 0.0
    hash_errors: list[str] = []
    missing_labels: list[str] = []
    image_hashes: dict[str, list[str]] = defaultdict(list)
    all_images: set[str] = set()

    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        boxes = 0
        split_images = 0
        split_manifest_mismatches = 0
        for image in images:
            split_images += 1
            all_images.add(image.name)
            manifest_row = manifest_by_file.get(image.name)
            if manifest_row is None or manifest_row.get("split") != split:
                split_manifest_mismatches += 1
            else:
                actual_hash = sha256_file(image)
                image_hashes[actual_hash].append(f"{split}/{image.name}")
                if actual_hash != manifest_row.get("sha256"):
                    hash_errors.append(f"{split}/{image.name}")

            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                missing_labels.append(f"{split}/{image.name}")
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                values = line.split()
                try:
                    class_id = int(values[0])
                    x_center, y_center, width, height = map(float, values[1:5])
                    structurally_valid = (
                        len(values) == 5
                        and 0 <= class_id < len(names)
                        and all(math.isfinite(value) for value in
                                (x_center, y_center, width, height))
                        and 0.0 <= x_center <= 1.0
                        and 0.0 <= y_center <= 1.0
                        and 0.0 < width <= 1.0
                        and 0.0 < height <= 1.0
                    )
                except (ValueError, IndexError):
                    structurally_valid = False
                    class_id = -1
                    x_center = y_center = width = height = 0.0
                edge_excess = max(
                    0.0,
                    -(x_center - width / 2.0),
                    x_center + width / 2.0 - 1.0,
                    -(y_center - height / 2.0),
                    y_center + height / 2.0 - 1.0,
                )
                if not structurally_valid:
                    malformed.append(f"{split}/{label.name}:{line_number}")
                    continue
                if edge_excess > 1e-4:
                    out_of_bounds.append(f"{split}/{label.name}:{line_number}")
                    continue
                if edge_excess > 0.0:
                    rounding_boundary_boxes += 1
                    max_edge_excess = max(max_edge_excess, edge_excess)
                class_counts[split][names[class_id]] += 1
                boxes += 1
            if manifest_row and manifest_row.get("annotations") != sum(
                1 for line in label.read_text(encoding="utf-8").splitlines() if line.strip()
            ):
                malformed.append(f"{split}/{label.name}:manifest_annotation_count")

        split_stats[split] = {
            "images": split_images,
            "label_files": sum(1 for p in label_dir.glob("*.txt") if p.is_file()),
            "boxes": boxes,
            "class_counts": dict(sorted(class_counts[split].items())),
            "manifest_split_mismatches": split_manifest_mismatches,
        }

    cross_split_duplicates = [
        paths for paths in image_hashes.values()
        if len(paths) > 1 and len({path.split("/", 1)[0] for path in paths}) > 1
    ]
    extra_manifest_files = sorted(set(manifest_by_file) - all_images)
    archive_info: dict[str, Any] | None = None
    if archive is not None:
        archive_info = {"path": archive.name, "bytes": archive.stat().st_size,
                        "sha256": sha256_file(archive)}

    return {
        "dataset": "Ultralytics Construction-PPE",
        "asset_release": ASSET_RELEASE,
        "source_url": DATASET_URL,
        "source_documentation_url": DATASET_DOCUMENTATION_URL,
        "source_yaml_url": DATASET_YAML_URL,
        "license": "CC BY 4.0",
        "license_evidence": license_evidence,
        "attribution": "Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0",
        "classes": names,
        "split": split_stats,
        "manifest_rows": len(manifest_rows),
        "manifest_extra_files": extra_manifest_files,
        "sha256_mismatches": hash_errors,
        "missing_labels": missing_labels,
        "malformed_annotations": malformed,
        "boxes_outside_image_beyond_rounding_tolerance": out_of_bounds,
        "rounding_boundary_boxes": rounding_boundary_boxes,
        "maximum_normalized_edge_excess": max_edge_excess,
        "cross_split_exact_duplicates": cross_split_duplicates,
        "provenance_fields_in_manifest": provenance_fields,
        "provenance_counts": dict(provenance_counts),
        "synthetic_image_proportion": (
            "not disclosed per image; release README says human-collected and generated images are mixed"
            if not provenance_fields else "computed from manifest provenance fields"
        ),
        "has_separate_test_split": (root / "images" / "test").is_dir(),
        "release_readme_mentions_generated_images": "generated images" in readme.lower(),
        "archive": archive_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.dataset_root, args.archive)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    problems = (
        report["sha256_mismatches"]
        or report["missing_labels"]
        or report["malformed_annotations"]
        or report["boxes_outside_image_beyond_rounding_tolerance"]
        or report["manifest_extra_files"]
        or not report["license_evidence"]["matches_cc_by_4_declaration"]
        or any(value["manifest_split_mismatches"] for value in report["split"].values())
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
