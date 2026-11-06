#!/usr/bin/env python3
"""Compare two first-frame RKNN input/detection dump directories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path
from typing import Any


def manifest(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in (path / "tensor-manifest.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data


def load_detections(path: Path) -> list[dict[str, Any]]:
    with (path / "rknn-detections.csv").open(newline="", encoding="utf-8") as stream:
        return [
            {
                "class_id": int(row["class_id"]),
                "label": row["label"],
                "confidence": float(row["confidence"]),
                "box": [float(row[key]) for key in ("left", "top", "right", "bottom")],
            }
            for row in csv.DictReader(stream)
        ]


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def compare_detections(first: list[dict[str, Any]], second: list[dict[str, Any]],
                       threshold: float) -> dict[str, Any]:
    candidates = sorted(
        (
            (iou(a["box"], b["box"]), i, j)
            for i, a in enumerate(first) for j, b in enumerate(second)
            if a["class_id"] == b["class_id"]
        ),
        reverse=True,
    )
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for overlap, i, j in candidates:
        if overlap < threshold or i in matched_a or j in matched_b:
            continue
        matched_a.add(i)
        matched_b.add(j)
        pairs.append((i, j, overlap))
    confidence_deltas = [
        abs(first[i]["confidence"] - second[j]["confidence"])
        for i, j, _ in pairs
    ]
    return {
        "first_detection_count": len(first),
        "second_detection_count": len(second),
        "matched_same_class_iou_at_least_threshold": len(pairs),
        "iou_threshold": threshold,
        "unmatched_first": len(first) - len(pairs),
        "unmatched_second": len(second) - len(pairs),
        "mean_matched_iou": sum(overlap for _, _, overlap in pairs) / len(pairs) if pairs else None,
        "mean_matched_confidence_delta": (
            sum(confidence_deltas) / len(confidence_deltas) if confidence_deltas else None
        ),
        "max_matched_confidence_delta": max(confidence_deltas) if confidence_deltas else None,
    }


def compare_float_tensors(first_path: Path, second_path: Path) -> dict[str, Any]:
    first = first_path.read_bytes()
    second = second_path.read_bytes()
    if len(first) != len(second) or len(first) % 4:
        raise ValueError(f"output tensor byte counts are incompatible: {first_path.name}")
    first_values = struct.iter_unpack("<f", first)
    second_values = struct.iter_unpack("<f", second)
    count = len(first) // 4
    absolute_differences = [
        abs(a[0] - b[0]) for a, b in zip(first_values, second_values)
    ]
    return {
        "bytes": len(first),
        "float_count": count,
        "exact_float_fraction": sum(value == 0.0 for value in absolute_differences) / count if count else 1.0,
        "mean_absolute_difference": sum(absolute_differences) / count if count else 0.0,
        "maximum_absolute_difference": max(absolute_differences, default=0.0),
    }


def output_values(path: Path, requested_type: str, metadata: dict[str, str],
                  output_index: str):
    raw = path.read_bytes()
    if requested_type == "float32":
        if len(raw) % 4:
            raise ValueError(f"float32 tensor byte count is not aligned: {path.name}")
        return (item[0] for item in struct.iter_unpack("<f", raw)), len(raw) // 4
    if requested_type != "native":
        raise ValueError(f"unknown requested output representation: {requested_type}")
    prefix = f"{output_index}."
    tensor_type = metadata.get(prefix + "type")
    quantization = metadata.get(prefix + "qnt_type")
    if quantization not in {"AFFINE", "AFFINE_ASYMMETRIC"} or tensor_type not in {"INT8", "UINT8"}:
        raise ValueError(f"unsupported native tensor representation for {output_index}")
    scale = float(metadata[prefix + "scale"])
    zero_point = int(metadata[prefix + "zero_point"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid quantization scale for {output_index}")
    if tensor_type == "INT8":
        values = ((byte - 256 if byte >= 128 else byte) - zero_point for byte in raw)
    else:
        values = (byte - zero_point for byte in raw)
    return ((value * scale for value in values), len(raw))


def compare_numeric_outputs(first_path: Path, first_metadata: dict[str, str],
                            second_path: Path, second_metadata: dict[str, str],
                            output_index: str) -> dict[str, Any]:
    first_shape = first_metadata.get(f"{output_index}.shape")
    second_shape = second_metadata.get(f"{output_index}.shape")
    if first_shape != second_shape:
        return {"comparison_skipped": "tensor shapes differ", "shapes": [first_shape, second_shape]}
    first_values, first_count = output_values(
        first_path, first_metadata.get("output_requested_type", ""), first_metadata, output_index
    )
    second_values, second_count = output_values(
        second_path, second_metadata.get("output_requested_type", ""), second_metadata, output_index
    )
    if first_count != second_count:
        return {"comparison_skipped": "tensor element counts differ", "element_counts": [first_count, second_count]}
    absolute_sum = 0.0
    maximum = 0.0
    exact_count = 0
    dot = norm_first = norm_second = 0.0
    for first, second in zip(first_values, second_values):
        difference = abs(first - second)
        absolute_sum += difference
        maximum = max(maximum, difference)
        exact_count += difference == 0.0
        dot += first * second
        norm_first += first * first
        norm_second += second * second
    cosine = dot / math.sqrt(norm_first * norm_second) if norm_first and norm_second else None
    return {
        "element_count": first_count,
        "exact_value_fraction": exact_count / first_count if first_count else 1.0,
        "mean_absolute_difference": absolute_sum / first_count if first_count else 0.0,
        "maximum_absolute_difference": maximum,
        "cosine_similarity": cosine,
        "first_representation": first_metadata.get("output_requested_type"),
        "second_representation": second_metadata.get("output_requested_type"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("iou-threshold must be between 0 and 1")

    first_manifest = manifest(args.first)
    second_manifest = manifest(args.second)
    contract_fields = (
        "input_layout", "input_type", "input_shape", "input_channel_order", "input_bytes",
        "source_width", "source_height", "letterbox_x_pad", "letterbox_y_pad",
        "letterbox_scale",
    )
    contract_mismatches = {
        key: [first_manifest.get(key), second_manifest.get(key)]
        for key in contract_fields if first_manifest.get(key) != second_manifest.get(key)
    }
    first_tensor = (args.first / first_manifest["input_file"]).read_bytes()
    second_tensor = (args.second / second_manifest["input_file"]).read_bytes()
    if len(first_tensor) != len(second_tensor):
        parser.error("input tensor byte counts differ")
    differences = [abs(a - b) for a, b in zip(first_tensor, second_tensor)]
    source_sequence_matches = first_manifest.get("source_sequence") == second_manifest.get("source_sequence")
    try:
        source_pts_matches = abs(
            float(first_manifest["source_pts_ms"]) - float(second_manifest["source_pts_ms"])
        ) <= 0.01
    except (KeyError, ValueError):
        source_pts_matches = False
    output_tensors = {}
    output_indexes = sorted({
        key.split(".")[0] for key in first_manifest if key.startswith("output[") and key.endswith(".file")
    })
    for output_index in output_indexes:
        first_name = first_manifest.get(f"{output_index}.file")
        second_name = second_manifest.get(f"{output_index}.file")
        if first_name is None or second_name is None:
            output_tensors[output_index] = {"error": "output tensor file missing from manifest"}
            continue
        try:
            output_tensors[output_index] = compare_numeric_outputs(
                args.first / first_name, first_manifest,
                args.second / second_name, second_manifest, output_index,
            )
        except (KeyError, ValueError) as error:
            output_tensors[output_index] = {"comparison_skipped": str(error)}
    report = {
        "first_model": first_manifest.get("model"),
        "second_model": second_manifest.get("model"),
        "contract_mismatches": contract_mismatches,
        "model_input_interfaces": {
            key: [first_manifest.get(key), second_manifest.get(key)]
            for key in (
                "input_model_type", "input_model_format", "input_quantization",
                "input_scale", "input_zero_point",
            )
        },
        "same_source_frame": {
            "sequence_matches": source_sequence_matches,
            "pts_matches_within_0_01ms": source_pts_matches,
            "first_sequence": first_manifest.get("source_sequence"),
            "second_sequence": second_manifest.get("source_sequence"),
            "first_pts_ms": first_manifest.get("source_pts_ms"),
            "second_pts_ms": second_manifest.get("source_pts_ms"),
        },
        "input_tensor": {
            "bytes": len(first_tensor),
            "exact_byte_fraction": (
                sum(difference == 0 for difference in differences) / len(differences)
                if differences else 1.0
            ),
            "mean_absolute_byte_difference": (
                sum(differences) / len(differences) if differences else 0.0
            ),
            "maximum_absolute_byte_difference": max(differences, default=0),
        },
        "raw_output_tensors": output_tensors,
        "detections": compare_detections(
            load_detections(args.first),
            load_detections(args.second),
            args.iou_threshold,
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 1 if contract_mismatches or not source_sequence_matches or not source_pts_matches else 0


if __name__ == "__main__":
    raise SystemExit(main())
