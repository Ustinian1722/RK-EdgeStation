#!/usr/bin/env python3
"""Compare detection sets from two saved RKNN output directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_onnx_rknn_outputs import (
    decode_detections,
    detection_agreement,
    read_manifest,
    read_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path, help="first saved RKNN output directory")
    parser.add_argument("--second", required=True, type=Path, help="second saved RKNN output directory")
    parser.add_argument("--first-label", default="first")
    parser.add_argument("--second-label", default="second")
    parser.add_argument("--confidence-sweep", default="0.25,0.345,0.50")
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    first_root = args.first.resolve()
    second_root = args.second.resolve()
    first_manifest = read_manifest(first_root / "tensor-manifest.txt")
    second_manifest = read_manifest(second_root / "tensor-manifest.txt")
    first_outputs = read_outputs(first_root, first_manifest, 9)
    second_outputs = read_outputs(second_root, second_manifest, 9)
    tensor_comparison = []
    for index, (first, second) in enumerate(zip(first_outputs, second_outputs)):
        difference = second.astype(np.float64) - first.astype(np.float64)
        first64 = first.astype(np.float64, copy=False).ravel()
        second64 = second.astype(np.float64, copy=False).ravel()
        tensor_comparison.append({
            "index": index,
            "name": first_manifest[f"output[{index}].name"],
            "shape": list(first.shape),
            "first_min": float(np.min(first)),
            "first_max": float(np.max(first)),
            "second_min": float(np.min(second)),
            "second_max": float(np.max(second)),
            "mean_absolute_difference": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "max_absolute_difference": float(np.max(np.abs(difference))),
            "cosine_similarity": float(np.dot(first64, second64) /
                                        (np.linalg.norm(first64) * np.linalg.norm(second64) + 1e-12)),
        })
    thresholds = [float(value.strip()) for value in args.confidence_sweep.split(",") if value.strip()]
    detection_comparisons = []
    for threshold in thresholds:
        first_detections = decode_detections(
            first_outputs, first_manifest, threshold, args.nms_threshold
        )
        second_detections = decode_detections(
            second_outputs, second_manifest, threshold, args.nms_threshold
        )
        detection_comparisons.append({
            "confidence_threshold": threshold,
            "agreement": detection_agreement(first_detections, second_detections, args.match_iou),
            "first_detections": first_detections,
            "second_detections": second_detections,
        })

    report = {
        "reference": args.first_label,
        "candidate": args.second_label,
        "model": "output tensors saved by the board-side RKNN comparison tool",
        "nms_threshold": args.nms_threshold,
        "detection_match_iou_threshold": args.match_iou,
        "tensor_comparison": tensor_comparison,
        "detection_comparisons": detection_comparisons,
        "interpretation": (
            "Preprocessing sensitivity comparison only. Agreement is not labeled detection accuracy; "
            "one source frame and one model execution per preprocessing path are represented."
        ),
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
