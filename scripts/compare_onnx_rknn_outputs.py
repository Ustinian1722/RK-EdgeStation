#!/usr/bin/env python3
"""Compare first-frame RKNN float outputs against ONNX Runtime on the saved RGB input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def read_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_outputs(root: Path, manifest: dict[str, str], count: int) -> list[np.ndarray]:
    outputs = []
    for index in range(count):
        prefix = f"output[{index}]"
        shape = tuple(int(value) for value in manifest[f"{prefix}.shape"].split(","))
        path = root / manifest[f"{prefix}.file"]
        output = np.fromfile(path, dtype="<f4")
        expected = int(np.prod(shape))
        if output.size != expected:
            raise ValueError(f"{path} has {output.size} floats; expected {expected} for {shape}")
        outputs.append(output.reshape(shape))
    return outputs


def tensor_metrics(reference: np.ndarray, candidate: np.ndarray, index: int,
                   manifest: dict[str, str]) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"output {index} shape mismatch: ONNX {reference.shape}, RKNN {candidate.shape}")
    reference = reference.astype(np.float32, copy=False)
    candidate = candidate.astype(np.float32, copy=False)
    difference = candidate - reference
    ref_flat = reference.astype(np.float64, copy=False).ravel()
    candidate_flat = candidate.astype(np.float64, copy=False).ravel()
    cosine = float(np.dot(ref_flat, candidate_flat) /
                   (np.linalg.norm(ref_flat) * np.linalg.norm(candidate_flat) + 1e-12))
    mean_abs = float(np.mean(np.abs(difference), dtype=np.float64))
    rmse = float(np.sqrt(np.mean(np.square(difference, dtype=np.float64))))
    value_range = float(np.max(reference) - np.min(reference))
    prefix = f"output[{index}]"
    return {
        "index": index,
        "name": manifest[f"{prefix}.name"],
        "shape": list(reference.shape),
        "onnx_min": float(np.min(reference)),
        "onnx_max": float(np.max(reference)),
        "rknn_min": float(np.min(candidate)),
        "rknn_max": float(np.max(candidate)),
        "mean_absolute_error": mean_abs,
        "rmse": rmse,
        "max_absolute_error": float(np.max(np.abs(difference))),
        "range_normalized_rmse": rmse / max(value_range, 1e-12),
        "cosine_similarity": cosine,
    }


def decode_detections(outputs: list[np.ndarray], manifest: dict[str, str],
                      confidence_threshold: float, nms_threshold: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    model_height, model_width, _ = (int(value) for value in manifest["input_shape"].split(","))
    for branch in range(3):
        box_tensor = outputs[branch * 3][0]
        class_tensor = outputs[branch * 3 + 1][0]
        score_sum_tensor = outputs[branch * 3 + 2][0, 0]
        _, grid_height, grid_width = box_tensor.shape
        stride = model_height // grid_height
        dfl_length = box_tensor.shape[0] // 4
        distributions = box_tensor.reshape(4, dfl_length, grid_height, grid_width)
        shifted = distributions - np.max(distributions, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        bins = np.arange(dfl_length, dtype=np.float32).reshape(1, dfl_length, 1, 1)
        distances = np.sum(probabilities * bins, axis=1)

        best_scores = np.max(class_tensor, axis=0)
        best_classes = np.argmax(class_tensor, axis=0)
        keep = (best_scores > confidence_threshold) & (score_sum_tensor >= confidence_threshold)
        for row, column in np.argwhere(keep):
            left = (-distances[0, row, column] + column + 0.5) * stride
            top = (-distances[1, row, column] + row + 0.5) * stride
            right = (distances[2, row, column] + column + 0.5) * stride
            bottom = (distances[3, row, column] + row + 0.5) * stride
            candidates.append({
                "class_id": int(best_classes[row, column]),
                "confidence": float(best_scores[row, column]),
                "box_model": [float(left), float(top), float(right), float(bottom)],
            })

    def inclusive_iou(first: list[float], second: list[float]) -> float:
        intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]) + 1.0)
        intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]) + 1.0)
        intersection = intersection_width * intersection_height
        first_area = max(0.0, first[2] - first[0] + 1.0) * max(0.0, first[3] - first[1] + 1.0)
        second_area = max(0.0, second[2] - second[0] + 1.0) * max(0.0, second[3] - second[1] + 1.0)
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        suppressed = any(
            kept["class_id"] == candidate["class_id"] and
            inclusive_iou(kept["box_model"], candidate["box_model"]) > nms_threshold
            for kept in selected
        )
        if not suppressed:
            selected.append(candidate)
            if len(selected) >= 128:
                break

    source_width = int(manifest["source_width"])
    source_height = int(manifest["source_height"])
    pad_x = float(manifest["letterbox_x_pad"])
    pad_y = float(manifest["letterbox_y_pad"])
    scale = float(manifest["letterbox_scale"])
    for detection in selected:
        left, top, right, bottom = detection.pop("box_model")
        box = [left - pad_x, top - pad_y, right - pad_x, bottom - pad_y]
        box = [max(0.0, min(float(model_width), box[0])),
               max(0.0, min(float(model_height), box[1])),
               max(0.0, min(float(model_width), box[2])),
               max(0.0, min(float(model_height), box[3]))]
        detection["box"] = [
            max(0, min(source_width, int(box[0] / scale))),
            max(0, min(source_height, int(box[1] / scale))),
            max(0, min(source_width, int(box[2] / scale))),
            max(0, min(source_height, int(box[3] / scale))),
        ]
    return selected


def box_iou(first: list[int], second: list[int]) -> float:
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def detection_agreement(first: list[dict[str, Any]], second: list[dict[str, Any]],
                        iou_threshold: float) -> dict[str, Any]:
    unmatched_second = set(range(len(second)))
    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for detection in sorted(first, key=lambda item: item["confidence"], reverse=True):
        candidates = [
            (box_iou(detection["box"], second[index]["box"]), index)
            for index in unmatched_second
            if detection["class_id"] == second[index]["class_id"]
        ]
        if not candidates:
            continue
        overlap, index = max(candidates)
        if overlap >= iou_threshold:
            unmatched_second.remove(index)
            matches.append((detection, second[index], overlap))
    match_count = len(matches)
    denominator = len(first) + len(second)
    return {
        "iou_threshold": iou_threshold,
        "first_count": len(first),
        "second_count": len(second),
        "matched_count": match_count,
        "unmatched_first_count": len(first) - match_count,
        "unmatched_second_count": len(second) - match_count,
        "set_f1_agreement": 2 * match_count / denominator if denominator else 1.0,
        "matched_mean_iou": float(np.mean([row[2] for row in matches])) if matches else None,
        "matched_mean_absolute_confidence_delta": (
            float(np.mean([abs(row[0]["confidence"] - row[1]["confidence"]) for row in matches]))
            if matches else None
        ),
    }


def read_cpp_detections(path: Path) -> list[dict[str, Any]]:
    import csv

    result = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result.append({
                "class_id": int(row["class_id"]),
                "confidence": float(row["confidence"]),
                "box": [int(row[key]) for key in ("left", "top", "right", "bottom")],
            })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-model", required=True, type=Path)
    parser.add_argument("--dump-dir", required=True, type=Path,
                        help="Directory produced by --dump-first-rknn-output")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.dump_dir.resolve()
    manifest = read_manifest(root / "tensor-manifest.txt")
    nhwc_shape = tuple(int(value) for value in manifest["input_shape"].split(","))
    rgb = np.fromfile(root / manifest["input_file"], dtype=np.uint8)
    if rgb.size != int(np.prod(nhwc_shape)):
        raise ValueError("saved RGB input size does not match the tensor manifest")
    rgb = rgb.reshape(nhwc_shape)
    nchw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]

    session = ort.InferenceSession(str(args.onnx_model), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {session.get_inputs()[0].name: nchw})
    rknn_outputs = read_outputs(root, manifest, len(onnx_outputs))
    tensor_results = [tensor_metrics(reference, candidate, index, manifest)
                      for index, (reference, candidate) in enumerate(zip(onnx_outputs, rknn_outputs))]

    onnx_detections = decode_detections(onnx_outputs, manifest, args.confidence, args.nms_threshold)
    rknn_detections = decode_detections(rknn_outputs, manifest, args.confidence, args.nms_threshold)
    report: dict[str, Any] = {
        "onnx_model_filename": args.onnx_model.name,
        "onnx_sha256": sha256(args.onnx_model),
        "rknn_model": manifest["model"],
        "rknn_original_tensor_types": sorted({
            manifest[f"output[{index}].type"] for index in range(len(onnx_outputs))
        }),
        "rknn_dump_dtype": manifest["output_requested_type"],
        "input_rgb_sha256": sha256(root / manifest["input_file"]),
        "onnxruntime_version": ort.__version__,
        "input_shape_nhwc": list(nhwc_shape),
        "input_normalization": "RGB float32 / 255, NHWC to NCHW",
        "confidence_threshold": args.confidence,
        "nms_threshold": args.nms_threshold,
        "detection_match_iou_threshold": args.match_iou,
        "tensor_comparison": tensor_results,
        "detection_agreement": detection_agreement(onnx_detections, rknn_detections,
                                                    args.match_iou),
        "onnx_detections": onnx_detections,
        "rknn_detections": rknn_detections,
    }
    cpp_path = root / "rknn-detections.csv"
    if cpp_path.exists():
        cpp_detections = read_cpp_detections(cpp_path)
        report["rknn_python_vs_cpp_postprocess"] = detection_agreement(
            cpp_detections, rknn_detections, 0.90)
        report["cpp_detections"] = cpp_detections

    output_path = args.output or root / "onnx-rknn-comparison.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
