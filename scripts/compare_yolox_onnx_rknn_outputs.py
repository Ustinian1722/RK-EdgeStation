#!/usr/bin/env python3
"""Compare saved RKNN YOLOX head maps with ONNX Runtime on the same input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_outputs(root: Path, manifest: dict[str, str], count: int) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for index in range(count):
        prefix = f"output[{index}]"
        shape = tuple(int(value) for value in manifest[f"{prefix}.shape"].split(","))
        output = np.fromfile(root / manifest[f"{prefix}.file"], dtype="<f4")
        expected = int(np.prod(shape))
        if output.size != expected:
            raise ValueError(f"{prefix} has {output.size} floats; expected {expected} for {shape}")
        outputs.append(output.reshape(shape))
    return outputs


def tensor_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"output shape mismatch: ONNX {reference.shape}, RKNN {candidate.shape}")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    reference64 = reference.astype(np.float64, copy=False).ravel()
    candidate64 = candidate.astype(np.float64, copy=False).ravel()
    cosine = float(np.dot(reference64, candidate64) /
                   (np.linalg.norm(reference64) * np.linalg.norm(candidate64) + 1e-12))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    value_range = float(np.max(reference64) - np.min(reference64))
    return {
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "max_absolute_error": float(np.max(np.abs(difference))),
        "range_normalized_rmse": rmse / max(value_range, 1e-12),
        "cosine_similarity": cosine,
        "reference_min": float(np.min(reference64)),
        "reference_max": float(np.max(reference64)),
        "candidate_min": float(np.min(candidate64)),
        "candidate_max": float(np.max(candidate64)),
    }


def decode_detections(outputs: list[np.ndarray], manifest: dict[str, str],
                      confidence_threshold: float, nms_threshold: float) -> list[dict[str, Any]]:
    input_height, input_width, channels = (int(value) for value in manifest["input_shape"].split(","))
    if channels != 3:
        raise ValueError(f"expected a three-channel input, got {channels}")
    source_width = int(manifest["source_width"])
    source_height = int(manifest["source_height"])
    pad_x = float(manifest["letterbox_x_pad"])
    pad_y = float(manifest["letterbox_y_pad"])
    scale = float(manifest["letterbox_scale"])
    candidates: list[dict[str, Any]] = []

    for tensor in outputs:
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(f"expected [1,C,H,W] YOLOX output, got {tensor.shape}")
        if tensor.shape[1] == 17:
            maps = tensor[0]
        elif tensor.shape[3] == 17:
            maps = tensor[0].transpose(2, 0, 1)
        else:
            raise ValueError(f"expected 17 YOLOX channels (4 box + objectness + 12 classes), got {tensor.shape}")
        _, grid_height, grid_width = maps.shape
        if grid_height <= 0 or grid_width <= 0 or input_height % grid_height or input_width % grid_width:
            raise ValueError(f"invalid YOLOX grid shape {maps.shape}")
        stride_y, stride_x = input_height // grid_height, input_width // grid_width
        if stride_y != stride_x:
            raise ValueError(f"non-square YOLOX stride {stride_x}x{stride_y}")
        stride = stride_x
        for y in range(grid_height):
            for x in range(grid_width):
                objectness = float(maps[4, y, x])
                class_scores = maps[5:, y, x]
                class_id = int(np.argmax(class_scores))
                score = objectness * float(class_scores[class_id])
                if not np.isfinite(score) or score < confidence_threshold:
                    continue
                tx, ty, tw, th = (float(maps[index, y, x]) for index in range(4))
                if not np.isfinite([tx, ty, tw, th]).all() or tw > 20.0 or th > 20.0:
                    continue
                center_x = (tx + x) * stride
                center_y = (ty + y) * stride
                width = float(np.exp(tw)) * stride
                height = float(np.exp(th)) * stride
                left = (center_x - width * 0.5 - pad_x) / scale
                top = (center_y - height * 0.5 - pad_y) / scale
                right = (center_x + width * 0.5 - pad_x) / scale
                bottom = (center_y + height * 0.5 - pad_y) / scale
                box = [
                    max(0.0, min(float(source_width), left)),
                    max(0.0, min(float(source_height), top)),
                    max(0.0, min(float(source_width), right)),
                    max(0.0, min(float(source_height), bottom)),
                ]
                if box[2] > box[0] and box[3] > box[1]:
                    candidates.append({"class_id": class_id, "confidence": score, "box": box})

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        suppressed = False
        for kept in selected:
            if kept["class_id"] != candidate["class_id"]:
                continue
            first, second = kept["box"], candidate["box"]
            iw = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
            ih = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
            intersection = iw * ih
            area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
            area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
            union = area_first + area_second - intersection
            if union > 0.0 and intersection / union > nms_threshold:
                suppressed = True
                break
        if not suppressed:
            selected.append(candidate)
            if len(selected) >= 128:
                break
    return selected


def box_iou(first: list[float], second: list[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0.0 else 0.0


def detection_agreement(first: list[dict[str, Any]], second: list[dict[str, Any]],
                        iou_threshold: float = 0.5) -> dict[str, Any]:
    unmatched = set(range(len(second)))
    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for detection in sorted(first, key=lambda item: item["confidence"], reverse=True):
        choices = [(box_iou(detection["box"], second[index]["box"]), index)
                   for index in unmatched if detection["class_id"] == second[index]["class_id"]]
        if choices:
            overlap, index = max(choices)
            if overlap >= iou_threshold:
                unmatched.remove(index)
                matches.append((detection, second[index], overlap))
    count = len(matches)
    denominator = len(first) + len(second)
    return {
        "first_count": len(first),
        "second_count": len(second),
        "matched_count": count,
        "unmatched_first_count": len(first) - count,
        "unmatched_second_count": len(second) - count,
        "set_f1_agreement": 2.0 * count / denominator if denominator else 1.0,
        "matched_mean_iou": float(np.mean([item[2] for item in matches])) if matches else None,
        "matched_mean_absolute_confidence_delta": (
            float(np.mean([abs(item[0]["confidence"] - item[1]["confidence"]) for item in matches]))
            if matches else None
        ),
    }


def read_cpp_detections(path: Path) -> list[dict[str, Any]]:
    import csv

    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append({
                "class_id": int(row["class_id"]),
                "confidence": float(row["confidence"]),
                "box": [float(row[key]) for key in ("left", "top", "right", "bottom")],
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-model", required=True, type=Path)
    parser.add_argument("--dump-dir", required=True, type=Path,
                        help="Directory produced by --dump-first-rknn-output")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="threshold used by the worker detection CSV; default 0.25")
    parser.add_argument("--confidence-sweep", default="0.10,0.25,0.50")
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.dump_dir.resolve()
    manifest = read_manifest(root / "tensor-manifest.txt")
    height, width, channels = (int(value) for value in manifest["input_shape"].split(","))
    input_bytes = np.fromfile(root / manifest["input_file"], dtype=np.uint8)
    expected_bytes = height * width * channels
    if input_bytes.size != expected_bytes:
        raise ValueError(f"saved input has {input_bytes.size} bytes; expected {expected_bytes}")
    nhwc = input_bytes.reshape(height, width, channels)
    stored_order = manifest.get("input_channel_order", "RGB").upper()
    if stored_order not in {"RGB", "BGR"}:
        raise ValueError(f"unsupported saved input channel order: {stored_order}")
    rknn_outputs = read_outputs(root, manifest, 3)
    session = ort.InferenceSession(str(args.onnx_model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    variants: dict[str, np.ndarray] = {}
    stored = nhwc.astype(np.float32)
    if stored_order == "RGB":
        rgb, bgr = stored, stored[:, :, ::-1].copy()
    else:
        bgr, rgb = stored, stored[:, :, ::-1].copy()
    for name, image, divisor in (
        ("rgb_0_255", rgb, 1.0), ("bgr_0_255", bgr, 1.0),
        ("rgb_0_1", rgb, 255.0), ("bgr_0_1", bgr, 255.0),
    ):
        variants[name] = np.transpose(image / divisor, (2, 0, 1))[None, ...].astype(np.float32)

    thresholds = [float(value.strip()) for value in args.confidence_sweep.split(",") if value.strip()]
    if not (np.isfinite(args.confidence) and 0.0 <= args.confidence <= 1.0):
        raise ValueError("default confidence threshold must be between 0 and 1")
    if not thresholds or any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("confidence thresholds must be finite numbers between 0 and 1")

    reports: dict[str, Any] = {}
    for name, tensor in variants.items():
        onnx_outputs = session.run(None, {input_name: tensor})
        if len(onnx_outputs) != len(rknn_outputs):
            raise ValueError(f"ONNX returned {len(onnx_outputs)} heads; RKNN saved {len(rknn_outputs)}")
        raw = [tensor_metrics(reference, candidate)
               for reference, candidate in zip(onnx_outputs, rknn_outputs)]
        detections = decode_detections(onnx_outputs, manifest, args.confidence, args.nms_threshold)
        rknn_detections = decode_detections(rknn_outputs, manifest, args.confidence, args.nms_threshold)
        sweep = []
        for threshold in thresholds:
            reference = decode_detections(onnx_outputs, manifest, threshold, args.nms_threshold)
            candidate = decode_detections(rknn_outputs, manifest, threshold, args.nms_threshold)
            sweep.append({
                "confidence_threshold": threshold,
                "agreement": detection_agreement(reference, candidate, args.match_iou),
                "onnx_detections": reference,
                "rknn_detections": candidate,
            })
        reports[name] = {
            "onnx_input_shape": list(tensor.shape),
            "default_confidence_threshold": args.confidence,
            "tensor_comparison": raw,
            "mean_cosine_similarity": float(np.mean([item["cosine_similarity"] for item in raw])),
            "mean_range_normalized_rmse": float(np.mean([item["range_normalized_rmse"] for item in raw])),
            "default_threshold_onnx_detections": detections,
            "default_threshold_rknn_detections": rknn_detections,
            "confidence_sweep": sweep,
        }

    cpp = read_cpp_detections(root / "rknn-detections.csv")
    rknn_default = decode_detections(rknn_outputs, manifest, args.confidence, args.nms_threshold)
    report = {
        "onnx_model": args.onnx_model.name,
        "onnx_sha256": sha256(args.onnx_model),
        "rknn_model": manifest.get("model"),
        "input_file_sha256": hashlib.sha256(input_bytes.tobytes()).hexdigest(),
        "saved_input": {
            "layout": manifest.get("input_layout"),
            "type": manifest.get("input_type"),
            "channel_order": manifest.get("input_channel_order", "unspecified"),
            "shape": [height, width, channels],
            "source_size": [int(manifest["source_width"]), int(manifest["source_height"])],
            "letterbox_scale": float(manifest["letterbox_scale"]),
            "letterbox_padding": [float(manifest["letterbox_x_pad"]), float(manifest["letterbox_y_pad"])],
        },
        "rknn_outputs": [{
            "name": manifest[f"output[{index}].name"],
            "shape": [int(value) for value in manifest[f"output[{index}].shape"].split(",")],
            "runtime_type": manifest[f"output[{index}].type"],
        } for index in range(3)],
        "thresholds": thresholds,
        "default_confidence_threshold": args.confidence,
        "nms_threshold": args.nms_threshold,
        "match_iou_threshold": args.match_iou,
        "input_variants": reports,
        "cpp_detection_count": len(cpp),
        "cpp_vs_rknn_decode": detection_agreement(cpp, rknn_default, args.match_iou),
        "interpretation": (
            "Numerical consistency on one saved board input only. Input variants diagnose channel order "
            "and scaling; this report is not labeled detector accuracy or event accuracy."
        ),
    }
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
