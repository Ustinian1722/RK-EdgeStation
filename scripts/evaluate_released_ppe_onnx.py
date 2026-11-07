#!/usr/bin/env python3
"""Run a released YOLOv8 PPE ONNX model on a shared YOLO-label image split.

This is a host-side model-selection evaluator. It uses the exported Ultralytics
letterbox contract (RGB, float32, 0..1) and emits the prediction JSONL consumed
by evaluate_ppe_rknn_predictions.py after mapping candidate labels to the
project's canonical YOLOX-PPE class IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = int(round(source_width * scale))
    resized_height = int(round(source_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (width - resized_width) // 2
    pad_y = (height - resized_height) // 2
    output = np.full((height, width, 3), 114, dtype=np.uint8)
    output[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    return output, scale, pad_x, pad_y


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    max_detections: int = 300,
) -> list[int]:
    kept: list[int] = []
    for class_id in np.unique(classes):
        class_indices = np.flatnonzero(classes == class_id)
        xywh = [
            [float(boxes[i, 0]), float(boxes[i, 1]),
             float(boxes[i, 2] - boxes[i, 0]), float(boxes[i, 3] - boxes[i, 1])]
            for i in class_indices
        ]
        selected = cv2.dnn.NMSBoxes(
            xywh, scores[class_indices].astype(float).tolist(), 0.001, iou_threshold
        )
        if len(selected):
            kept.extend(int(class_indices[int(index)]) for index in np.asarray(selected).reshape(-1))
    kept.sort(key=lambda index: float(scores[index]), reverse=True)
    return kept[:max_detections]


def decode_yolov8(
    output: np.ndarray,
    source_width: int,
    source_height: int,
    input_width: int,
    input_height: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    class_map: dict[int, int],
    score_floor: float,
    nms_iou: float,
) -> list[dict[str, Any]]:
    prediction = np.asarray(output, dtype=np.float32)
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f"unsupported YOLOv8 output rank/shape: {prediction.shape}")
    # Ultralytics export uses [batch, xywh + class scores, anchors].
    if prediction.shape[0] < prediction.shape[1] and prediction.shape[0] <= 128:
        prediction = prediction.T
    if prediction.shape[1] < 5:
        raise ValueError(f"YOLOv8 output has too few channels: {prediction.shape}")

    class_scores = prediction[:, 4:]
    source_class = np.argmax(class_scores, axis=1).astype(np.int32)
    scores = class_scores[np.arange(class_scores.shape[0]), source_class]
    supported = np.fromiter((int(c) in class_map for c in source_class), dtype=bool,
                            count=source_class.size)
    keep = np.flatnonzero((scores >= score_floor) & supported)
    if not keep.size:
        return []

    centers = prediction[keep, :2]
    sizes = prediction[keep, 2:4]
    boxes = np.column_stack((centers - sizes / 2.0, centers + sizes / 2.0))
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, source_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, source_height)
    mapped_classes = np.fromiter(
        (class_map[int(source_class[i])] for i in keep), dtype=np.int32, count=keep.size
    )
    selected = class_aware_nms(boxes, scores[keep], mapped_classes, nms_iou)
    return [
        {
            "class_id": int(mapped_classes[i]),
            "score": float(scores[keep[i]]),
            "bbox_xyxy": [round(float(value), 3) for value in boxes[i]],
        }
        for i in selected
    ]


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def evaluate_images(
    model_path: Path,
    image_dir: Path,
    class_map: dict[int, int],
    predictions_path: Path,
    report_path: Path,
    score_floor: float,
    nms_iou: float,
) -> dict[str, Any]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise ValueError(f"expected fixed 1x3xHxW model input, got {shape}")
    input_height, input_width = int(shape[2]), int(shape[3])
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise ValueError(f"no JPEG evaluation images found under {image_dir}")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    started = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8", newline="\n") as stream:
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to decode {image_path}")
            height, width = image.shape[:2]
            t0 = time.perf_counter()
            boxed, scale, pad_x, pad_y = letterbox(image, input_width, input_height)
            rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
            tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
            preprocess_ms.append((time.perf_counter() - t0) * 1000)
            t1 = time.perf_counter()
            output = session.run(None, {input_meta.name: tensor})[0]
            inference_ms.append((time.perf_counter() - t1) * 1000)
            t2 = time.perf_counter()
            detections = decode_yolov8(
                output, width, height, input_width, input_height, scale, pad_x, pad_y,
                class_map, score_floor, nms_iou,
            )
            postprocess_ms.append((time.perf_counter() - t2) * 1000)
            stream.write(json.dumps({"file_name": image_path.name, "detections": detections},
                                    separators=(",", ":")) + "\n")
    elapsed = time.perf_counter() - started
    report = {
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "input": {"shape": shape, "dtype": input_meta.type, "color_order": "RGB",
                  "range": "float32 0..1", "letterbox_color": 114},
        "outputs": [{"name": item.name, "shape": item.shape, "dtype": item.type}
                    for item in session.get_outputs()],
        "class_map_source_to_canonical": {str(k): v for k, v in sorted(class_map.items())},
        "image_count": len(images),
        "score_floor": score_floor,
        "nms_iou": nms_iou,
        "max_detections": 300,
        "wall_ms_per_image": {
            "preprocess_mean": statistics.fmean(preprocess_ms),
            "preprocess_p95": percentile(preprocess_ms, 95),
            "inference_mean": statistics.fmean(inference_ms),
            "inference_p50": percentile(inference_ms, 50),
            "inference_p95": percentile(inference_ms, 95),
            "postprocess_mean": statistics.fmean(postprocess_ms),
            "postprocess_p95": percentile(postprocess_ms, 95),
        },
        "wall_seconds": elapsed,
        "images_per_second": len(images) / elapsed,
        "predictions_sha256": sha256(predictions_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def evaluate_video(
    model_path: Path,
    video_path: Path,
    class_map: dict[int, int],
    predictions_path: Path,
    report_path: Path,
    score_floor: float,
    nms_iou: float,
) -> dict[str, Any]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise ValueError(f"expected fixed 1x3xHxW model input, got {shape}")
    input_height, input_width = int(shape[2]), int(shape[3])
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"failed to open video {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    event_input_path = predictions_path.with_suffix(".event-input.csv")
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    started = time.perf_counter()
    frame_count = 0
    try:
        with predictions_path.open("w", encoding="utf-8", newline="\n") as prediction_stream, \
             event_input_path.open("w", encoding="utf-8", newline="") as event_stream:
            event_writer = csv.writer(event_stream, lineterminator="\n")
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                height, width = image.shape[:2]
                pts_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if not np.isfinite(pts_ms) or pts_ms < 0 or (frame_count > 0 and pts_ms == 0):
                    pts_ms = frame_count * 1000.0 / source_fps if source_fps > 0 else frame_count * 1000.0 / 15.0

                t0 = time.perf_counter()
                boxed, scale, pad_x, pad_y = letterbox(image, input_width, input_height)
                rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
                tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
                preprocess_ms.append((time.perf_counter() - t0) * 1000)

                t1 = time.perf_counter()
                output = session.run(None, {input_meta.name: tensor})[0]
                inference_ms.append((time.perf_counter() - t1) * 1000)

                t2 = time.perf_counter()
                detections = decode_yolov8(
                    output, width, height, input_width, input_height, scale, pad_x, pad_y,
                    class_map, score_floor, nms_iou,
                )
                postprocess_ms.append((time.perf_counter() - t2) * 1000)
                prediction_stream.write(json.dumps(
                    {"frame_index": frame_count, "source_pts_ms": round(pts_ms, 3),
                     "file_name": f"frame-{frame_count:06d}", "detections": detections},
                    separators=(",", ":")) + "\n")
                if detections:
                    for detection in detections:
                        x1, y1, x2, y2 = detection["bbox_xyxy"]
                        event_writer.writerow((frame_count, f"{pts_ms:.3f}", width, height,
                                               detection["class_id"], f"{detection['score']:.8f}",
                                               f"{x1:.3f}", f"{y1:.3f}", f"{x2:.3f}", f"{y2:.3f}"))
                else:
                    event_writer.writerow((frame_count, f"{pts_ms:.3f}", width, height,
                                           -1, 0, 0, 0, 0, 0))
                frame_count += 1
    finally:
        capture.release()
    if frame_count == 0:
        raise ValueError(f"video contains no decodable frames: {video_path}")

    elapsed = time.perf_counter() - started
    report = {
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "input": {"video": str(video_path), "video_sha256": sha256(video_path),
                  "source_fps": source_fps, "frames": frame_count,
                  "color_order": "RGB", "dtype": input_meta.type,
                  "range": "float32 0..1", "letterbox_color": 114,
                  "shape": shape},
        "outputs": [{"name": item.name, "shape": item.shape, "dtype": item.type}
                    for item in session.get_outputs()],
        "class_map_source_to_canonical": {str(k): v for k, v in sorted(class_map.items())},
        "score_floor": score_floor,
        "nms_iou": nms_iou,
        "max_detections": 300,
        "event_replay_input": str(event_input_path),
        "wall_ms_per_frame": {
            "preprocess_mean": statistics.fmean(preprocess_ms),
            "preprocess_p95": percentile(preprocess_ms, 95),
            "inference_mean": statistics.fmean(inference_ms),
            "inference_p50": percentile(inference_ms, 50),
            "inference_p95": percentile(inference_ms, 95),
            "postprocess_mean": statistics.fmean(postprocess_ms),
            "postprocess_p95": percentile(postprocess_ms, 95),
        },
        "wall_seconds": elapsed,
        "frames_per_second": frame_count / elapsed,
        "predictions_sha256": sha256(predictions_path),
        "event_replay_input_sha256": sha256(event_input_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--images", type=Path)
    input_group.add_argument("--video", type=Path)
    parser.add_argument("--class-map", required=True,
                        help="comma-separated source_id:canonical_id pairs, e.g. 0:5,1:0")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    args = parser.parse_args()
    class_map = {}
    for pair in args.class_map.split(","):
        source, target = pair.split(":", maxsplit=1)
        class_map[int(source)] = int(target)
    if not class_map or any(value < 0 for value in class_map.values()):
        parser.error("--class-map must contain non-negative source_id:canonical_id pairs")
    if args.video:
        result = evaluate_video(args.model, args.video, class_map, args.predictions,
                                args.report, args.score_floor, args.nms_iou)
        keys = ("frames_per_second", "input", "wall_ms_per_frame")
    else:
        result = evaluate_images(args.model, args.images, class_map, args.predictions,
                                 args.report, args.score_floor, args.nms_iou)
        keys = ("image_count", "images_per_second", "wall_ms_per_image")
    print(json.dumps({key: result[key] for key in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
