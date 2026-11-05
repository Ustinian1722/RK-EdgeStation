#!/usr/bin/env python3
"""Convert an ONNX model with a pinned RKNN-Toolkit2 installation."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from rknn.api import RKNN


def read_calibration_manifest(path: Path) -> list[str]:
    entries = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if not entries:
        raise ValueError(f"calibration manifest is empty: {path}")
    missing = [entry for entry in entries if not Path(entry).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} calibration files are missing: {preview}")
    return entries


def parse_channel_values(raw: str, option_name: str) -> list[float]:
    try:
        values = [float(value.strip()) for value in raw.split(",")]
    except ValueError as error:
        raise ValueError(f"{option_name} must be three comma-separated numbers") from error
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{option_name} must be three finite comma-separated numbers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-platform", default="rk3576")
    parser.add_argument("--dtype", choices=("fp", "int8"), default="fp")
    parser.add_argument("--input-mean", default="0,0,0",
                        help="Toolkit2 input mean per channel; default matches YOLOv8 exports")
    parser.add_argument("--input-std", default="255,255,255",
                        help="Toolkit2 input divisor per channel; set 1,1,1 for raw-pixel YOLOX exports")
    parser.add_argument("--dataset", type=Path,
                        help="one absolute image path per line; required for int8 calibration")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.onnx_model.is_file():
        parser.error(f"ONNX model does not exist: {args.onnx_model}")
    if args.dtype == "int8" and args.dataset is None:
        parser.error("--dataset is required for --dtype int8")
    if args.dataset is not None and not args.dataset.is_file():
        parser.error(f"dataset manifest does not exist: {args.dataset}")
    return args


def main() -> int:
    args = parse_args()
    mean_values = parse_channel_values(args.input_mean, "--input-mean")
    std_values = parse_channel_values(args.input_std, "--input-std")
    if any(value == 0.0 for value in std_values):
        raise ValueError("--input-std values must be non-zero")
    quantize = args.dtype == "int8"
    dataset_entries = read_calibration_manifest(args.dataset) if quantize else []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"toolkit_dtype={args.dtype}")
    print(f"target_platform={args.target_platform}")
    print(f"input_mean={mean_values}")
    print(f"input_std={std_values}")
    print(f"onnx_model={args.onnx_model.resolve()}")
    print(f"output={args.output.resolve()}")
    if quantize:
        print(f"calibration_dataset={args.dataset.resolve()}")
        print(f"calibration_images={len(dataset_entries)}")

    rknn = RKNN(verbose=args.verbose)
    try:
        print("--> Configure model")
        status = rknn.config(
            mean_values=[mean_values],
            std_values=[std_values],
            target_platform=args.target_platform,
        )
        if status != 0:
            print(f"rknn.config failed with code {status}", file=sys.stderr)
            return status

        print("--> Load ONNX")
        status = rknn.load_onnx(model=str(args.onnx_model.resolve()))
        if status != 0:
            print(f"rknn.load_onnx failed with code {status}", file=sys.stderr)
            return status

        print("--> Build RKNN")
        status = rknn.build(
            do_quantization=quantize,
            dataset=str(args.dataset.resolve()) if quantize else None,
        )
        if status != 0:
            print(f"rknn.build failed with code {status}", file=sys.stderr)
            return status

        print("--> Export RKNN")
        status = rknn.export_rknn(str(args.output.resolve()))
        if status != 0:
            print(f"rknn.export_rknn failed with code {status}", file=sys.stderr)
            return status
        print("conversion_status=success")
        return 0
    finally:
        rknn.release()


if __name__ == "__main__":
    raise SystemExit(main())
