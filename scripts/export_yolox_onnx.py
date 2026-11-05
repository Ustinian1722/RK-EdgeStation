#!/usr/bin/env python3
"""Export YOLOX-S to the three raw head maps expected by RKNN Model Zoo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


class RawHeadMaps(nn.Module):
    """Expose the stride-8/16/32 pre-decode tensors used by RKNN YOLOX."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        head = self.model.head
        features = self.model.backbone(images)
        outputs = []
        for index, feature in enumerate(features):
            shared = head.stems[index](feature)
            cls_feature = head.cls_convs[index](shared)
            reg_feature = head.reg_convs[index](shared)
            reg = head.reg_preds[index](reg_feature)
            obj = head.obj_preds[index](reg_feature).sigmoid()
            cls = head.cls_preds[index](cls_feature).sigmoid()
            outputs.append(torch.cat((reg, obj, cls), dim=1))
        return tuple(outputs)


def load_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("ema_model", "model", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint does not contain a state dictionary")
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    from yolox.exp import get_exp

    exp = get_exp(str(args.exp))
    model = exp.get_model().cpu().eval()
    state = load_state(args.checkpoint)
    model_state = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    if not compatible:
        raise ValueError("checkpoint has no parameter tensors matching the configured model")
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if len(compatible) < int(0.90 * len(model_state)):
        raise ValueError(
            f"only {len(compatible)}/{len(model_state)} tensors loaded; "
            "refusing to export an incomplete model"
        )

    model.head.decode_in_inference = False
    wrapper = RawHeadMaps(model).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, args.size, args.size, dtype=torch.float32)
    with torch.inference_mode():
        shapes = [list(tensor.shape) for tensor in wrapper(dummy)]
    expected = [[1, 17, 80, 80], [1, 17, 40, 40], [1, 17, 20, 20]]
    if shapes != expected:
        raise ValueError(f"raw head shapes {shapes} do not match expected {expected}")

    torch.onnx.export(
        wrapper,
        dummy,
        str(args.output),
        input_names=["images"],
        output_names=["head_s8", "head_s16", "head_s32"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    import onnx

    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "loaded_tensors": len(compatible),
        "model_tensors": len(model_state),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
        "input_shape": [1, 3, args.size, args.size],
        "input_contract": {
            "source_color_order": "BGR (OpenCV training and validation input)",
            "source_value_range": "0..255",
            "onnx_layout": "NCHW float32 without normalization",
            "rknn_input": "NHWC uint8 BGR with Toolkit2 std_values 1,1,1",
        },
        "output_names": ["head_s8", "head_s16", "head_s32"],
        "output_shapes": shapes,
        "onnx_opset": args.opset,
    }
    report_path = args.output.with_suffix(".export.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
