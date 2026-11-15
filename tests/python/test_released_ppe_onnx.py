import importlib.util
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_released_ppe_onnx.py"
SPEC = importlib.util.spec_from_file_location("evaluate_released_ppe_onnx", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReleasedPpeOnnxTests(unittest.TestCase):
    def test_letterbox_centers_and_pads_with_114(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        boxed, scale, pad_x, pad_y = MODULE.letterbox(image, 640, 640)
        self.assertEqual(boxed.shape, (640, 640, 3))
        self.assertEqual(scale, 3.2)
        self.assertEqual((pad_x, pad_y), (0, 160))
        self.assertTrue(np.all(boxed[:160] == 114))
        self.assertTrue(np.all(boxed[160:480] == 0))

    def test_decode_maps_classes_and_nms_is_class_aware(self):
        output = np.zeros((1, 6, 12), dtype=np.float32)
        output[0, :4, :3] = np.array([
            [100, 100, 50, 50],
            [101, 101, 50, 50],
            [100, 100, 50, 50],
        ], dtype=np.float32).T
        output[0, 4:, 0] = [0.90, 0.05]
        output[0, 4:, 1] = [0.80, 0.10]
        output[0, 4:, 2] = [0.03, 0.95]
        detections = MODULE.decode_yolov8(
            output, 640, 640, 640, 640, 1.0, 0, 0,
            {0: 5, 1: 6}, 0.25, 0.45,
        )
        self.assertEqual(len(detections), 2)
        self.assertEqual({item["class_id"] for item in detections}, {5, 6})
        self.assertAlmostEqual(max(item["score"] for item in detections), 0.95)


if __name__ == "__main__":
    unittest.main()
