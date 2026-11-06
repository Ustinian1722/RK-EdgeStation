import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_ppe_rknn_predictions.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ppe_rknn_predictions", SCRIPT)
EVALUATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVALUATOR)


class PpeRknnEvaluatorTests(unittest.TestCase):
    def test_exact_prediction_has_full_ap_and_violation_recall(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "instances_val.json"
            predictions = root / "predictions.jsonl"
            annotations.write_text(json.dumps({
                "images": [{"id": 1, "file_name": "one.jpg"}],
                "categories": [{"id": index + 1, "name": name}
                               for index, name in enumerate(EVALUATOR.MODEL_CLASSES)],
                "annotations": [{"image_id": 1, "category_id": 7,
                                 "bbox": [10, 10, 20, 20]}],
            }), encoding="utf-8")
            predictions.write_text(json.dumps({
                "index": 0,
                "file_name": "one.jpg",
                "detections": [{"class_id": 6, "score": 0.9,
                                "bbox_xyxy": [10, 10, 30, 30]}],
            }) + "\n", encoding="utf-8")

            result = EVALUATOR.evaluate(annotations, predictions, 0.25)

        self.assertEqual(result["image_count"], 1)
        self.assertAlmostEqual(result["AP50"], 1.0)
        self.assertAlmostEqual(result["AP50_95"], 1.0)
        self.assertAlmostEqual(result["required_violation_class_recall"]["no_helmet"], 1.0)

    def test_duplicate_image_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "instances_val.json"
            predictions = root / "predictions.jsonl"
            annotations.write_text(json.dumps({
                "images": [{"id": 1, "file_name": "one.jpg"}],
                "categories": [{"id": index + 1, "name": name}
                               for index, name in enumerate(EVALUATOR.MODEL_CLASSES)],
                "annotations": [],
            }), encoding="utf-8")
            row = {"index": 0, "file_name": "one.jpg", "detections": []}
            predictions.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n",
                                   encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected"):
                EVALUATOR.evaluate(annotations, predictions, 0.25)


if __name__ == "__main__":
    unittest.main()
