# PPE model and event acceptance

**Status:** The separate PPE/person/forklift model converts to RKNN and runs on the KICKPI K7 / RK3576. The model-to-event/API loop passed two fixed-frame board replays. This is a prototype acceptance result; the data does not establish independent deployment accuracy.

## Dataset and training

The experiment used the Ultralytics Construction-PPE dataset under CC BY 4.0: 1,000 images, 800 train and 200 validation, with 14,863 boxes across 12 classes. The release says human-collected and generated images are mixed, but does not identify generated images individually; their fraction is unknown. There is no separate test split. See the [dataset audit](evidence/ppe/construction-ppe-audit.json), [training protocol](TRAINING_PPE.md), and [per-epoch metrics](evidence/ppe/training-epoch-metrics.jsonl).

The Apache-2.0 YOLOX-S experiment used upstream commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`, froze the COCO-pretrained feature extractor, and trained the feature-pyramid neck and 12-class head in FP32. Training ran on the Windows RTX 5060 (8 GB) with PyTorch 2.9.0+cu130, 640-pixel input, batch size 1, and seed 42. The constrained run logged 27 complete validation epochs (indices 0–26) before stopping short of the 40-epoch budget. Best validation was at epoch index 26: **47.22 AP50:95 and 72.51 AP50** on the development validation split.

| Class | AP50:95 |
|---|---:|
| helmet | 59.70 |
| gloves | 56.12 |
| vest | 67.90 |
| boots | 49.82 |
| goggles | 55.07 |
| Person | 69.06 |
| no_helmet | 34.08 |
| no_goggle | 9.76 |
| no_gloves | 33.93 |
| no_boots | 48.64 |
| forklift | 41.09 |
| no_vest | 41.52 |

The violation classes are less reliable than the positive equipment classes, especially `no_goggle`; the current event rules intentionally cover only `no_helmet` and `no_vest`. The custom model remains an experimental candidate. It does not replace the COCO baseline.

## Conversion and board verification

ONNX export produced three raw YOLOX head maps at 640×640 (opset 12). RKNN-Toolkit2 2.0.0b0 in the isolated x86-64 WSL2 environment converted the graph for `rk3576` in non-quantized mode. Conversion is optional host tooling; the worker, RKNN C Runtime, video, API, and event storage all ran on the board.

The final input contract is BGR, raw 0–255 values. ONNX receives NCHW float32; the RKNN C worker supplies NHWC uint8 with zero mean and unit standard deviation. An earlier conversion with the generic `/255` normalization produced no useful detections; the converter now accepts explicit mean/std settings and the worker retains BGR for `yolox-ppe`. The corrected contract was rechecked against the saved board input.

| Artifact | SHA-256 |
|---|---|
| Best checkpoint | `e38fb877f9959fbc104407be31000135e7164b23aab7b01229b7017538e4d195` |
| ONNX | `27f99bea4cc115f8a5f45edb94230d408363ae3dd5dcff19745f5514d99ef711` |
| RKNN | `64cc93f556816286f17c9e21f69cd2f818c3602ef13dbb2d13ed48616f74482b` |
| Saved board input | `8e0aa023ef1fa5555ff7064fdf8182f89cb7c70b1bdb658599a76ba2c5fdd663` |

On that one board input, semantic BGR/raw-255 ONNX and RKNN maps had mean cosine similarity 0.999992 and mean range-normalized RMSE 0.000629. Decoded sets matched at IoU 0.5 for confidence 0.10 (12/12), 0.25 (11/11), and 0.50 (9/9). The C++ worker decoder also matched the Python decode 11/11, with mean IoU 0.9872. Full numeric output is in the [ONNX/RKNN comparison evidence](evidence/ppe/rk3576-yolox-onnx-rknn-comparison.json). This checks conversion and decoder consistency for one saved input; it is not detector accuracy.

For the one positive source image, the board's 0.25-threshold detections matched all 9 labeled objects and added 2 extra `boots` detections (precision 81.8%, recall 100% for that single frame). Those two extras are outside the current helmet/vest alert scope. This one-image result is not a generalization estimate.

## Event rules and board API runs

The event state machine is independent of the detector backend. A `no_helmet` or `no_vest` detection triggers a PPE alert and carries either a matched `Person` track or `person_association="unknown"` when no person box contains it. Person-level status is counted as a violation only when a violation detection is associated; it is compliant only after both positive `helmet` and `vest` detections are associated. Other observed people remain `unknown`; absence of a violation is not treated as compliance. Other dataset classes remain visible detections but do not trigger PPE violation events.

The final board build passed CTest 2/2 and the board Python suite 9/9. The [build/test record](evidence/ppe/rk3576-board-build-test.txt) includes the candidate worker hash and confirms that no service or benchmark process remained running afterward.

The same worker also supports normalized danger/conflict zones, confirmed ROI entry/leave/dwell, forklift/person co-occurrence after configured persistence, severity labels, deduplication/cooldown, timestamped evidence images, SQLite history, event acknowledgement, and health/metrics/events APIs. Zone geometry, confirmation, dwell, and cooldown values are configured through the worker/service parameters. A conflict-zone event means sustained co-occurrence in the configured image region. It does not estimate physical distance from a monocular image.

Two 150-frame, 15-FPS files each repeat a single validation image. Both board services completed with exit code 0, storage available, and zero event-write failures. Health, metrics, and event-history requests returned HTTP 200. The positive replay emitted one associated `no_vest` violation, two configured-zone intrusions, and one forklift/person co-occurrence; the violation acknowledgement was stored in SQLite. The safe replay emitted zero PPE violations and counted its one person as compliant. See the [board API smoke record](evidence/ppe/rk3576-ppe-api-smoke.json).

These two replays establish that selected rules and the event/API persistence path work on the board. They do not establish temporal event precision/recall, tracking identity quality, or real-camera behavior. They use two images from the same development validation split.

## Performance and limitations

Three interleaved runs on the same 150-frame PPE clip measured a median **4.40 processed FPS** for YOLOX-S on RK3576, with median decode-to-result p95 312.3 ms and 105/150 dropped frames in latest-frame mode. The COCO YOLOv8n baseline on the same worker and input measured 8.27 FPS and 189.0 ms p95. The PPE model is materially slower; retain YOLOv8n as the fast COCO baseline and select YOLOX-S only when the PPE classes are required. The [four-group comparison report](PERFORMANCE_COMPARISON_REPORT.md) and raw [three-run board summary](evidence/ppe/rk3576-yolov8-coco-vs-yolox-ppe.json) include stage times, CPU, RSS, frame counts, and hashes.

The fixed-frame video is intentionally useful for repeatable board validation, but its event labels do not provide temporal ground truth. A larger independently labeled test set, moving PPE footage, multiple lighting/camera conditions, and optimization for the board's throughput are still needed before treating this model as a production detector.
