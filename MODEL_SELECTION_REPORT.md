# Released PPE Model Selection and RK3576 Validation

**Run date:** 2026-11-07

**Branch/base:** `research/dynamic-video-dmabuf-20261107` / `a0beba7cab484ad32af9a76f6cf8bb01bf9e5af2`

**Decision:** Keep the existing YOLOX-S PPE model as the production Worker model. Neither released candidate passed the common-class quality gate. Keep the current GStreamer/MPP/RGA/RKNN and shared-input options unchanged; no candidate was inserted into the production inference path.

## 1. Question and method

This study compares mature released PPE weights with the existing project model. It does not train or fine-tune a new model. The same image split, class mapping, score threshold, NMS, and event state machine were used wherever the formats allowed.

The detection comparison uses the existing 200-image `development validation` split and its labels. This split participated in selecting the original YOLOX-S FP checkpoint; it is not an independent test set. Candidate training datasets do not have image-level manifests or hashes available here, so overlap between their training data and this split cannot be excluded. Published mAP claims on other datasets are recorded only as provenance and are not compared numerically.

The common operating point is score confidence `0.25`, IoU `0.50`; AP is 101-point interpolated per class and AP50:95 averages IoU 0.50–0.95 in 0.05 increments. The common five classes are `Person`, `helmet`, `vest`, `no_helmet`, and `no_vest`. The canonical Worker class IDs are 5, 0, 2, 6, and 11 respectively. Per-class predictions are class-aware NMS, IoU 0.45, at most 300 detections. Raw predictions and metric JSON are under [`evidence/model-selection-20261107/metrics`](evidence/model-selection-20261107/metrics/).

## 2. Candidate audit

| Model | Retrieved artifact and provenance | Input/output contract | License and scope |
|---|---|---|---|
| Existing project YOLOX-S PPE | Project trained model; 12 classes; dataset archive is the Construction-PPE export from `ultralytics/assets` v0.0.0, 800 train and 200 development-validation images. FP RKNN 20,710,742 B (`64cc93…74482b`); INT8 RKNN 11,566,602 B (`88b10a…2c502c`). | Existing PPE Worker contract is BGR, raw 0–255, NHWC uint8; decoded by the project YOLOX-S postprocessor. | Existing project model and dataset terms are documented in [PPE training](TRAINING_PPE.md). No independent test split is available. |
| Constructelligence v0.3 | Downloaded ONNX `buildvision-hazards-v0.3.onnx`, 44,766,819 B, SHA-256 `55f4dd2f53158eac4ff093d938b9fbda80c99454a03672145bbc36f539988249`, HF revision `79673cff5a8bb249fc8ec9cca3daa19e2ef95191`. YOLOv8s fine-tuned from COCO; card lists construction-safety sources. | RGB float32 NCHW `[1,3,640,640]`, range 0–1, letterbox fill 114; raw `[1,18,8400]` = four box coordinates plus 14 class scores; class-aware NMS required. The card says 14 classes while repository file metadata says 15; the ONNX tensor and README's 14-class order resolve the exported graph as 14. On RKNN the input is NHWC RGB uint8 0–255 using Toolkit2 mean 0/std 255. | HF marks weights AGPL-3.0 and training data CC BY 4.0. The card lists `construction-site-safety` and `construction-safety-gsnvb`; its own held-out test numbers are not used in our comparison. [Model card](https://huggingface.co/constructelligence/construction-site-safety-hazards) |
| Abdelrahman-cs19 v1.0.0 | Release tag commit `c03642b`, published 2026-09-12. Release contains `ppe_yolo.pt`, 22,521,315 B, SHA-256 `8f125b787557db8d9cb25d97305ebed2f30c9e63818131d0f8f947b9ed2ce9e8`. The release did not contain ONNX; it was exported locally with Ultralytics 8.4.171 to ONNX, 44,725,485 B, SHA-256 `abac0b618f38d3a887c1ab95c5756ab498575f6055cb4bd23bc9383cd8b1b1a7`. | RGB float32 NCHW `[1,3,640,640]`, range 0–1, letterbox fill 114; exported raw `[1,9,8400]` = four box coordinates plus five class scores; class-aware NMS required. Class order is helmet, no-helmet, no-vest, person, vest. | Repository code is MIT, but the release does not state a separate license for its checkpoint. We therefore do not assume the MIT code license grants redistribution rights to weights. The release describes its source only as a construction-site PPE dataset and gives no immutable training-image manifest. [Release](https://github.com/Abdelrahman-cs19/ppe-safety-detection/releases/tag/v1.0.0) · [repository license](https://github.com/Abdelrahman-cs19/ppe-safety-detection/blob/main/LICENSE) |

### Class mapping and unsupported classes

Constructelligence source IDs map as `0→Person(5)`, `1→helmet(0)`, `2→no_helmet(6)`, `3→vest(2)`, `4→no_vest(11)`, `6→gloves(1)`, `7→boots(3)`, `11→goggles(4)`, `12→no_goggle(7)`, and `13→no_gloves(8)`. Its excavator, wheel-loader, dump-truck, and no-mask outputs are not mapped into this Worker; it has no forklift or no-boots class. The GitHub release maps `0→helmet(0)`, `1→no_helmet(6)`, `2→no_vest(11)`, `3→Person(5)`, and `4→vest(2)`; gloves, goggles, boots, and forklift are absent. The five-class table below therefore excludes unsupported categories instead of treating their missing predictions as detector errors.

## 3. Same-split detection comparison

All rows below use the same 200 images and labels. AP values are the macro average over the common five classes. The original INT8 metric record was produced from a reserialized annotation copy whose file hash differs from the candidate/FP copy; the image split and boxes are the same. This is a serialization-provenance difference, not an independent evaluation set.

| Model / execution | Common-5 AP50 | Common-5 AP50:95 | `no_helmet` recall | `no_vest` recall |
|---|---:|---:|---:|---:|
| YOLOX-S FP, RK3576 | **0.7938** | **0.5376** | **0.5978** | **0.7308** |
| YOLOX-S INT8, RK3576 | 0.7900 | 0.5289 | 0.5978 | 0.7077 |
| Constructelligence ONNX, host | 0.5902 | 0.3408 | 0.1397 | 0.4538 |
| Constructelligence FP RKNN, RK3576 | 0.5886 | 0.3393 | 0.1397 | 0.4538 |
| Constructelligence INT8 RKNN, RK3576 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Abdelrahman ONNX, host | 0.5762 | 0.3025 | 0.1285 | 0.4769 |

These results reject both candidates for the current Worker: the existing model has substantially stronger person/equipment and violation-class recall on this split. The GitHub model's reported mAP50 0.93 / mAP50:95 0.51 and the HF card's test metrics are from their own datasets and are not evidence of superiority here.

On the same host CPU/ONNX Runtime environment, the 200-image wall rate was 9.326 images/s for Constructelligence and 9.361 images/s for the GitHub export. ONNX inference P50/P95 was 70.84/88.03 ms and 70.73/86.32 ms respectively; these host values are environment diagnostics, not RK3576 performance.

The existing dataset contains labeled forklift and additional PPE classes. These are not included in the common-five macro metrics. Candidate models missing a class cannot be assessed for that class by this comparison; the HF candidate also has no direct forklift replacement for the project's person/forklift rules.

## 4. RKNN Toolkit2 compatibility and conversion

The isolated host environment used RKNN-Toolkit2 `2.0.0b0+9bab5682`; the target used RKNN Runtime `2.0.0b0` on the existing RK3576 system. No board Runtime or driver upgrade was made.

Constructelligence ONNX converted to RKNN FP successfully. The board loaded the model and accepted the exact RGB uint8 input. Comparing host and board raw outputs on the same image gave cosine similarity `0.99999797`, raw MAE `0.04679`; at confidence 0.25, the three matched detections had mean box IoU `0.99521`. This establishes the FP export/input/output path, not model quality.

An INT8 build also completed using 20 calibration images selected to cover all 12 classes in the existing PPE dataset; the selected image hashes and class coverage are in [`calibration-selection.json`](evidence/model-selection-20261107/calibration-selection.json). Conversion used mean `[0,0,0]`, std `[255,255,255]`, target `rk3576`. It produced a 14,724,763 B RKNN (`d66b438547f71108897a9ec14dfa4f72248a17704c616cf6c1677d2c95b8dd76`). However, Runtime reports a shared per-tensor INT8 output scale of about `2.502` and zero-point `-128` for the tensor that mixes pixel-coordinate boxes and probabilities. The small class scores collapse to zero after the required output dequantization; all common-class AP/recall becomes zero. An initial `want_float=1` evaluation also returned zero detections; the corrected raw-INT8/dequantized evaluation still returned zero. Both failures are retained. This conversion is rejected; no threshold change was used to hide the quantization failure.

The generic conversion entry point remains [`scripts/convert_onnx_to_rknn.py`](scripts/convert_onnx_to_rknn.py). The host evaluation and board raw-output tools are [`scripts/evaluate_released_ppe_onnx.py`](scripts/evaluate_released_ppe_onnx.py), [`tools/rknn_yolov8_raw_bench.cpp`](tools/rknn_yolov8_raw_bench.cpp), and [`tools/replay_ppe_prediction_events.cpp`](tools/replay_ppe_prediction_events.cpp). We did not convert the weaker GitHub candidate to RKNN: its same-split ONNX score was lower than Constructelligence, so it did not pass the selection gate for board deployment work.

## 5. RK3576 candidate model-only performance

The three repeats use the same 30-image manifest, unpaced image input, host OpenCV preprocessing and board RKNN Runtime. They isolate candidate model/input/run/fetch costs; these are **not** Worker FPS, MPP/RGA end-to-end measurements, or real-time camera rates.

| Constructelligence RKNN | Median throughput | RKNN run P50 / P95 | CPU core / system | Peak RSS |
|---|---:|---:|---:|---:|
| FP | 3.3925 images/s | 184.485 / 202.464 ms | 58.22% / 7.28% | 161,276 KiB |
| INT8 | 5.1965 images/s | 113.256 / 122.373 ms | 65.43% / 8.18% | 122,712 KiB |

INT8 increased isolated throughput by about 53% and lowered peak RSS by about 24%, while increasing measured process CPU. Its detection outputs are unusable, so this is a speed/quality failure rather than an acceptable optimization. The 200-image board run was 3.3379 FPS FP and 5.2505 FPS INT8, with the same quality failure. Stage timing and three-repeat JSON are in [`board/`](evidence/model-selection-20261107/board/).

The PPE workload remains compute-bound on RK3576: candidate FP board `rknn_run` P50/P95 was about 191/213 ms; INT8 was 114/125 ms. These are model-only probe timings, not per-core NPU utilization. `/dev/rknpu*` utilization counters are unavailable on this board.

## 6. Dynamic video and event review

### Video provenance and labels

1. **Haringey real construction video:** Acabashi, Wikimedia Commons, CC BY-SA 4.0; 34 seconds, 510 unique H.264 frames, 1280×720 at 15 FPS; derivative SHA-256 `e1fe2432f23d816e58d9f8d54770fb5cd994efc013d278ef90cda9306fe71d2b`. This is real construction work, not a factory line. The sparse manual review sees PPE-wearing workers, partial occlusion and coarse person entry/exit, but does not provide dense PPE labels or confirmed violation intervals. Source and attribution are recorded in [the source page](https://commons.wikimedia.org/wiki/File:Building_construction_Moira_Close_Broadwater_Farm_Haringey_2025_16.webm) and [prior annotation metadata](evidence/dynamic-video/manual-annotations.json).
2. **Independent short CC0 construction clip:** Akum20, Wikimedia Commons, CC0 1.0; source 5.243 seconds, 1080×1920; H.264 derivative 79 frames at 15 FPS and 640×1138, SHA-256 `8b10516bdc76a5bac2601a2e6afbf0286e0c050094cabeea99523d6abdd9d961`. The foreground worker visibly lacks hardhat and vest. Other workers and all frame-level PPE states were not exhaustively labeled. [Source and license](https://commons.wikimedia.org/wiki/File:Men_at_work_in_building_site.webm)

Video frames were evaluated on the host ONNX Runtime CPU path with confidence 0.25 and NMS IoU 0.45. The generated per-frame detections were replayed through the production C++ `IoUTracker` and `PPEEventMachine` at three-frame confirmation and 30-second cooldown. This is a detector plus production event-rule comparison, not RK3576 inference. Candidate weights were not connected to the live Worker because they failed the image quality gate.

### Candidate host-video event replay

| Video / candidate | Event output | Manual interpretation |
|---|---|---|
| Haringey / Constructelligence | 1 `no_vest` at 8.267 s | Apparent false alert: sampled frame shows high-visibility clothing. No true violation interval is annotated. |
| Haringey / Abdelrahman | 1 `no_vest` at 8.200 s | Apparent false alert on the visible hi-vis worker; no full-video PPE truth. |
| CC0 / Constructelligence | 4 events: three `no_vest` at 0.133, 1.533, 4.333 s; one `no_helmet` at 1.667 s | The foreground target's visible missing PPE is detected at least once for each class. Three no-vest events are associated with changing track IDs; because identity labels are sparse, do not count these as three independent people or as a precise repeat-alert rate. |
| CC0 / Abdelrahman | 2 `no_vest` at 0.133 and 1.467 s; no `no_helmet` event | The foreground target has visible missing helmet and vest; the no-helmet event is absent in this replay. This is a qualitative miss on a short clip, not an estimated recall. |

The raw detector outputs, event-input CSV files, event JSONL and host timing summaries are preserved under [`dynamic/host/`](evidence/model-selection-20261107/dynamic/host/). Host throughput is unpaced and must not be compared to board media-paced Worker throughput.

### Current Worker regression runs after event-rule fixes

The current patched Worker ran the Haringey clip on RK3576 using existing YOLOX-S PPE INT8, GStreamer/MPP + RGA, safe video-memory `copy` mode, RKNN shared input, media pacing and a latest-frame queue of capacity 1. It decoded 510 frames, processed 272 and dropped 238; effective processing was 7.947 FPS at a 15-FPS source. Decode-to-result latency was P50/P95 158.545/189.590 ms; process CPU was 58.345% core-equivalent / 7.293% system, peak RSS 122,156 KiB. Explicit source-copy accounting was 705,024,000 B. The run emitted two PPE events: an unknown/partly occluded `no_helmet` at 1.667 s and an apparent false `no_vest` at 7.067 s. It generated 10 ROI entries and every entry was class ID 5 (`Person`). The old class-agnostic run's 191 mixed-class ROI entries and repeated false no-vest alerts remain in the dynamic report as historical pre-fix evidence.

The independent CC0 clip also ran through the current Worker with YOLOX-S FP, the same MPP/RGA copy path, shared input, media pacing and latest queue capacity 1. It decoded 79 frames, processed 23, dropped 56 and reached 4.186 FPS. Decode-to-result P50/P95 was 269.415/322.789 ms; CPU 66.360% core-equivalent / 8.295% system; peak RSS 130,468 KiB. Four PPE events occurred: two on the foreground target matched the visible no-vest/no-helmet evidence, and two other-worker no-vest detections remain unknown. Nine ROI entries were Person-only. The severe frame drop and tiny clip make this a functional diagnostic, not an accuracy benchmark.

Both Worker runs used ordinary copy mode, not the experimental DMA-BUF-retained route. These runs preserve the existing media pipeline and validate event fixes; they do not claim a new low-copy or performance improvement.

## 7. Business fixes

- `RoiEventMachine` now accepts the configured Person class ID and ignores helmet, vest, and violation boxes. The default COCO path uses class 0; the PPE Worker passes class 5. A regression test mixes Person and equipment boxes and asserts that only Person can create ROI occupancy/enter events.
- PPE violation confirmation/cooldown keys use `(associated Person Track ID, violation class)` when association succeeds. Unassociated detections fall back to their own violation Track ID. Different violation classes remain independent. The same Person/class cooldown survives violation-box ID churn while that Person track remains active. Regression tests cover track churn and separate classes.
- Stream reset, queue behavior, media backends, RGA paths, RKNN input configuration, and event/business feature set were not redesigned in this task.

## 8. Final selection and limitations

| Decision | Reason |
|---|---|
| Retain YOLOX-S PPE FP/INT8 as existing selectable Worker model | Best common-five development-set quality; it already supports forklift and the full project's class map. Its own INT8 quality/speed tradeoff remains documented separately. |
| Do not integrate Constructelligence v0.3 | FP conversion works, but common-five quality is substantially lower; INT8 output quantization destroys class scores. AGPL-3.0 weights also require license review before product redistribution. |
| Do not integrate Abdelrahman v1.0.0 | Lower common-five quality and no forklift/extra PPE classes. The code is MIT, but the checkpoint has no separately stated license; do not redistribute pending clarification. |
| Keep current MPP/RGA/RKNN pipeline | Candidate model selection did not justify changing or replacing the working pipeline. No new claim is made about copies or speed from these model tests. |

The two videos are short, sparse and construction-site footage. Candidate event replay is host-side; board Worker results are from the existing YOLOX-S only. No independent factory video set, dense PPE labels, forklift motion clip, external test split, candidate integrated Worker result, or production safety validation exists. Unlabeled frames are not treated as compliant/violating ground truth. No model weights or video assets are committed.

## 9. Reproduction and evidence index

Use the Toolkit2 2.0.0b0 environment and acquire external weights under their published terms. Keep model/data files local. `scripts/evaluate_released_ppe_onnx.py` accepts either the common image manifest or a video and writes predictions, a compact timing report and event replay CSV. `tools/replay_ppe_prediction_events.cpp` applies the production tracker/event rules to that CSV. `tools/rknn_yolov8_raw_bench.cpp` runs raw candidate RKNN outputs on board. The existing `scripts/convert_onnx_to_rknn.py` supports FP or representative INT8 conversion.

The current branch configured and built the event replay target plus `edge_core_tests` and `edge_media_tests` on the RK3576. Board CTest passed 2/2. Under the isolated Toolkit2 Python environment, `tests/python/test_released_ppe_onnx.py` passed 2/2. The board Worker also completed the two short dynamic clips described above. These checks establish build/runtime compatibility and regression behavior, not a PPE accuracy acceptance threshold.

Evidence:

- [`metrics/`](evidence/model-selection-20261107/metrics/) contains per-model predictions-derived metrics and input/output/inference summaries.
- [`board/`](evidence/model-selection-20261107/board/) contains three-repeat 30-image FP/INT8 records and 200-image raw benchmark records.
- [`calibration-selection.json`](evidence/model-selection-20261107/calibration-selection.json) records the 20 selected calibration image hashes and class coverage; images are not included.
- [`dynamic/host/`](evidence/model-selection-20261107/dynamic/host/) contains host candidate prediction, timing, and event replay artifacts for both clips.
- [`dynamic/moira-fixed.stdout.jsonl`](evidence/model-selection-20261107/dynamic/moira-fixed.stdout.jsonl) and [`dynamic/men-at-work-copy.stdout.jsonl`](evidence/model-selection-20261107/dynamic/men-at-work-copy.stdout.jsonl) contain patched board Worker records.
- [`scripts/evaluate_released_ppe_onnx.py`](scripts/evaluate_released_ppe_onnx.py), [`tools/replay_ppe_prediction_events.cpp`](tools/replay_ppe_prediction_events.cpp), and [`tools/rknn_yolov8_raw_bench.cpp`](tools/rknn_yolov8_raw_bench.cpp) are reusable evaluation/probe tools; they do not add candidate models to the Worker.
