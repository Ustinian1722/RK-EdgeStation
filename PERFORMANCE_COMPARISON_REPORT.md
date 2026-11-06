# RK3576 end-to-end comparison report

**Board:** KICKPI K7 / RK3576 / aarch64, RKNN Runtime 2.0.0b0, driver 0.9.7. The worker's end-to-end latency is decode-to-result age for a frame read from a local video. It is not camera-capture-to-display latency. The comparisons below include historical fixed-input A/Bs plus the integrated worker follow-up dated 2026-11-07.

## Flagship integrated pipeline follow-up (2026-11-07)

The current Worker supports OpenCV/FFmpeg or GStreamer/MPP and CPU or RGA preprocessing. Three-repeat board matrices used the same model/input for A0–A3 and separated no-pacing throughput from media-clock input. On the 10-FPS moving H.264 clip, OpenCV/CPU FP reached 7.656 FPS, MPP/RGA FP 7.981 FPS, and MPP/RGA INT8 9.991 FPS; respective P95 decode-to-result ages were 236.6, 220.6, and 106.3 ms. MPP/RGA FP used lower CPU/RSS with only a modest throughput change. PPE YOLOX-S INT8 increased Worker throughput from 4.593 to 7.975 FPS on the fixed 15-FPS positive replay; it reduced AP50:95 by 1.00 pp and `no_vest` recall by 2.31 pp on the development validation split. Full tables, stage data, raw records and limitations are in [FLAGSHIP_OPTIMIZATION_REPORT.md](FLAGSHIP_OPTIMIZATION_REPORT.md).

## Historical board-side comparison groups

| Group | Same-input design | Main result | Reading |
|---|---|---|---|
| Tracker association | Greedy IoU vs motion-predicted one-to-one association; 3 runs each, 150-frame moving clip, queue 150 | Every run processed 150/150 with zero drops and 1,194 detections. Entry events changed 33→29. Throughput ranges were 7.721–8.055 vs 7.602–8.044 FPS; p95 was 3.506–4.263 vs 3.571–4.614 s. RSS ranges were 134,476–171,992 vs 136,016–178,536 kB. | This is a repeatable event-output difference, not a tracking-accuracy score. Queue 150 creates seconds of backlog. CPU was not sampled in these historical runs. See [P1 tracker evidence](evidence/p1/yolov8n-rk3576-fp-tracker-ab-summary.txt). |
| YOLOv8n FP vs INT8 | Same graph and worker; 3 interleaved runs per model on a 150-frame H.264 clip; queue 1 and queue 150 | Queue 1: 7.602→9.986 FPS, 115/35→150/0 processed/dropped, 71.370→54.577 ms inference interval, and 236.099→115.343 ms p95. Median RSS was 138,716→119,444 kB. Queue 150: 7.965→9.984 FPS and 3,723.155→107.494 ms p95. | INT8 was faster on this workload, but the small single-scene calibration produced threshold-sensitive output differences and no labeled accuracy evidence. It remains experimental. See [P2 report](P2_ACCEPTANCE_REPORT.md). |
| Latest vs FIFO | Same FP worker, same 150-frame/10-FPS H.264 input; 3 interleaved runs per policy; capacity 1 | Both medians were 119/150 processed and 31 dropped. Throughput was 7.892 vs 7.859 FPS; p50 172.878 vs 199.284 ms; p95 232.028 vs 263.793 ms (latest/FIFO). | Latest reduced measured p50/p95 under this overload sample, with nearly equal throughput. CPU was not sampled in these historical runs. See the [queue evidence](evidence/p2/queue-policy-ab-summary.jsonl). |
| COCO YOLOv8n vs PPE YOLOX-S | Same current worker binary, same positive PPE clip (SHA-256 `a23d6d5c057f7c2eb52c79c883de1607d96b97f06eb645f06311b5f3bc6c439b`), confidence 0.25, capacity 1/latest, zones and event settings held fixed; 3 interleaved pairs in reference/candidate, candidate/reference, reference/candidate order | YOLOv8n: 8.271 FPS, 84/150 processed, 66 dropped, p50/p95 145.650/188.988 ms. YOLOX-S: 4.403 FPS, 45/150 processed, 105 dropped, p50/p95 262.723/312.263 ms. | PPE YOLOX-S has the required PPE classes but is slower and has a different class set/postprocess path. Detection counts are workload observations, not an accuracy or equivalence comparison. See the [raw board summary](evidence/ppe/rk3576-yolov8-coco-vs-yolox-ppe.json). |

## Stage and resource results for the PPE model comparison

The final comparison used the same C++ worker and video input for both models. Queue and zone settings were identical; the model family and model file were the intended independent variables. Worker order alternated between pairs to reduce ordering bias. Each value below is the median of three run-level values.

| Metric | YOLOv8n COCO | YOLOX-S PPE |
|---|---:|---:|
| Processed / decoded / dropped | 84 / 150 / 66 | 45 / 150 / 105 |
| Throughput | 8.271 FPS | 4.403 FPS |
| Last preprocess stage | 1.844 ms | 2.568 ms |
| Last RKNN inference + output fetch | 57.602 ms | 164.426 ms |
| Last postprocess stage | 0.904 ms | 1.938 ms |
| Last queue wait | 69.066 ms | 66.692 ms |
| Last decode-to-result sample | 186.292 ms | 271.160 ms |
| Per-run decode-to-result p50, summarized across runs | 145.650 ms | 262.723 ms |
| Per-run decode-to-result p95, summarized across runs | 188.988 ms | 312.263 ms |
| Worker metric RSS at run completion | 142,188 kB | 155,860 kB |
| `/proc` sampled average RSS | 133,459.6 KiB | 144,930.7 KiB |
| `/proc` sampled peak RSS (median of run peaks) | 141,800 KiB | 153,964 KiB |
| Process CPU average (one-core normalized) | 85.83% | 69.95% |
| Process CPU peak (one-core normalized) | 145.96% | 115.35% |

The process CPU values can exceed 100% when the process uses more than one core; they are not whole-board utilization. Stage values named `last_*` are the final worker metric sample from each run, then summarized across the three runs. The raw JSON keeps each run's measurements and model/video hashes.

The YOLOX-S median inference interval was about 2.85× the COCO model's; its p95 decode-to-result age was about 65% higher. Under the 15-FPS source, it processed about 45 frames per run and dropped about 105 with the latest-frame queue. This is the direct performance cost observed for the PPE model on this board. PPE versus COCO detections are not compared as accuracy because their label sets differ. Separate one-input ONNX/RKNN consistency checks and the two event smoke runs are documented in [PPE acceptance](PPE_ACCEPTANCE_REPORT.md).

## Scope of claims

- The FP/INT8 and tracker/queue reports predate process-level CPU sampling; they retain worker RSS where captured. CPU values are not reconstructed for those runs.
- The original MPP decode and RGA letterbox results below were stage-only. The 2026-11-07 follow-up integrated them into the Worker; its controlled end-to-end results are reported above and in the flagship report. No zero-copy path is claimed.
- Detection-output consistency is separate from event correctness. The PPE board output comparison covers one saved input. The event replays now have a source-label-grounded, clip-level presence score of 2/2 across one positive and one compliant static scene; they still do not form an independent labeled video test set or support temporal event precision/recall. See the [PPE event evaluation](evidence/ppe/event-level-replay-evaluation.json) and [acceptance report](PPE_ACCEPTANCE_REPORT.md).
- Keep the YOLOv8n COCO path as the fast general-object baseline. The PPE YOLOX-S path is a demonstrable, slower task-specific candidate with measured limits.
