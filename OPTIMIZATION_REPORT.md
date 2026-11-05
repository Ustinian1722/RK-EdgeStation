# Optimization report

Status: no video-pipeline or compute-speed optimization has been accepted. Initial RKNN/CPU-decode baselines and a separate tracker-association experiment were recorded on 2026-11-05; they do not establish a product-speed gain.

## P0 constraints discovered

- The board already exposes MPP 1.5.0, RGA 2.2.0, `/dev/mpp_service`, `/dev/rga`, and a system DMA heap. This establishes API/device presence only.
- RKNN Runtime is 2.0.0b0 and `/dev/rknpu*` nodes are absent. Both the official bundled MobileNet sample and the project YOLOv8n worker completed through the board C API. The runtime access path and direct NPU utilization are not traced.
- No physical camera was detected. A fixed local video file will be used for the first comparable pipeline run.
- User-space MPP/RGA calls and DMA-BUF availability do not establish hardware execution or zero-copy behavior.

## P1 smoke baseline (2026-11-05)

| Input / condition | Processed / decoded / dropped | Observed measurements | Limit |
|---|---:|---|---|
| 16-frame, 4-FPS MP4 made by repeating `bus.jpg`; latest queue capacity 1 | 16 / 16 / 0 | 3.986 processed FPS (file-paced); last RKNN run plus output-fetch interval 94.654 ms; end-to-end p50 149.770 ms / p95 189.583 ms; last preprocessing 4.411 ms; last postprocessing 1.059 ms | Smoke only; 16 repeated still frames; the reported throughput is paced by the source |
| 150-frame, 15-FPS MP4 made by repeating `bus.jpg`; latest queue capacity 1 | 88 / 150 / 62 | 8.726 processed FPS; end-to-end p50 141.266 ms / p95 184.429 ms; last RSS sample 130,472 kB | Bounded-load probe, not a live-motion or peak-throughput benchmark |
| Second 150-frame, 15-FPS repeated-still run with IRQ sampling | 91 / 150 / 59 | 9.028 processed FPS; end-to-end p50 143.433 ms / p95 173.631 ms; shared NPU/IOMMU IRQ aggregate +5,096 during run and 0 change in a two-second idle check | Activity correlation probe; not an accuracy or product benchmark |
| 150-frame moving sample (`vtest.avi`), 10 FPS; latest queue capacity 1 | 130 / 150 / 20 | 8.587 processed FPS; 1,033 detections; 42 enter events; end-to-end p50 165.068 ms / p95 223.086 ms; last RSS sample 132,228 kB | One sample only; 42 unique event track IDs indicate possible fragmentation; two low-confidence `skis` outputs overlap a pedestrian; no labeled accuracy or identity score |
| Tracker A/B: greedy last-box IoU vs velocity-predicted global association; 3 runs each, queue capacity 150 | 150 / 150 / 0 in every run | All runs reported 1,194 detections. Baseline emitted 33 enter events each time; candidate emitted 29. p95 range 3.506–4.263 s / 3.571–4.614 s and throughput range 7.721–8.055 / 7.602–8.044 FPS (baseline/candidate) | Repeatable 12.1% fewer entry events, but no GT identities; may be reduced fragmentation or incorrect merging. Queue backlog makes this a tracking association experiment, not a live-performance result |

The worker's inference interval includes `rknn_run` and `rknn_outputs_get`. The tracker candidate extrapolates box centers across missed frames, applies class/area gates, then uses globally sorted one-to-one IoU/center-distance candidates. A separate latest-frame/API smoke with this code completed 129/150 frames, dropped 21, and persisted 29 events. The controlled queue-150 comparison keeps every frame but raises p95 end-to-end delay into 3.5–4.6 seconds because queue wait accumulates; the live path therefore retains the capacity-1 latest-frame queue. No speed gain is claimed. Devfreq stayed at 300 MHz; `/sys/class/devfreq/27700000.npu/load` read `100@300000000Hz` during the run and after idle, so it is not treated as a utilization percentage. A later same-boot read of the dynamic run's shared NPU/IOMMU IRQ lines was lower than the captured in-run value; the counter discrepancy remains unresolved, and IRQ data is exploratory rather than a utilization measure. The shared lines do not report per-core utilization. Logs and metrics are in `evidence/p1/yolov8n-rk3576-fp-*`.

No optimized pipeline was compared. The current path uses OpenCV software decode/preprocessing and the RKNN C API; MPP, RGA, quantization, asynchronous inference, and DMA-BUF remain future controlled comparisons.

## Planned baseline matrix

| Stage | Reference | Candidate | Required evidence |
|---|---|---|---|
| Video decode | OpenCV/CPU decode | GStreamer MPP decode | Same file, codec, dimensions, timestamps; CPU, throughput and drops |
| Resize/color | OpenCV CPU | RGA | Pixel comparison, stride/format, operation time and full pipeline latency |
| Inference | RKNN C API single path | INT8 or asynchronous path | Matching outputs, NPU-only duration, end-to-end effect, memory and errors |
| Buffer transport | explicit CPU copy | imported DMA-BUF if compatible | fd/allocator ownership, cache sync, copy count, correctness and measured impact |
| Scheduling | FIFO | latest-frame policy | queue wait, frame age, drops, event freshness and bounded RSS |

Before changing an implementation, save its source commit and raw run data. An optimization is accepted only if it improves the named metric without violating output correctness or stability. Neutral and negative results remain in the report.
