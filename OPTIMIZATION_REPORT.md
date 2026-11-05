# Optimization report

Status: no video-pipeline or compute-speed optimization has been accepted. P1 baselines and P2 MPP/RGA stage probes were recorded on 2026-11-05; the P2 measurements do not establish a product-speed gain.

## P0 constraints discovered

- The board already exposes MPP 1.5.0, RGA 2.2.0, `/dev/mpp_service`, `/dev/rga`, and a system DMA heap. This establishes API/device presence only.
- RKNN Runtime is 2.0.0b0 and `/dev/rknpu*` nodes are absent. Both the official bundled MobileNet sample and the project YOLOv8n worker completed through the board C API. The runtime access path and direct NPU utilization are not traced.
- No physical camera was detected. A fixed local video file will be used for the first comparable pipeline run.
- The P2 probes exercised GStreamer `mppvideodec` and `librga`; DMA-BUF availability does not establish zero-copy behavior.

## P1 smoke baseline (2026-11-05)

| Input / condition | Processed / decoded / dropped | Observed measurements | Limit |
|---|---:|---|---|
| 16-frame, 4-FPS MP4 made by repeating `bus.jpg`; latest queue capacity 1 | 16 / 16 / 0 | 3.986 processed FPS (file-paced); last RKNN run plus output-fetch interval 94.654 ms; end-to-end p50 149.770 ms / p95 189.583 ms; last preprocessing 4.411 ms; last postprocessing 1.059 ms | Smoke only; 16 repeated still frames; the reported throughput is paced by the source |
| 150-frame, 15-FPS MP4 made by repeating `bus.jpg`; latest queue capacity 1 | 88 / 150 / 62 | 8.726 processed FPS; end-to-end p50 141.266 ms / p95 184.429 ms; last RSS sample 130,472 kB | Bounded-load probe, not a live-motion or peak-throughput benchmark |
| Second 150-frame, 15-FPS repeated-still run with IRQ sampling | 91 / 150 / 59 | 9.028 processed FPS; end-to-end p50 143.433 ms / p95 173.631 ms; shared NPU/IOMMU IRQ aggregate +5,096 during run and 0 change in a two-second idle check | Activity correlation probe; not an accuracy or product benchmark |
| 150-frame moving sample (`vtest.avi`), 10 FPS; latest queue capacity 1 | 130 / 150 / 20 | 8.587 processed FPS; 1,033 detections; 42 enter events; end-to-end p50 165.068 ms / p95 223.086 ms; last RSS sample 132,228 kB | One sample only; 42 unique event track IDs indicate possible fragmentation; two low-confidence `skis` outputs overlap a pedestrian; no labeled accuracy or identity score |
| Tracker A/B: greedy last-box IoU vs velocity-predicted global association; 3 runs each, queue capacity 150 | 150 / 150 / 0 in every run | All runs reported 1,194 detections. Baseline emitted 33 enter events each time; candidate emitted 29. p95 range 3.506–4.263 s / 3.571–4.614 s and throughput range 7.721–8.055 / 7.602–8.044 FPS (baseline/candidate) | Repeatable 12.1% fewer entry events, but no GT identities; may be reduced fragmentation or incorrect merging. Queue backlog makes this a tracking association experiment, not a live-performance result |

The worker's inference interval includes `rknn_run` and `rknn_outputs_get`. The tracker candidate extrapolates box centers across missed frames, applies class/area gates, then uses globally sorted one-to-one IoU/center-distance candidates. A separate latest-frame/API smoke with this code completed 129/150 frames, dropped 21, and persisted 29 events. The controlled queue-150 comparison keeps every frame but raises p95 end-to-end delay into 3.5–4.6 seconds because queue wait accumulates; the live path therefore retains the capacity-1 latest-frame queue. No speed gain is claimed. Devfreq stayed at 300 MHz; `/sys/class/devfreq/27700000.npu/load` read `100@300000000Hz` during the run and after idle, so it is not treated as a utilization percentage. A later same-boot read of the dynamic run's shared NPU/IOMMU IRQ lines was lower than the captured in-run value; the counter discrepancy remains unresolved, and IRQ data is exploratory rather than a utilization measure. The shared lines do not report per-core utilization. Logs and metrics are in `evidence/p1/yolov8n-rk3576-fp-*`.

## P2 isolated media/preprocessing results (2026-11-05)

The 1,500-frame H.264 sample was decoded by GStreamer `mppvideodec` to NV12 at a median 1,683.2 FPS over three measured repetitions. This ends at a fakesink and excludes RGB conversion, resize, inference, and DMA-BUF import. Adding `videoconvert` to produce BGR reduced the GStreamer path to 162.7 FPS median. OpenCV FFmpeg produced BGR at 505.1 FPS median on the same file. Therefore the NV12 number is not a fair comparison to the BGR paths, and the measured MPP-to-BGR path did not improve this software baseline.

For the worker's 640×640 letterbox on a 768×576 frame, RGA measured 2.342 ms/frame median versus 2.810 ms/frame for the OpenCV path across three batches of 1,000 operations after 30 warmups (about 1.20× at preprocessing only). The RGA process held `/dev/rga`; calls used synchronous `improcess` with CPU virtual-address input/output and a CPU-filled 114 padding canvas. No zero-copy path was tested. Mean pixel difference over six frames was 0.148–0.202, maximum 6–7. Same-model RKNN comparisons matched detection sets on six sampled frames at confidence 0.345 and 0.50. At confidence 0.25, frame 125 differed by one detection (10 CPU, 11 RGA; 10 matched at IoU 0.50). No ground truth is available, so the extra output cannot be classified as correct or incorrect.

These stage measurements do not establish an end-to-end gain. The worker remains on the OpenCV CPU decode/preprocess path by default; MPP/RGA integration, quantization, asynchronous inference, and DMA-BUF still require controlled end-to-end comparisons. See [P2_ACCEPTANCE_REPORT.md](P2_ACCEPTANCE_REPORT.md) and the raw files in [evidence/p2](evidence/p2/).

## Planned baseline matrix

| Stage | Reference | Candidate | Required evidence |
|---|---|---|---|
| Video decode | OpenCV/CPU decode | GStreamer MPP decode | Stage probe completed; MPP NV12 is high-throughput, while the measured MPP+BGR path is slower than OpenCV FFmpeg. Full worker A/B remains open |
| Resize/color | OpenCV CPU | RGA | Worker-matched stage probe completed at about 1.20×; six-frame tensor/detection sensitivity recorded. Full worker A/B and labeled accuracy remain open |
| Inference | RKNN C API single path | INT8 or asynchronous path | Matching outputs, NPU-only duration, end-to-end effect, memory and errors |
| Buffer transport | explicit CPU copy | imported DMA-BUF if compatible | fd/allocator ownership, cache sync, copy count, correctness and measured impact |
| Scheduling | FIFO | latest-frame policy | queue wait, frame age, drops, event freshness and bounded RSS |

Before changing an implementation, save its source commit and raw run data. An optimization is accepted only if it improves the named metric without violating output correctness or stability. Neutral and negative results remain in the report.
