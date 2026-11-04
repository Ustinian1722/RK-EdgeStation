# Optimization report

Status: no optimization has been attempted; no baseline or Before/After result exists.

## P0 constraints discovered

- The board already exposes MPP 1.5.0, RGA 2.2.0, `/dev/mpp_service`, `/dev/rga`, and a system DMA heap. This establishes API/device presence only.
- RKNN Runtime is 2.0.0b0 and the expected `/dev/rknpu` node is absent. NPU execution is not yet verified.
- No physical camera was detected. A fixed local video file will be used for the first comparable pipeline run.
- User-space MPP/RGA calls and DMA-BUF availability do not establish hardware execution or zero-copy behavior.

## Planned baseline matrix

| Stage | Reference | Candidate | Required evidence |
|---|---|---|---|
| Video decode | OpenCV/CPU decode | GStreamer MPP decode | Same file, codec, dimensions, timestamps; CPU, throughput and drops |
| Resize/color | OpenCV CPU | RGA | Pixel comparison, stride/format, operation time and full pipeline latency |
| Inference | RKNN C API single path | INT8 or asynchronous path | Matching outputs, NPU-only duration, end-to-end effect, memory and errors |
| Buffer transport | explicit CPU copy | imported DMA-BUF if compatible | fd/allocator ownership, cache sync, copy count, correctness and measured impact |
| Scheduling | FIFO | latest-frame policy | queue wait, frame age, drops, event freshness and bounded RSS |

Before changing an implementation, save its source commit and raw run data. An optimization is accepted only if it improves the named metric without violating output correctness or stability. Neutral and negative results remain in the report.
