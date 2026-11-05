# RK-EdgeStation

RK-EdgeStation is an independent, single-board RK3576 edge-vision project. Its target is a maintainable industrial vision terminal that accepts local video or RTSP, runs a Rockchip NPU model, tracks objects, emits evidence-backed events, and exposes a local management interface. It has no Jetson dependency or second compute board.

The project has completed **P0** and reached **P1 board-side still and moving-video detection-to-event/API runs**. The official RK3576 MobileNet C++ sample and project YOLOv8n worker both ran through the board's RKNN Runtime. A motion-predicted tracker refinement completed a 150-frame moving sample with 129 processed frames, 21 dropped, 1,021 detections, and 29 persisted events. The ROI layer now debounces enter/leave transitions, emits one dwell event per confirmed visit, and reports current occupancy plus cumulative entries. A 150-frame board run produced 17 enters, 4 leaves, and 7 dwells; each dwell followed its enter by the configured 1.2 seconds of media time. A six-frame same-input ONNX CPU vs RKNN comparison matched detection sets at the configured 0.25 threshold, but a confidence sweep exposed one bus-frame difference at 0.345 and raw low-score tensors differ. This is not a labeled accuracy result; two low-confidence `skis` predictions remain. Full P1 acceptance is open. See the [P0 report](P0_ACCEPTANCE_REPORT.md) and [P1 report](P1_ACCEPTANCE_REPORT.md).

## Current target and findings

- Board: KICKPI K7; device tree identifies Rockchip RK3576.
- OS/kernel: Ubuntu 24.04.3, vendor Linux 6.1.75, aarch64, 8 GiB RAM.
- Present: GCC 13.3, CMake 3.28, OpenCV 4.6, MPP 1.5, RGA 2.2, GStreamer 1.24, RKNN Runtime 2.0.0b0.
- MPP decoder and RGA device nodes are present. No connected camera sensor or USB camera was identified; `/dev/video0` cannot be opened.
- The official RKNN C API sample and project YOLOv8n worker ran with Runtime 2.0.0b0 / driver 0.9.7. One repeated-still run recorded shared NPU/IOMMU interrupts +5,096 during the worker and 0 during a two-second idle check. A dynamic-run counter later read lower than its captured in-run value, so IRQ data remains exploratory; `/dev/rknpu*` nodes were not visible, per-core utilization is unavailable, and devfreq stayed at 300 MHz.
- The board has about 44 GiB free on its root filesystem. Host conversion uses an isolated Ubuntu 22.04 x86-64 WSL2 distro and authorized Toolkit2 2.0.0b0. The host needs no Rockchip silicon or runtime: conversion is optional tooling, while native build and inference use the RK3576 board. The reproducible WSL setup is documented at `D:\WSL\RK-EdgeStation-bootstrap\README.md`.

Evidence: [hardware audit](HARDWARE_AUDIT.md), [captured output](evidence/p0/hardware-audit.txt).

## System scope

The planned product has six cooperating subsystems:

1. C++17 video ingest and media processing for MP4/RTSP, with camera input enabled only after a physical device is verified.
2. RKNN C API inference, with a CPU/OpenCV reference path, model and tensor checks, safe resource ownership, and measured preprocessing/inference/postprocessing.
3. Detection, lightweight tracking, ROI rules, enter/leave/dwell/count events, and timestamped evidence.
4. Bounded multithread scheduling, backpressure, FIFO/latest-frame modes, expiry and timeout handling, and separate latency/age/throughput/drop metrics.
5. Local REST and web management, SQLite, structured logs, authentication, recovery, and systemd integration.
6. RKNN-OptLab for graph/conversion inspection, FP16/INT8 comparisons, runtime profiling, attribution, and reports.

The staged requirements and architecture are in [REQUIREMENTS.md](REQUIREMENTS.md) and [ARCHITECTURE.md](ARCHITECTURE.md). The implementation will not call a pipeline zero-copy until buffer ownership and DMA-BUF import are demonstrated on this board.

The P1 prototype includes a native RKNN YOLO worker, a bounded queue with selectable latest-frame (drop oldest pending) and FIFO (drop newest while full) policies, a class-gated motion-predicted IoU/center tracker, debounced ROI enter/leave/dwell events, occupancy and cumulative-entry metrics, saved frame evidence, SQLite persistence, and a loopback-only HTTP dashboard. Latest-frame remains the default. Dwell uses monotonicized media timestamps, with nominal-FPS or decode-time fallback when source timestamps are invalid. Confirmed exits require consecutive outside observations; a track that expires while still inside is removed from occupancy without a synthetic leave event. The non-quantized (`fp`) YOLOv8n model was converted in WSL and run on RK3576. Six exploratory single-frame ONNX CPU vs RKNN comparisons matched at confidence 0.25, while a bus-frame comparison diverged at confidence 0.345 and raw class/score-sum outputs differ in low-score regions; broad model equivalence and labeled accuracy remain open. The controlled tracker/event comparison used a 150-frame queue to preserve every frame, but p95 end-to-end delay reached 3.5–4.6 seconds; the live freshness path uses capacity 1. Ground-truth ID continuity, per-core NPU utilization, and product performance remain open.

## Phases

| Phase | Scope | State |
|---|---|---|
| P0 | Hardware audit, open-source/license review, requirements, architecture, official RKNN demo | Complete; official RK3576 C API smoke test passed |
| P1 | Single-stream detection-to-event loop and minimum local web UI | Still/moving API loops and ROI enter/leave/dwell/count plumbing pass; six-frame model comparison found a confidence-boundary difference; tracker identity and labeled accuracy remain open |
| P2 | MPP/RGA pipeline, quantization comparison, scheduling and profiling | INT8 A/B: 9.986 vs 7.602 median FPS in latest-frame mode, without labeled accuracy. Queue A/B: latest p95 232.0 ms vs FIFO 263.8 ms on one 10-FPS sample. SQLite path and bounded-write failure recovery now have board simulations; actual filesystem exhaustion remains open. INT8 remains experimental; MPP/RGA remain stage-only |
| P3 | Device management, failure/reliability testing, 24-hour run, release package | Reliability tooling is underway; RTSP reconnect, service restart, actual low-disk behavior, the 24-hour run, and release handoff remain open |

## Documentation

- [Build and deploy](BUILD_AND_DEPLOY.md)
- [Model conversion](MODEL_CONVERSION.md)
- [P1 acceptance report](P1_ACCEPTANCE_REPORT.md)
- [P3 stability runner and deployment controls](BUILD_AND_DEPLOY.md#p3-stability-run)
- `build/bin/create_static_video` creates and verifies a deterministic repeated-frame AVI test source for long-run experiments.
- [Benchmark protocol](BENCHMARK_PROTOCOL.md)
- [Optimization report](OPTIMIZATION_REPORT.md)
- [Test plan](TEST_PLAN.md) and [results](TEST_RESULTS.md)
- [Open-source references](OPEN_SOURCE_REFERENCES.md)
- [Interview guide](INTERVIEW_GUIDE.md)

## Repository hygiene

Model weights, videos, build output, local logs, and credentials are excluded from Git. Hardware evidence committed here is deliberately limited to sanitized audit output and reproducible reports. The board workspace is `/home/kickpi/projects/RK-EdgeStation`; existing board projects and system configuration are outside the project scope.
