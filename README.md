# RK-EdgeStation

RK-EdgeStation is an independent, single-board RK3576 edge-vision project. Its target is a maintainable industrial vision terminal that accepts local video or RTSP, runs a Rockchip NPU model, tracks objects, emits evidence-backed events, and exposes a local management interface. It has no Jetson dependency or second compute board.

The project has completed **P0** and reached **P1 board-side still and moving-video detection-to-event/API runs**. The official RK3576 MobileNet C++ sample and the project YOLOv8n worker both ran through the board's RKNN Runtime. A 150-frame moving sample completed on RK3576 with 130 frames processed, 20 dropped, 1,033 detections, 42 persisted events, and HTTP 200 from health/metrics/events. The output also exposed likely track fragmentation and two low-confidence `skis` predictions on a pedestrian, so this remains a pipeline smoke rather than an accuracy result; full P1 acceptance remains open. See the [P0 report](P0_ACCEPTANCE_REPORT.md) and [P1 report](P1_ACCEPTANCE_REPORT.md).

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

The P1 prototype includes a native RKNN YOLO worker, bounded latest-frame queue, IoU tracker, ROI entry events with saved frame evidence, SQLite persistence, and a loopback-only HTTP dashboard. The non-quantized (`fp`) YOLOv8n model was converted in WSL and run on RK3576. A 150-frame moving sample run processed 130 and dropped 20 under the one-frame latest queue. Dynamic-scene ID continuity, model-vs-ONNX numerical comparison, labeled accuracy, per-core NPU utilization, and product performance remain open.

## Phases

| Phase | Scope | State |
|---|---|---|
| P0 | Hardware audit, open-source/license review, requirements, architecture, official RKNN demo | Complete; official RK3576 C API smoke test passed |
| P1 | Single-stream detection-to-event loop and minimum local web UI | Static-video model/API smoke passes on board; full acceptance remains in progress |
| P2 | MPP/RGA pipeline, quantization comparison, scheduling and profiling | Not started |
| P3 | Device management, failure/reliability testing, 24-hour run, release package | Not started |

## Documentation

- [Build and deploy](BUILD_AND_DEPLOY.md)
- [Model conversion](MODEL_CONVERSION.md)
- [P1 acceptance report](P1_ACCEPTANCE_REPORT.md)
- [Benchmark protocol](BENCHMARK_PROTOCOL.md)
- [Optimization report](OPTIMIZATION_REPORT.md)
- [Test plan](TEST_PLAN.md) and [results](TEST_RESULTS.md)
- [Open-source references](OPEN_SOURCE_REFERENCES.md)
- [Interview guide](INTERVIEW_GUIDE.md)

## Repository hygiene

Model weights, videos, build output, local logs, and credentials are excluded from Git. Hardware evidence committed here is deliberately limited to sanitized audit output and reproducible reports. The board workspace is `/home/kickpi/projects/RK-EdgeStation`; existing board projects and system configuration are outside the project scope.
