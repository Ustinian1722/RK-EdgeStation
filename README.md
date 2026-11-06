# RK-EdgeStation

RK-EdgeStation is an independent, single-board RK3576 edge-vision project. Its target is a maintainable industrial vision terminal that accepts local video or RTSP, runs a Rockchip NPU model, tracks objects, emits evidence-backed events, and exposes a local management interface. It has no Jetson dependency or second compute board.

The RK3576 demo runs a single-stream loop: MP4/RTSP input, RKNN inference, tracking, ROI and safety events, evidence images, SQLite history, local REST API, and a web dashboard. The existing YOLOv8n COCO model remains the fast general-object baseline. A separate 12-class YOLOX-S PPE/person/forklift candidate was trained, converted to RKNN, and exercised on the board. A label-grounded replay evaluator matched the expected PPE-event presence in 2/2 fixed-scene clips (one positive, one compliant). Both clips repeat one development-validation image, so this is a workflow check rather than an independent or temporal accuracy estimate. See the [PPE acceptance report](PPE_ACCEPTANCE_REPORT.md) and [event evaluation record](evidence/ppe/event-level-replay-evaluation.json).

The selectable MPP/RGA Worker path has been measured end to end against the OpenCV/CPU reference. Earlier matched tests found modest decode/preprocess gains and a same-graph PPE INT8 speedup, with a measurable `no_vest` recall tradeoff on a development-validation split. A new 2026-11-07 follow-up adds an optional partial low-copy path: GStreamer/MPP DMA-BUF → RGA → RKNN-owned input memory. On a 15-FPS dynamic construction clip, the three-run media-clock median was 8.046 FPS with 184.9 ms decode-to-result P95, versus 7.937 FPS and 186.0 ms for the stable Z0 path; explicit inference-ingress source copies fell from 1.3824 MB/frame to zero. The throughput gain is small, so Z0 remains the default. See [zero-copy feasibility](ZERO_COPY_FEASIBILITY.md), [implementation details](ZERO_COPY_IMPLEMENTATION.md), [benchmark results](ZERO_COPY_BENCHMARK_REPORT.md), and [dynamic event evaluation](DYNAMIC_VIDEO_EVALUATION.md).

## Current target and findings

- Board: KICKPI K7; device tree identifies Rockchip RK3576.
- OS/kernel: Ubuntu 24.04.3, vendor Linux 6.1.75, aarch64, 8 GiB RAM.
- Present: GCC 13.3, CMake 3.28, OpenCV 4.6, MPP 1.5, RGA 2.2, GStreamer 1.24, RKNN Runtime 2.0.0b0.
- MPP decoder and RGA device nodes are present. No connected camera sensor or USB camera was identified; `/dev/video0` cannot be opened.
- The official RKNN C API sample and project YOLOv8n worker ran with Runtime 2.0.0b0 / driver 0.9.7. One repeated-still run recorded shared NPU/IOMMU interrupts +5,096 during the worker and 0 during a two-second idle check. A dynamic-run counter later read lower than its captured in-run value, so IRQ data remains exploratory; `/dev/rknpu*` nodes were not visible, per-core utilization is unavailable, and devfreq stayed at 300 MHz.
- The board has about 44 GiB free on its root filesystem. Host conversion uses an isolated Ubuntu 22.04 x86-64 WSL2 distro and authorized Toolkit2 2.0.0b0. The host needs no Rockchip silicon or runtime: conversion is optional tooling, while native build and inference use the RK3576 board. See [development environments](DEVELOPMENT_ENVIRONMENT.md) and the workstation bootstrap guide at `D:\WSL\RK-EdgeStation-bootstrap\README.md`.

Evidence: [hardware audit](HARDWARE_AUDIT.md), [captured output](evidence/p0/hardware-audit.txt).

## System scope

The planned product has six cooperating subsystems:

1. C++17 video ingest and media processing for MP4/RTSP, with camera input enabled only after a physical device is verified.
2. RKNN C API inference, with a CPU/OpenCV reference path, model and tensor checks, safe resource ownership, and measured preprocessing/inference/postprocessing.
3. Detection, lightweight tracking, ROI and safety-zone rules, PPE association/status, forklift/person co-occurrence alerts, enter/leave/dwell/count events, and timestamped evidence.
4. Bounded multithread scheduling, backpressure, FIFO/latest-frame modes, expiry and timeout handling, and separate latency/age/throughput/drop metrics.
5. Local REST and web management, SQLite, structured logs, authentication, recovery, and systemd integration.
6. RKNN-OptLab for graph/conversion inspection, FP16/INT8 comparisons, runtime profiling, attribution, and reports.

The implemented selectable pipeline is in [ARCHITECTURE.md](ARCHITECTURE.md). OpenCV/FFmpeg plus CPU preprocessing remains the reference. GStreamer/MPP offers a stable NV12 row-copy path and an optional RGA path; Z2 retains the MPP GstSample, imports its DMA-BUF into RGA, and writes directly to an RKNN-allocated input buffer. This removes the Worker's visible NV12 CPU row-copy, but internal driver/runtime transfers are not observable and full zero-copy is not claimed.

RKNN inference and the service run on the RK3576. Windows/WSL is used for model training/export/conversion; after a compatible `.rknn` exists, the host is not required for board inference. Latest-frame remains the default bounded-queue mode. A real-motion construction-site clip now exercises tracking and events, but yielded three visually apparent false PPE alerts and one uncertain alert; it had no confirmed PPE violation or forklift. Independent PPE accuracy, real-camera behavior, per-core NPU utilization, and production performance remain open. Forklift/person co-occurrence is an image-space risk cue, not a physical-distance estimate.

## Phases

| Phase | Scope | State |
|---|---|---|
| P0 | Hardware audit, license review, requirements, official RKNN smoke | Complete; official RK3576 C API sample passed |
| P1 | Single-stream detection, events, persistence, and local UI | MP4 and synthetic RTSP paths, ROI transitions, PPE rules, SQLite/API history and acknowledgement exercised on the board. Repeated still and one dynamic construction clip provide workflow/diagnostic evidence only; no independent PPE event accuracy is established |
| P2 | Media paths, quantization, scheduling, and profiling | OpenCV and GStreamer/MPP sources plus CPU/RGA preprocessing are integrated. Z0/Z1/Z2 board comparisons cover unpaced FIFO and media-clock Latest modes. Z2 is a validated partial low-copy option, not full zero-copy; Z0 remains default |
| P3 | Recovery and handoff | Manual restart, synthetic RTSP reconnect, and temporary systemd user-service recovery passed. A 24-hour soak is outside this project's acceptance scope; the earlier exploratory run stopped after 4 h 52 min and is not a completed soak |

## Documentation

- [Build and deploy](BUILD_AND_DEPLOY.md)
- [Development environments](DEVELOPMENT_ENVIRONMENT.md)
- [Model conversion](MODEL_CONVERSION.md)
- [P1 acceptance report](P1_ACCEPTANCE_REPORT.md)
- [PPE training protocol](TRAINING_PPE.md) and [PPE acceptance report](PPE_ACCEPTANCE_REPORT.md)
- [PPE event replay video generator](scripts/generate_ppe_replay_videos.py) and [label-grounded replay evaluator](scripts/evaluate_ppe_replay_events.py)
- [Four-group board performance report](PERFORMANCE_COMPARISON_REPORT.md)
- [Flagship MPP/RGA/INT8 optimization report](FLAGSHIP_OPTIMIZATION_REPORT.md)
- [DMA-BUF feasibility audit](ZERO_COPY_FEASIBILITY.md), [implementation](ZERO_COPY_IMPLEMENTATION.md), and [Z0/Z1/Z2 benchmark report](ZERO_COPY_BENCHMARK_REPORT.md)
- [Dynamic video and event evaluation](DYNAMIC_VIDEO_EVALUATION.md) and the board ablation runner / summarizer at [`scripts/run_zero_copy_ablation.py`](scripts/run_zero_copy_ablation.py) and [`scripts/summarize_zero_copy_ablation.py`](scripts/summarize_zero_copy_ablation.py)
- [Board PPE RKNN validation tools](scripts/make_ppe_rknn_image_manifest.py), [`ppe_rknn_dataset_eval`](tools/ppe_rknn_dataset_eval.cpp), and [metric evaluator](scripts/evaluate_ppe_rknn_predictions.py)
- [Optional long-run diagnostic and deployment controls](BUILD_AND_DEPLOY.md#optional-monitored-long-run-probe)
- [P3 graceful restart recovery probe](scripts/restart_recovery_smoke.py)
- `build/bin/create_static_video` creates and verifies a deterministic repeated-frame AVI test source for long-run experiments.
- [Benchmark protocol](BENCHMARK_PROTOCOL.md)
- [Optimization report](OPTIMIZATION_REPORT.md)
- [Test plan](TEST_PLAN.md) and [results](TEST_RESULTS.md)
- [P3 reliability and release acceptance](P3_ACCEPTANCE_REPORT.md)
- [Open-source references](OPEN_SOURCE_REFERENCES.md)
- [Interview guide](INTERVIEW_GUIDE.md)

## Repository hygiene

Model weights, videos, build output, local logs, and credentials are excluded from Git. Hardware evidence committed here is deliberately limited to sanitized audit output and reproducible reports. The board workspace is `/home/kickpi/projects/RK-EdgeStation`; existing board projects and system configuration are outside the project scope.
