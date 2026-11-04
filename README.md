# RK-EdgeStation

RK-EdgeStation is an independent, single-board RK3576 edge-vision project. Its target is a maintainable industrial vision terminal that accepts local video or RTSP, runs a Rockchip NPU model, tracks objects, emits evidence-backed events, and exposes a local management interface. It has no Jetson dependency or second compute board.

The project has completed **P0: audit, design, and official RKNN smoke test** and is progressing through P1. The official RK3576 MobileNet C++ sample compiled on the board and completed inference with the installed runtime. This is not a product detection benchmark. See the [P0 report](P0_ACCEPTANCE_REPORT.md) for evidence and open gates.

## Current target and findings

- Board: KICKPI K7; device tree identifies Rockchip RK3576.
- OS/kernel: Ubuntu 24.04.3, vendor Linux 6.1.75, aarch64, 8 GiB RAM.
- Present: GCC 13.3, CMake 3.28, OpenCV 4.6, MPP 1.5, RGA 2.2, GStreamer 1.24, RKNN Runtime 2.0.0b0.
- MPP decoder and RGA device nodes are present. No connected camera sensor or USB camera was identified; `/dev/video0` cannot be opened.
- The official RKNN C API sample ran successfully and reported Runtime 2.0.0b0 / driver 0.9.7. `/dev/rknpu*` nodes were not visible; the successful runtime access path has not been traced. The installed runtime is not managed by dpkg.
- The board has 44 GiB free on its root filesystem. The host is Windows 11; Python 3.12 and 3.13 are installed, but WSL's partially registered Ubuntu 22.04 distro cannot currently be queried or started. The host has no CMake/C++ compiler, so native board builds are used. Toolkit2 conversion needs a compatible Linux x86-64 software environment, but no Rockchip host hardware; the deployed application itself runs on the board.

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

The P1 prototype now includes a native RKNN YOLO worker, bounded latest-frame queue, IoU tracker, ROI entry events with saved frame evidence, SQLite persistence, and a loopback-only HTTP dashboard. The worker builds natively on RK3576; queue/tracker tests, generated MP4 software encode/decode, and Python API/storage tests pass. Project-model inference remains pending conversion of a compatible FP16 model.

## Phases

| Phase | Scope | State |
|---|---|---|
| P0 | Hardware audit, open-source/license review, requirements, architecture, official RKNN demo | Complete; official RK3576 C API smoke test passed |
| P1 | Single-stream detection-to-event loop and minimum local web UI | Source/build/unit checks pass on board; project-model end-to-end run pending |
| P2 | MPP/RGA pipeline, quantization comparison, scheduling and profiling | Not started |
| P3 | Device management, failure/reliability testing, 24-hour run, release package | Not started |

## Documentation

- [Build and deploy](BUILD_AND_DEPLOY.md)
- [Model conversion](MODEL_CONVERSION.md)
- [Benchmark protocol](BENCHMARK_PROTOCOL.md)
- [Optimization report](OPTIMIZATION_REPORT.md)
- [Test plan](TEST_PLAN.md) and [results](TEST_RESULTS.md)
- [Open-source references](OPEN_SOURCE_REFERENCES.md)
- [Interview guide](INTERVIEW_GUIDE.md)

## Repository hygiene

Model weights, videos, build output, local logs, and credentials are excluded from Git. Hardware evidence committed here is deliberately limited to sanitized audit output and reproducible reports. The board workspace is `/home/kickpi/projects/RK-EdgeStation`; existing board projects and system configuration are outside the project scope.
