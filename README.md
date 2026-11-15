# RK-EdgeStation

**Hardware-accelerated industrial edge vision on Rockchip RK3576.**

C++17 · GStreamer/MPP · RGA · RKNN · DMA-BUF · Python · SQLite

[中文文档](README_CN.md) · [Quick Start](docs/quickstart.md) · [Architecture](docs/architecture.md) · [Benchmarks](benchmarks/README.md)

RK-EdgeStation is an on-device industrial vision engine that connects video ingest, hardware-accelerated image processing, NPU inference and temporal event reasoning in a unified pipeline. It turns camera frames into actionable PPE and ROI events, with local evidence storage and management — all running on a single RK3576.

![RK-EdgeStation architecture](assets/architecture.svg)

## Engineering highlights

| | |
| --- | --- |
| **Heterogeneous acceleration** | GStreamer/MPP hardware decoding, RGA resize/color conversion/letterbox and RKNN NPU inference; interchangeable OpenCV reference backends. |
| **DMA-BUF shared memory** | Retained decoder-buffer lifetime, FD import into RGA, RKNN-owned input allocation and explicit cache synchronization. |
| **Inference optimization** | FP/INT8 RKNN conversion, tensor-contract validation, output parity checks, accuracy/latency trade-offs and stage-level profiling. |
| **Real-time scheduling** | C++17 producer–consumer pipeline, bounded queues, Latest/FIFO policies, backpressure, independent source state and weighted fair scheduling. |
| **Industrial event engine** | Object tracking, PPE association, ROI entry/exit/dwell, temporal confirmation, cooldown and evidence-backed alerts. |
| **Edge management** | Python REST service, web dashboard, SQLite event history, RTSP reconnection and systemd service recovery. |

## Performance on RK3576

| Optimization | Baseline → optimized |
| --- | ---: |
| COCO single-stream throughput (10 FPS input) | **7.66 → 9.99 FPS** |
| COCO decode-to-result P95 | **236.6 → 106.3 ms** |
| COCO process CPU (one-core equivalent) | **94.1% → 47.5%** |
| YOLOX-S PPE FP → INT8 throughput | **4.59 → 7.98 FPS** |
| YOLOX-S PPE model size | **20.71 → 11.57 MB** |
| DMA-BUF explicit NV12 ingress copy per frame | **1.38 MB → 0 B** |
| DMA-BUF preprocessing P50 | **3.65 → 2.72 ms** |

![Matched board optimization results](evidence/flagship/flagship-ablation.png)

Measurements: KICKPI K7 / RK3576; COCO OpenCV/CPU FP versus MPP/RGA INT8, matched YOLOX-S PPE FP/INT8 experiments, and 720p DMA-BUF shared-input tests. PPE development-set AP50:95 changed from 46.77% to 45.76%. For configurations, metric definitions, accuracy evaluation and raw records, see the [benchmark index](benchmarks/README.md).

## System architecture

```text
MP4 / RTSP → VideoSource → MPP / OpenCV → bounded frame queues
                                             ↓
                            RGA / CPU → RKNN → detection
                                             ↓
                              tracking → PPE / ROI rules
                                             ↓
                              SQLite · REST API · Web UI
```

The optional low-copy path imports MPP DMA-BUF memory into RGA and writes directly to RKNN-owned input memory, eliminating the application-level NV12 ingress copy. The standard copy path remains available for compatible deployment across input layouts.

## Quick start

Tested platform: KICKPI K7, RK3576, Ubuntu 24.04.3 aarch64, OpenCV 4.6, GStreamer 1.24 and RKNN Runtime 2.0.0b0.

On a board with the corresponding development libraries installed:

```bash
git clone https://github.com/Ustinian1722/RK-EdgeStation.git
cd RK-EdgeStation
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
bash scripts/run_demo.sh /path/to/coco.rknn /path/to/input.mp4
```

Open the local dashboard at **http://127.0.0.1:8080**. Supply a compatible RKNN model; model files and vendor runtimes are obtained separately. See the [deployment guide](docs/quickstart.md) for PPE, RTSP and DMA-BUF examples.

## Explore the implementation

| Area | Entry point |
| --- | --- |
| Video ingest and memory ownership | [video_source.cpp](src/video_source.cpp) |
| RGA / RKNN worker | [preprocess.cpp](src/preprocess.cpp), [main.cpp](src/main.cpp) |
| Scheduling and queue policies | [bounded_queue.hpp](src/bounded_queue.hpp), [multistream.cpp](src/multistream.cpp) |
| Tracking and event logic | [tracker.cpp](src/tracker.cpp), [ppe_events.cpp](src/ppe_events.cpp), [roi_events.cpp](src/roi_events.cpp) |
| Service and management | [server.py](python/edge_station/server.py), [storage.py](python/edge_station/storage.py), [web](web/) |
| Testing and reproducibility | [tests](tests/), [benchmark tools](scripts/), [documentation](docs/README.md) |

## License

Apache-2.0 project source. External models, datasets and runtime components have their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
