# Requirements and phase gates

Status: P0 requirements freeze, 2026-11-05. The product boundary is one RK3576 board acting as an independent edge device. Jetson is out of scope, and there is no two-board architecture.

## Product objective

Build a reproducible Linux edge-vision device that ingests video, performs real RKNN inference, turns detections into track- and ROI-based events, and makes status, video, event evidence, and runtime metrics available locally. A normal detection box alone is not an event decision.

## Functional requirements

### Video and media

- Required sources: local MP4 and RTSP.
- Optional sources: USB or MIPI cameras only after a physical device and a usable V4L2 format have been confirmed.
- Keep a CPU/OpenCV path as a correctness and performance reference.
- Evaluate MPP decoding and RGA resize/color conversion on the actual KICKPI K7. Keep explicit copy and fallback paths.
- DMA-BUF or other low-copy paths require observed buffer identity/ownership and an end-to-end board measurement; never infer zero-copy from API availability.

### Inference and models

- C++17 core uses RKNN C API/RKNPU2 and RAII ownership for contexts, outputs, buffers, and threads.
- Model setup validates file, runtime/driver compatibility, input/output counts, tensor attributes, layouts, quantization, and dimensions before inference.
- Required lifecycle: load, inspect, preprocess, infer, postprocess, release, and handle each error path.
- The first model is an official RK3576-supported YOLOv8n or YOLO11n model. Model/toolkit/runtime versions must match; the board currently has Runtime 2.0.0b0, so the latest Model Zoo 2.5.0 path is not the first test target.
- Compare FP16 and INT8 only after selecting an identical graph, calibration inputs, and output-matching procedure. Keep model artifacts outside Git.
- RKNN-OptLab analyzes ONNX structure, conversion logs, output error, runtime timing, pre/postprocessing, and available CPU/NPU resource data. A second model must be converted and run on supported hardware before being described as validated.

### Vision events

- Detection provides class, confidence, and source-frame timestamp.
- A lightweight tracker assigns stable IDs and handles lost/reacquired tracks.
- ROI configuration supports enter/leave, dwell, and count rules with hysteresis/debounce where needed.
- Every emitted event stores event type, state, timestamp, confidence, track/ROI association, source frame or image evidence, and rule/model versions.
- Stale tracks and failed inference must stop producing active events.

### Scheduling and observability

- Bounded queues with FIFO and latest-frame policies; each policy has explicit overflow behavior.
- Separate worker stages for ingest/decode, preprocessing, inference, postprocessing/events, and API/persistence.
- Track queue wait, inference duration, end-to-end latency, frame age, throughput, drops, timeouts, queue depth, process/board memory, temperature, and available power data independently.
- Overload must produce bounded memory and an explicit drop/timeout state; no stale result may be reported as current.

### Management and operations

- Local REST API and browser UI for source/model configuration, ROI, video preview, events, and live performance.
- SQLite persistence, structured logs, health status, restart behavior, and systemd service definition.
- REST listener defaults to loopback. If LAN access is enabled, require authentication and document the network boundary.
- No externally reachable service is enabled by default.

## Phase gates

### P0 — evidence before system design freeze

1. Inspect local host and board; record exact hardware, OS, kernel, drivers, runtimes, packages, devices, clocks, permissions, and storage.
2. Review mature upstream implementations and their licenses. Pin sources and distinguish vendor claims, community reports, and local measurements.
3. Freeze the requirement boundary and layered architecture using the audit.
4. Build and run an official RKNN C++ demo with an RK3576-compatible model. A successful compile alone does not pass this gate.
5. Preserve command output and explain any gate that remains blocked.

### P1 — single-stream closed loop

- Play one local MP4 through video ingest, real RK3576 NPU inference, tracking, a rule-triggered event with saved evidence, and a local web/API view.
- Demonstrate queue limits, timeout/error state, persisted event, and truthful runtime metrics.

### P2 — measured media and scheduling

- Compare CPU/OpenCV with MPP/RGA stages under the same input and resolution.
- Measure FP16/INT8 consistency and end-to-end impact; profile before changing code and keep Before/After records.
- Exercise overload, frame freshness policies, reconnect, and recovery.

### P3 — product and release

- Complete source/model/source-config management, local API/UI, persistence, systemd startup, and recovery.
- Execute failure tests and at least one 24-hour board run with original logs and a reproducible protocol.
- Produce the technical report, demo recording, full source, reproducible build, benchmark tooling, and truthful resume/interview material.

## Verification rules

- A mock verifies control flow and error handling only; it is never hardware evidence.
- Use physical board runs as the unit for hardware results. Retain raw samples and conditions.
- No performance target is asserted until measured on this board. Compare identical source, model, input, resolution, frequency/governor state, and software versions.
- Do not replace the board kernel/runtime/driver, flash storage, alter boot configuration, or expose a service as part of routine development.
