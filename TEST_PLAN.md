# Test plan

Status: requirements-derived plan. Current completed checks and limits are recorded in [TEST_RESULTS.md](TEST_RESULTS.md), [P1_ACCEPTANCE_REPORT.md](P1_ACCEPTANCE_REPORT.md), and [P2_ACCEPTANCE_REPORT.md](P2_ACCEPTANCE_REPORT.md). SQLite unavailable/retry behavior and malformed local media/model inputs have initial coverage; broader failure recovery and the P3 reliability campaign remain planned.

## CI CPU/mock tests

- Validate configuration, model/source errors, ROI geometry, event timestamps, evidence metadata, and SQLite schema migrations.
- Exercise FIFO/latest-frame queue behavior, capacity bounds, expiry, cancellation, timeout, reconnect epochs, and worker shutdown.
- Feed deterministic synthetic frames/results through detector adapter, tracker, ROI enter/leave/dwell/count state machines.
- Confirm the mock backend cannot be reported as hardware inference in result metadata or benchmark output.
- Build with sanitizers where supported; run CTest on host CI.

## RK3576 board tests

1. Official RKNN C++ demo: **passed** for the bundled RK3576 MobileNet model. The project YOLOv8n model also completed a board-side RKNN smoke run; direct accelerator utilization remains unconfirmed.
2. CPU/OpenCV MP4 decode and pixel-format baseline.
   - Current automated check writes and reads a generated MP4, then verifies 60 decoded frames, 320×240 dimensions, and 30 FPS. It validates OpenCV software I/O only; it does not establish MPP hardware decode.
3. MPP GStreamer H.264 and H.265 decode, if matching sample media is available.
4. RGA resize/color conversion with input/output checksum or pixel tolerance against the CPU reference.
5. RKNN model output repeatability and longer output-release loop remain planned; the current run verifies tensor layout and successful output release on the tested frames.
6. Detection → tracking → ROI event smoke passed on repeated still and moving sample video with timestamps, saved evidence, SQLite persistence, and API retrieval. Debounced enter/leave, one-shot dwell, occupancy, and cumulative-entry semantics passed synthetic tests and a 150-frame RK3576 run. Track expiry removes the ID from occupancy without emitting a synthetic leave. Validate event precision and identity continuity against annotated input; the motion-predicted association candidate's 29 versus 33 enter events is not an accuracy measure.
7. Queue behavior passed the C++ policy tests and an RK3576 overload A/B using [the reusable runner](scripts/benchmark_queue_policy_ab.sh). On the 10-FPS sample with capacity 1, both policies had medians of 119/150 processed and 31 dropped. Latest p95 was 232.028 ms; FIFO p95 was 263.793 ms (three interleaved runs per policy). The earlier latest-frame 15-FPS smoke processed 88/150 and dropped 62. Broader burst rates, long-run RSS bounds, stale-result handling, and recovery remain planned.
8. Board CTest includes explicit queue-policy unit cases: latest replaces the oldest pending frame, FIFO drops the newest frame while full and preserves the remaining pop order. The Python API tests verify latest as the default and FIFO forwarding to the worker. The suite passed after the policy changes; see [TEST_RESULTS.md](TEST_RESULTS.md).

## Failure and recovery tests

- Passed on RK3576 using [the reusable failure-path probe](scripts/probe_worker_failure_paths.sh): truncated MP4, empty MP4, and malformed RKNN model all returned nonzero within 30 seconds. The media cases exposed `status=error` with zero processed frames/events; invalid model reported `rknn_init` failure. Raw evidence is in [P2_ACCEPTANCE_REPORT.md](P2_ACCEPTANCE_REPORT.md).
- Passed on RK3576 with a simulated unavailable SQLite path: the health API reported storage unavailable, the event API returned HTTP 503, and a later request reinitialized SQLite after the obstruction was removed. This did not exhaust the board filesystem; see [TEST_RESULTS.md](TEST_RESULTS.md).
- RTSP disconnect/reconnect and authentication/network failure.
- Unsupported codec, missing input/model, permission denial, and runtime initialization/inference failure.
- Simulated inference-worker exception; verify worker status, event suppression, and clean shutdown.
- Expired frame, deadline timeout, actual low-disk behavior, and normal process/service restart.
- Check file descriptors, thread count, memory, DB integrity, and current-event state before/after each recovery test.

## Long-run test

P3: at least 24 continuous hours on the board, with fixed software/model/config/input, periodic temperature/memory/disk/latency samples, restart/error counters, and saved raw logs. Run the event pipeline against a deterministic source and inspect that evidence/event output remains consistent.
