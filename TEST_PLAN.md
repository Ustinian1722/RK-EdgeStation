# Test plan

Status: requirements-derived plan. Current completed checks and limits are recorded in [TEST_RESULTS.md](TEST_RESULTS.md) and [P1_ACCEPTANCE_REPORT.md](P1_ACCEPTANCE_REPORT.md); the remaining cases below are still planned.

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
6. Detection → tracking → ROI event smoke passed on repeated still and moving sample video with timestamps, saved evidence, SQLite persistence, and API retrieval. Identity continuity is not accepted yet: the moving run emitted 42 unique track IDs for 42 enter events, including likely fragmentation. Dwell/leave and labeled detection/event accuracy remain planned.
7. A latest-frame queue smoke passed at 15-FPS source pacing: 150 decoded, 88 processed, 62 dropped, capacity 1. Broader FIFO/latest overload, RSS bounds, stale-result, and recovery checks remain planned.

## Failure and recovery tests

- RTSP disconnect/reconnect and authentication/network failure.
- Corrupt, truncated, unsupported-codec, and empty media input.
- Missing or incompatible model, permission denial, runtime initialization failure, and failed inference call.
- Simulated inference-worker exception; verify worker status, event suppression, and clean shutdown.
- Queue overload, expired frame, deadline timeout, SQLite unavailable, low disk, and normal process/service restart.
- Check file descriptors, thread count, memory, DB integrity, and current-event state before/after each recovery test.

## Long-run test

P3: at least 24 continuous hours on the board, with fixed software/model/config/input, periodic temperature/memory/disk/latency samples, restart/error counters, and saved raw logs. Run the event pipeline against a deterministic source and inspect that evidence/event output remains consistent.
