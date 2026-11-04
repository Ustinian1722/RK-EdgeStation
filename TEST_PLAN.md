# Test plan

Status: requirements-derived plan. Tests will be added with the corresponding modules; no test result is implied by this document.

## CI CPU/mock tests

- Validate configuration, model/source errors, ROI geometry, event timestamps, evidence metadata, and SQLite schema migrations.
- Exercise FIFO/latest-frame queue behavior, capacity bounds, expiry, cancellation, timeout, reconnect epochs, and worker shutdown.
- Feed deterministic synthetic frames/results through detector adapter, tracker, ROI enter/leave/dwell/count state machines.
- Confirm the mock backend cannot be reported as hardware inference in result metadata or benchmark output.
- Build with sanitizers where supported; run CTest on host CI.

## RK3576 board tests

1. Official RKNN C++ demo: model load, runtime/driver query, one image, repeated inference, output inspection, release. Capture success or complete failure details.
2. CPU/OpenCV MP4 decode and pixel-format baseline.
   - Current automated check writes and reads a generated MP4, then verifies 60 decoded frames, 320×240 dimensions, and 30 FPS. It validates OpenCV software I/O only; it does not establish MPP hardware decode.
3. MPP GStreamer H.264 and H.265 decode, if matching sample media is available.
4. RGA resize/color conversion with input/output checksum or pixel tolerance against the CPU reference.
5. RKNN model output repeatability, tensor layout and output-release loop.
6. Full detection → tracking → ROI event path with timestamps and saved evidence.
7. FIFO and latest-frame load tests; verify finite queue/RSS and stale-result rejection.

## Failure and recovery tests

- RTSP disconnect/reconnect and authentication/network failure.
- Corrupt, truncated, unsupported-codec, and empty media input.
- Missing or incompatible model, permission denial, runtime initialization failure, and failed inference call.
- Simulated inference-worker exception; verify worker status, event suppression, and clean shutdown.
- Queue overload, expired frame, deadline timeout, SQLite unavailable, low disk, and normal process/service restart.
- Check file descriptors, thread count, memory, DB integrity, and current-event state before/after each recovery test.

## Long-run test

P3: at least 24 continuous hours on the board, with fixed software/model/config/input, periodic temperature/memory/disk/latency samples, restart/error counters, and saved raw logs. Run the event pipeline against a deterministic source and inspect that evidence/event output remains consistent.
