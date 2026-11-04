# Benchmark protocol

Status: protocol draft frozen for P1/P2; no performance values have been measured.

## Reproducibility record

Each run records board model/OS/kernel, CPU and NPU governor/frequency snapshot, Runtime/driver/library versions, compiler/build flags, model and video hashes, source URI, frame dimensions/codec/FPS, queue policy/capacity, worker counts, warmup, timed frame count, repetition, temperature, free memory, and any throttling/error.

## Measurements

- **NPU inference duration:** monotonic time around `rknn_run` only, excluding preprocessing/postprocessing.
- **End-to-end latency:** monotonic time from decoded-frame ingest to the corresponding result/event being available.
- **Frame age:** result-ready monotonic time minus capture time; for files, define capture time from presentation timestamp and a recorded playback clock mapping.
- **Throughput:** completed frames divided by measured wall time, with source pacing stated.
- **Drop rate:** intentionally/accidentally discarded frames divided by all frames observed at the named pipeline boundary; report reason-specific counts.
- **Queue wait/depth:** time and high-water mark for each bounded queue.
- **Accuracy/consistency:** match output tensors/detections against the source graph using documented preprocessing, confidence threshold, class match, and box IoU. Report unmatched detections and score/coordinate deltas.
- **Resource use:** process RSS, whole-board memory, CPU utilization per core, NPU frequency/load when exposed, temperature, errors, and power only if a real measurement interface is available.

## Repetition plan

For each stable model/source configuration: 30 warmup frames, 1,000 measured frames, three repetitions, same video and file hash, with p50/p95/p99 and arithmetic mean/std for latency. Record raw per-frame samples. If the source cannot supply 1,000 frames, state the actual count; do not silently extrapolate.

Use the same source, model, resolution, input path, queue policy, worker count, and power state for before/after comparisons. Warm the board to a recorded steady state; do not change clocks between paired runs. Report model inference and end-to-end changes separately.

## Overload and stability

- Exercise slow-consumer and burst-input cases at increasing rates while recording queue bounds, frame age, drops, timeouts, RSS, CPU/NPU, and event correctness.
- Use deterministic injected delays for queue-control tests; label them as scheduler tests, not hardware saturation results.
- Run reconnect, malformed input, model-load error, inference-worker failure, low disk, restart, and long-run tests from [TEST_PLAN.md](TEST_PLAN.md).
- P3 requires a continuous 24-hour run with timestamps, periodic resource snapshots, error counts, restart count, storage free space, and event sanity checks.
