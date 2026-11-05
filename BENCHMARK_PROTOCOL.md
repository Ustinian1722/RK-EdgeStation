# Benchmark protocol

Status: protocol draft frozen for P1/P2. P2 stage probes are recorded in [P2_ACCEPTANCE_REPORT.md](P2_ACCEPTANCE_REPORT.md); they are not a controlled end-to-end product benchmark.

## Reproducibility record

Each run records board model/OS/kernel, CPU and NPU governor/frequency snapshot, Runtime/driver/library versions, compiler/build flags, model and video hashes, source URI, frame dimensions/codec/FPS, queue policy/capacity, worker counts, warmup, timed frame count, repetition, temperature, free memory, and any throttling/error.

## Measurements

- **NPU inference duration:** the target metric is monotonic time around `rknn_run` only, excluding preprocessing/postprocessing. The current worker's `last_inference_ms` also includes `rknn_outputs_get`; report that implementation detail until a dedicated run-only metric is added.
- **End-to-end latency:** monotonic time from decoded-frame ingest to the corresponding result/event being available.
- **Frame age:** result-ready monotonic time minus capture time; for files, define capture time from presentation timestamp and a recorded playback clock mapping.
- **Throughput:** completed frames divided by measured wall time, with source pacing stated.
- **Drop rate:** intentionally/accidentally discarded frames divided by all frames observed at the named pipeline boundary; report reason-specific counts.
- **Queue wait/depth:** time and high-water mark for each bounded queue.
- **Accuracy/consistency:** match output tensors/detections against the source graph using documented preprocessing, confidence threshold, class match, and box IoU. Report unmatched detections and score/coordinate deltas.
- **Resource use:** process RSS, whole-board memory, CPU utilization per core, NPU frequency/load when exposed, temperature, errors, and power only if a real measurement interface is available.

The current board exposes RKNPU devfreq but no `/dev/rknpu*` node. One repeated-still 150-frame RKNN worker run recorded a +5,096 increment in shared NPU/IOMMU `/proc/interrupts` counters and no increment during a two-second idle check. A later moving-video sample rose from 12,377 to 19,626 in 58 captured samples, but a same-boot read several minutes later was 19,098. This unresolved discrepancy means IRQ values remain exploratory and do not establish utilization. Devfreq stayed at 300 MHz and its `load` string also remained unchanged during idle, so neither value is used as a utilization percentage or scaling claim. No per-core utilization is available.

## Repetition plan

For each stable model/source configuration: 30 warmup frames, 1,000 measured frames, three repetitions, same video and file hash, with p50/p95/p99 and arithmetic mean/std for latency. Record raw per-frame samples. If the source cannot supply 1,000 frames, state the actual count; do not silently extrapolate.

### P2 media-stage probe details

- MPP decode: 150-frame 768×576 H.264 Baseline/Constrained Baseline sample repeated ten times; one warmup run and three measured runs; count every decoded buffer and preserve first/last PTS, negotiated caps, bus warnings, process CPU and RSS. Keep the NV12/fakesink result separate from any BGR-conversion result.
- OpenCV decode: use the same H.264 file and FFmpeg backend, produce BGR, discard 30 warmup frames, then decode/count 1,500 frames in each of three repetitions.
- RGA preprocessing: use the same decoded BGR frame for both paths; compare the worker's 640×640 letterbox, fill value 114, and RGB order. Run 30 warmups and 1,000 timed calls per repetition, three repetitions. Include CPU fill of the RGA destination padding canvas in candidate time. Record image differences and keep input/output memory type explicit. A stage-only improvement is not an end-to-end gain.
- Model sensitivity: run CPU and RGA uint8 inputs sequentially through the same FP RKNN model/context for each sampled frame. Compare all nine tensors and postprocessed detections at more than one confidence threshold. A matching detection set is consistency evidence only; without labels it is not accuracy evidence.

### P2 FP/INT8 model comparison

- Convert the same ONNX graph with the pinned Toolkit2 version and target `rk3576`. Record the exact calibration manifest, image hashes, conversion log, model hashes, Toolkit warnings, and artifact sizes. Keep calibration images and model binaries outside Git.
- Keep calibration and holdout frames disjoint. For exploratory single-scene checks, report the scene and frame indices and do not call the result representative accuracy.
- For tensor/detection diagnostics, feed both board models byte-identical preprocessed RGB UINT8 NHWC inputs. State whether Runtime returns dequantized floats or native integer outputs. Report per-threshold detection agreement separately from labeled accuracy. Single-run diagnostic timings are not latency distributions.
- For the real worker A/B, use the same video, confidence, queue mode and scheduling conditions; interleave at least three repetitions per model. Report processed/dropped frames, throughput, worker inference interval (`rknn_run` plus output fetch), end-to-end p50/p95, memory, temperature and event/output counts. Capacity 1 represents latest-frame behavior; a large preserve-all queue measures backlog as well as throughput and must not be described as live latency.
- Keep the FP model as the product default until representative calibration and a labeled evaluation establish acceptable detection and event behavior. A faster or smaller INT8 artifact by itself does not pass the accuracy gate.

Use the same source, model, resolution, input path, queue policy, worker count, and power state for before/after comparisons. Warm the board to a recorded steady state; do not change clocks between paired runs. Report model inference and end-to-end changes separately.

## Overload and stability

- `latest`: on a full queue, evict the oldest pending frame and enqueue the new frame. `fifo`: preserve queued order and drop the newly decoded frame while full. Both must remain bounded and report their policy and dropped count.
- Exercise slow-consumer and burst-input cases under both policies at increasing rates while recording queue bounds, frame age, drops, timeouts, RSS, CPU/NPU, and event correctness. Compare the actual processed frame sequences; total drop counts alone do not establish freshness or correctness.
- Use deterministic injected delays for queue-control tests; label them as scheduler tests, not hardware saturation results.
- Run reconnect, malformed input, model-load error, inference-worker failure, low disk, restart, and long-run tests from [TEST_PLAN.md](TEST_PLAN.md).
- P3 requires a continuous 24-hour run with timestamps, periodic resource snapshots, error counts, restart count, storage free space, and event sanity checks.
