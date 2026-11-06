# RK3576 DMA-BUF / shared-input benchmark

## Result in one paragraph

The installed RK3576 stack supports an optional **partial low-copy** input path: GStreamer `mppvideodec` returns MPP-backed `GstMemory` with a valid DMA-BUF FD, librga imports that FD and an RKNN-owned input FD, then RGA writes the model tensor directly into the bound RKNN input allocation. This removes the Worker's explicit NV12 row-copy of 1,382,400 bytes per decoded frame. On a 15-FPS dynamic H.264 segment, three media-clock repetitions gave Z2 8.046 FPS versus Z0 7.937 FPS, P95 184.9 versus 186.0 ms, CPU 52.3% versus 56.2% of one core, and RSS 100.5 versus 107.5 MiB. The throughput improvement is small; **Z0 remains default and Z2 is optional**. We cannot see or rule out internal RGA, DMA-BUF, IOMMU, cache-maintenance, RKNN runtime, or output-transfer copies, so this is not full zero-copy.

## Board, model and input identity

- Board: KICKPI K7 / RK3576, Ubuntu 24.04.3, Linux 6.1.75 aarch64, 8 CPU cores.
- Runtime stack: GStreamer 1.24.2, Rockchip MPP GStreamer plugin 1.14.4, RKNN Runtime 2.0.0b0, librga API 1.10.1. CPU governor was `schedutil`; clocks were not forced.
- Final Worker SHA-256: `617a2dc21d1bb4e3155c90ea1cbc16f23071011f3a61504253588fc587fdeab0`.
- PPE INT8 model SHA-256: `88b10ac2711f4b00d2859dfa0e1089ccb34ebd78a73b7dc46bd51a56c92c502c` (11,566,602 bytes).
- Repeated benchmark segment: H.264 1280×720, 15 FPS, 8 s / 120 frames, SHA-256 `C97D49FD60776EBBD41D6F356337DCF7B20D0976983A6709B82720392040F8B4`.
- Full dynamic clip: 34 s / 510 frames, SHA-256 `E1FE2432F23D816E58D9F8D54770FB5CD994EFC013D278EF90CDA9306FE71D2B`.
- The source is real moving construction-site footage, not a factory line, and is not used as PPE accuracy truth. Provenance and manual observations are in [DYNAMIC_VIDEO_EVALUATION.md](DYNAMIC_VIDEO_EVALUATION.md).

The repeated 8-second segment matrix used the same model, video, RGA letterbox, 0.35 confidence, capacity 1 for Latest or capacity 512 for FIFO, and three repetitions per mode. Each run JSON records hashes, environment snapshot, command, raw metrics and stderr. Frequency uses the board's normal governor; no fixed frequency/thermal lock was applied. See the raw [media-clock matrix](evidence/dynamic-video/worker-ablation/raw/) and [offline FIFO matrix](evidence/dynamic-video/offline-fifo-ablation/raw/).

## Path definitions

| Mode | Input and memory path |
|---|---|
| Z0 | MPP `GstSample` → map + Y/UV row `memcpy` into packed NV12 → RGA writes a separate CPU-addressable model tensor → `rknn_inputs_set`. |
| Z1 | Same mapped NV12 row-copy → RGA writes through the RKNN-owned input allocation's virtual address → `rknn_mem_sync(TO_DEVICE)` → `rknn_run`. |
| Z2 | Retain MPP `GstSample` and borrowed FD → RGA imports the MPP source FD and RKNN input FD → NV12 color conversion/resize/letterbox into the RKNN allocation → `rknn_mem_sync(BIDIRECTIONAL)` → `rknn_run`. |

The Z2 source sample remains alive through synchronous `IM_SYNC` RGA work. Both imported RGA handles are released before the frame is released. Inference remains serial, so the single RKNN input allocation is not shared across concurrent in-flight frames. The Worker defaults to Z0 and requires explicit `--video-memory-mode dmabuf --rknn-input-mode shared` for Z2.

## Media-clock Latest results

Median of three runs. Latency columns are the median of each run's P95; CPU is process usage as percent of one core. “Source copies” counts only explicit inference-ingress visible NV12 row copies and excludes event JPEG/evidence work and unknown internal transfers.

| Mode | Processed / dropped of 120 | FPS | Decode-to-result P95 | Preprocess P50 | RKNN input submit / sync P50 | RKNN run P50 | CPU | RSS | Explicit source copies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Z0 | 65 / 55 | 7.937 | 186.0 ms | 3.652 ms | 0.948 / 0 ms | 116.857 ms | 56.2% | 107.5 MiB | 165,888,000 B |
| Z1 | 65 / 55 | 7.929 | 187.7 ms | 3.792 ms | 0 / 0.132 ms | 116.410 ms | 56.3% | 106.0 MiB | 165,888,000 B |
| Z2 | 65 / 55 | 8.046 | 184.9 ms | 2.718 ms | 0 / 0.242 ms | 116.591 ms | 52.3% | 100.5 MiB | 0 B |

For this 15-FPS input, Latest processes roughly 54% of decoded frames and drops older queued frames. This is an intentional freshness tradeoff, not decoder loss: backend dropped-frame counters stayed zero. Z2 reduced measured preprocessing by 25.6% and CPU by 3.9 percentage points versus Z0. NPU run time stayed around 116.6 ms and dominated the per-frame work. Z2 gained 1.4% throughput and improved P95 by only 1.1 ms, so it is useful for CPU/RSS headroom but not a compelling default FPS optimization.

## Unpaced FIFO throughput results

For offline throughput, the same 120-frame segment ran with media-clock pacing disabled, FIFO capacity 512, and three repetitions. All three modes processed 120/120 frames with zero drops. P95 includes a backlog of roughly 7 seconds median queue wait, so it is **not real-time latency**.

| Mode | FPS | Decode-to-result P95 | Queue wait P50 | Preprocess P50 | RKNN run P50 | CPU | Process RSS | Explicit source copies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Z0 | 8.048 | 13,847 ms | 7,111 ms | 3.838 ms | 114.956 ms | 51.5% | 258.2 MiB | 165,888,000 B |
| Z1 | 8.144 | 13,653 ms | 7,014 ms | 3.658 ms | 114.549 ms | 51.1% | 256.7 MiB | 165,888,000 B |
| Z2 | 8.071 | 13,871 ms | 7,160 ms | 2.863 ms | 115.293 ms | 50.2% | 100.9 MiB | 0 B |

Throughput is effectively unchanged (Z2 is 0.3% above Z0). Z0/Z1 hold 120 packed CPU NV12 frames in their FIFO queue, while Z2 retains decoder samples/DMA-BUF references; process RSS falls substantially because it no longer owns 120 packed NV12 copies. DMA-BUF pool memory may not be fully represented in process RSS, so the RSS delta is not a complete system-memory comparison.

The first no-pacing trial used Latest with capacity 1. It decoded the full clip and intentionally discarded most pending frames, processing only a few; it was not an offline-throughput measurement. A bug in the optional track-log enabled check was exposed by this trial, fixed to use `ofstream::is_open()`, and the final Worker was rebuilt before the reported matrices. The rejected trial is kept under `.work` locally and is not included in the final summary tables.

The first DMA-BUF sync experiments used `RKNN_MEMORY_SYNC_FROM_DEVICE`; those records are retained in the `*-from-device-sync` directories for traceability but are superseded. Final reported Z0/Z1/Z2 runs and parity probes use the explicit bidirectional sync required by the verified external RGA writer path.

## Full 34-second dynamic clip

One matched media-clock run per mode was made on the full 510-frame moving clip using the final Worker SHA. These are useful longer-run checks but are single repetitions, not a statistically stable matrix.

| Mode | Processed / dropped | FPS | Decode-to-result P50 / P95 | Preprocess P50 | RKNN run P50 | CPU | RSS | Explicit source copies |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Z0 | 272 / 238 | 7.971 | 161.2 / 188.2 ms | 3.825 ms | 116.900 ms | 56.4% | 107.5 MiB | 705,024,000 B |
| Z1 | 275 / 235 | 8.051 | 158.8 / 187.2 ms | 3.719 ms | 116.312 ms | 56.1% | 106.0 MiB | 705,024,000 B |
| Z2 | 277 / 233 | 8.117 | 156.7 / 187.0 ms | 2.676 ms | 116.338 ms | 52.3% | 100.5 MiB | 0 B |

All three reached EOS with exit status 0. Z2 is 1.8% faster than Z0 in this one run and P95 differs by 1.2 ms; this agrees with the repeated short media-clock result that the CPU preprocessing savings do not overcome the roughly 116-ms NPU limit. Raw records are under [full34-ablation](evidence/dynamic-video/full34-ablation/raw/). Event screenshots and track logging were disabled in this performance comparison.

## Input and output consistency

The final bidirectional-sync probe ran each PPE model three repeated measured inferences on the same captured MPP frame after warm-up:

- PPE INT8: Z0, Z1 and Z2 model input hashes matched; all three RKNN output hashes matched; all 14 detections matched exactly. Z2 RGA imports of MPP and RKNN FDs succeeded. See [`int8-shared-input-bidirectional.jsonl`](evidence/dynamic-video/int8-shared-input-bidirectional.jsonl).
- PPE FP: the same input/output/detection equality checks passed, with the FP model's larger native stride allocation. See [`ppe-fp-shared-input-bidirectional.jsonl`](evidence/dynamic-video/ppe-fp-shared-input-bidirectional.jsonl).
- The INT8 model input remains BGR NHWC UINT8 raw 0–255; FP retains the established UINT8 image submission contract. The direct-FD RGA conversion uses the same letterbox color and resize interpolation.

This validates outputs for one captured frame per model and three repetitions, not all frames, all colors/layouts, dynamic PPE accuracy, or internal Runtime copy behavior. COCO Z2 was not promoted by this PPE-focused experiment.

## What changed, what did not

- **Verified:** the board exposed a valid MPP DMA-BUF under ordinary NV12 caps; RGA imported it and the RKNN input allocation; bidirectional `rknn_mem_sync` succeeded; exact input/output/detection parity passed for PPE FP and INT8; ten repeated MP4 open/close cycles showed no per-cycle borrowed-FD leak; the Worker completed the dynamic clip under Z0/Z1/Z2.
- **Explicitly removed:** one CPU `memcpy` of visible NV12 rows per decoded frame in the inference path: 1,382,400 B/frame (921,600 B Y plane + 460,800 B UV plane). The RGA output is also written into the already bound RKNN input allocation, avoiding the separate Worker-owned model-input staging allocation and `rknn_inputs_set` call in Z1/Z2.
- **Not established:** zero internal copies or migrations; complete DMA/cache traffic; whether the RGA/runtime/kernel performs hidden synchronization or buffer movement; output zero-copy; camera/RTSP DMA-BUF reconnect; or all modifiers/strides. Event evidence still maps the source frame, copies NV12 rows, converts to BGR, draws and encodes JPEG. RKNN output fetching and CPU postprocessing remain.
- **Bottleneck:** PPE INT8 `rknn_run` P50 is around 115–117 ms. CPU preprocessor savings improve CPU load and memory use, while NPU execution fixes the effective ceiling near 8 FPS for this model/input.

## Recommendation and rerun

Keep `--video-memory-mode copy --rknn-input-mode api` as the default for the stable baseline. Enable `dmabuf` + `shared` only where the installed GStreamer/MPP/RGA/Runtime combination has the observed contiguous NV12 layout and the operator values reduced CPU/RSS. Reject unsupported layouts rather than guessing. The implementation, exact constraints and fallback behavior are in [ZERO_COPY_IMPLEMENTATION.md](ZERO_COPY_IMPLEMENTATION.md); the source-memory audit is [ZERO_COPY_FEASIBILITY.md](ZERO_COPY_FEASIBILITY.md).

The target-side runner is [`scripts/run_zero_copy_ablation.py`](scripts/run_zero_copy_ablation.py); it captures Worker/model/video SHA-256, board versions, command, clean JSONL metric/event records, full Worker stdout in `run.raw.log`, stderr and per-run JSON. For the same media-clock matrix, run:

```bash
python3 scripts/run_zero_copy_ablation.py \
  --worker build/bin/rk_edge_worker --model MODEL.rknn --video VIDEO.mp4 \
  --output-dir RESULTS/media --label ppe-media --repeat 3 \
  --pacing media --queue-capacity 1 --queue-policy latest
```

For offline throughput use `--pacing none --queue-capacity 512 --queue-policy fifo`; do not use Latest for full-clip offline throughput. Summaries and SVG charts are generated by [`scripts/summarize_zero_copy_ablation.py`](scripts/summarize_zero_copy_ablation.py).
