# P2 acceptance report — RK3576 media, preprocessing, and quantization probes

- **Date:** 2026-11-05
- **Board:** KICKPI K7 / RK3576 / Ubuntu 24.04.3 aarch64
- **Status:** MPP/RGA stage probes and a native INT8 worker A/B were run on RK3576. INT8 improved throughput on this single sample, but remains experimental because representative calibration and labeled accuracy/event evidence are absent; no optimization has been promoted as the product default.

## Test media and environment

The tests used the first 150 frames of OpenCV's `vtest.avi`, encoded as a 768×576 H.264 Constrained Baseline MP4 at 10 FPS and repeated ten times to form 1,500 frames. The test file SHA-256 is `9b37472e231a56c33a5731f6d4d93c72904c7a5def7d0fde10cc60e04d094126`. The board reports GStreamer 1.24.2, Rockchip MPP GStreamer plugin 1.14.4, MPP 1.5.0, RGA 2.2.0, and RGA API 1.10.1. The test used the existing `mppvideodec`, `videoconvert`, OpenCV FFmpeg, and `librga` installations; it did not install or change board packages.

## MPP decode results

The GStreamer MPP pipeline decoded all 1,500 frames into NV12 on each of three measured runs after one warmup run. The measured segment ended at a fakesink; it did not resize, convert to RGB, run RKNN, or import DMA-BUFs. Median throughput was **1,683.2 frames/s**, with no GStreamer bus warnings in the benchmark records. The MPP element held `/dev/mpp_service` while running.

For a BGR-output comparison, `videoconvert` after MPP produced a median **162.7 frames/s**. OpenCV's FFmpeg software path produced BGR at a median **505.1 frames/s** on the same clip and decoded all 1,500 frames in every run. The GStreamer BGR result includes its conversion/filter path; the NV12-only 1,683.2 frames/s figure must not be compared directly with the BGR result or presented as an end-to-end acceleration ratio. Under these test conditions, the MPP-to-BGR GStreamer path was slower than the OpenCV FFmpeg path.

## RGA preprocessing results

The first operation-level probe resized BGR 768×576 to RGB 640×480. One synchronous `improcess` operation measured 1.345 ms versus 2.331 ms for OpenCV resize plus color conversion; mean absolute pixel difference was 0.197 and maximum difference was 7. This is not the worker's full 640×640 letterbox.

A second probe matched the worker's 640×640 letterbox for 768×576 input: resize to 640×480, pad 80 pixels at the top and bottom with value 114, and convert BGR to RGB. Three batches of 1,000 calls after 30 warmups measured median **2.810 ms/frame** for the current OpenCV path and **2.342 ms/frame** for RGA, about **1.20×** at this isolated stage. The RGA measurement includes CPU filling the 640×640 padding canvas. `improcess` was synchronous; source and destination were CPU virtual addresses. The benchmark process repeatedly held `/dev/rga`, and the call returned output pixels. No DMA-BUF or zero-copy path was used.

The RGA pixels differed slightly from OpenCV (mean absolute difference 0.148–0.202 on the six sampled frames; maximum difference 6–7). With the same FP RKNN model, CPU and RGA inputs were compared on frames 0, 15, 45, 75, 100, and 125. The output tensors changed slightly (minimum cosine similarity across the reported tensor comparisons: 0.999880). At confidence thresholds 0.345 and 0.50, the six detection sets matched at IoU 0.50. At the worker's default confidence 0.25, five frames matched fully; frame 125 produced 10 CPU detections versus 11 RGA detections, with 10 matched. Detection sets were decoded from the saved tensors by the project's Python comparison helper; the application worker itself was not run with RGA preprocessing. The unmatched result may be a false positive or a true detection; there is no ground truth in this sample. These are preprocessing-sensitivity results, not detection-accuracy results.

## INT8 model and worker A/B

Toolkit2 2.0.0b0 converted the same Model Zoo v2.0.0 YOLOv8n graph used for the FP baseline, targeting RK3576. Calibration used 20 unique images sampled at 2 FPS from the first 10 seconds of the sample video. Ten later frames (100, 105, 110, 115, 120, 125, 130, 135, 140, 145) were held out from calibration. This is a small single-scene development sample, not a representative calibration or labeled accuracy set. The INT8 model is 5,118,678 bytes versus 8,317,332 bytes for FP (38.5% smaller). Toolkit2 warned that the graph uses opset 12 while opset 19 is recommended, flagged weight outliers, and changed input/output tensor dtypes to INT8. The full conversion log and image/model hashes are retained in [conversion log](evidence/p2/yolov8n-rk3576-int8-conversion.txt) and [calibration manifest](evidence/p2/int8-calibration-manifest.txt).

On the ten held-out frames, the board-side comparator used identical preprocessed RGB UINT8 NHWC inputs for both models. It requested runtime-dequantized FP32 outputs (`want_float=1`) and used the Model Zoo C++ postprocessor; this isolates output differences but is not the worker's native integer-output postprocess branch. Detection sets matched 9/10 frames at confidence 0.25 (one extra INT8 detection on frame 105), 10/10 at 0.345, and 9/10 at 0.50 (one extra INT8 detection on frame 140). Minimum cosine similarity across the compared output tensors was about 0.953. These comparisons are consistency evidence only. Per-frame `rknn_run`/fetch times are single diagnostic observations; worker A/B below is the performance evidence.

The actual `rk_edge_worker` was then run on the RK3576 with the same 150-frame, 10-FPS H.264 file, confidence 0.25, and three interleaved repetitions per model in each queue mode:

| Queue mode | Model | Median processed FPS | Median processed / dropped | Median worker inference interval | Median end-to-end p95 | Median RSS |
|---|---|---:|---:|---:|---:|---:|
| Capacity 1, latest-frame | FP | 7.602 | 115 / 35 | 71.370 ms | 236.099 ms | 138,716 kB |
| Capacity 1, latest-frame | INT8 | 9.986 | 150 / 0 | 54.577 ms | 115.343 ms | 119,444 kB |
| Capacity 150, preserve-all | FP | 7.965 | 150 / 0 | 75.379 ms | 3,723.155 ms | 149,268 kB |
| Capacity 150, preserve-all | INT8 | 9.984 | 150 / 0 | 46.414 ms | 107.494 ms | 119,448 kB |

The inference interval includes `rknn_run` plus output fetch. In capacity-1 mode, the INT8 worker sustained the 10-FPS input in all three repetitions, while FP dropped 32–36 frames. In capacity-150 mode, both models processed all frames, but FP accumulated a large backlog; that p95 is an offline preserve-all result, not the live latency of a capacity-1 deployment. Temperatures remained about 27–29°C during these runs. Detection and event counts differed, but the queue-1 runs processed different frame subsets and the sample has no labels, so counts do not establish accuracy or event quality. Raw logs and summaries are linked from [queue-1 evidence](evidence/p2/int8-queue1-summary.jsonl) and [queue-150 evidence](evidence/p2/int8-queue150-summary.jsonl); the frame-level input hashes and comparisons are under `evidence/p2/int8-holdout-*`.

The reported inference interval is each run's final `last_inference_ms` sample; its table value is the median of three run-level samples, not the median across every frame. Throughput and RSS are likewise summarized across three runs. The end-to-end p95 is computed over frame latencies within each run and then summarized across runs.

## Negative-input and model-load probes

On RK3576, [the reusable failure probe](scripts/probe_worker_failure_paths.sh) gave the worker a 1,024-byte truncated MP4 and a zero-byte MP4 with the valid FP model, then a 64-byte zero-filled RKNN file with the valid sample video. All three invocations exited nonzero within the 30-second timeout. The two video cases emitted `status=error`, zero decoded/processed/detected frames and no events; the malformed model returned `rknn_init` error `-6`. No hangs or success-shaped events occurred. A separate Python API test simulated SQLite being unavailable by placing a directory at the database path: health stayed responsive, `/api/events` returned 503, and a later request recreated the database after the obstruction was removed. Another board test applied a process-local `RLIMIT_FSIZE` cap to writes in a temporary directory; SQLite returned `disk I/O error` on a large event, the failed write was counted, and a subsequent event persisted after restoring the original limit. Health includes event/metric write-failure counts, and the page continues to show the latest storage error after recovery. These simulations exercise path and write failures without exhausting the filesystem; actual low-disk behavior remains open. RTSP reconnect, permission errors, runtime inference failure, process restart and long-run recovery also remain open. Inputs use temporary directories; raw logs and conditions are recorded in the [SQLite failure manifests](evidence/p2/).

## FIFO and latest-frame overload comparison

The worker now exposes `--queue-policy latest|fifo`; latest replaces the oldest pending frame while FIFO preserves pending order and drops a new arrival while full. Both modes use the same bounded, non-blocking queue and report the selected policy and total dropped frames. C++ tests check the exact replacement/drop result and FIFO pop order; a Python test checks latest as the default and FIFO forwarding from the service to the worker.

Reproduce the three interleaved repetitions on RK3576 with a new output directory:

```bash
bash scripts/benchmark_queue_policy_ab.sh \
  .work/p1-yolov8n-rk3576-fp/yolov8n-rk3576-fp.rknn \
  .work/p2-mpp-test/vtest-150f-h264.mp4 \
  .work/p2-queue-policy-next 3 1 0.25
```

On RK3576, the non-quantized YOLOv8n worker processed the same 150-frame, 10-FPS H.264 sample at confidence 0.25, queue capacity 1, with three interleaved runs per policy:

| Policy | Median processed / dropped | Median throughput | Median end-to-end p50 | Median end-to-end p95 |
|---|---:|---:|---:|---:|
| Latest | 119 / 31 | 7.892 FPS | 172.878 ms | 232.028 ms |
| FIFO | 119 / 31 | 7.859 FPS | 199.284 ms | 263.793 ms |

All runs decoded 150 frames and completed without worker errors; board temperatures were 26.846–28.692°C. Median processed/dropped counts and throughput were nearly identical. Latest had 26.406 ms lower p50 and 31.765 ms lower p95 than FIFO in these runs. The worker's end-to-end values are per-frame decode-to-result latencies; p50/p95 are computed per run then summarized across the three repetitions. Since the two policies process different frames, event and detection counts are not accuracy comparisons. Keep latest as the default for freshness; FIFO remains selectable when preserving the order of already queued frames is preferred. See the [six-run summary](evidence/p2/queue-policy-ab-summary.jsonl), [run manifest](evidence/p2/queue-policy-ab-manifest.txt), per-run JSONL/temperature files, and [board build/test log](evidence/p2/b43354b-board-build-test.txt).

## Decision and remaining P2 gates

- Keep OpenCV CPU preprocessing as the worker default. The isolated RGA stage is about 0.47 ms faster per frame in this test, but that does not establish a meaningful end-to-end gain, and a threshold-sensitive detection difference appeared at confidence 0.25.
- Do not integrate MPP into the live worker based on the NV12/fakesink throughput. The measured GStreamer BGR path was slower than OpenCV FFmpeg, and no MPP-to-RKNN buffer path was measured.
- No zero-copy, asynchronous inference, labeled accuracy, or peak throughput is claimed. INT8 shows a measured throughput and queue-age improvement on this one sample, but remains an experimental model and does not replace the FP default until calibration and labeled evaluation cover representative scenes and event behavior. A production RGA/MPP candidate still needs controlled worker A/B runs across representative media, event/output comparison, and labeled evaluation before it can replace the CPU path.

## Evidence

- [MPP NV12/fakesink benchmark](evidence/p2/mpp-nv12-benchmark.json), [MPP device-FD probe](evidence/p2/mpp-device-fd-probe-samples.txt), [MPP plus BGR conversion](evidence/p2/mpp-bgr-benchmark.json), and [OpenCV FFmpeg BGR baseline](evidence/p2/opencv-cpu-decode-benchmark.jsonl)
- [RGA operation-level comparison](evidence/p2/rga-640x480-operation-benchmark.jsonl), [three exact-letterbox repetitions](evidence/p2/rga-letterbox-640x640-repetitions.jsonl), and [observed `/dev/rga` process descriptors](evidence/p2/rga-device-fd-probe-samples.txt)
- [Six-frame preprocessing and tensor/detection comparison summary](evidence/p2/rga-cpu-vs-rga-six-frame-summary.json), [frame 125 threshold-sensitive detection details](evidence/p2/rga-frame125-detection-details.json), and six per-frame tensor reports
- [Preprocessed tensor SHA-256 values](evidence/p2/rga-preprocessed-input-sha256.txt), [board probe compilation/CLI check](evidence/p2/p2-probe-build-check.txt), and [WSL Python syntax check](evidence/p2/p2-wsl-python-compile-check.txt)
- [Commit/archive synchronization and post-push board build/test](evidence/p2/aa6f500-board-sync-verification.txt), with the [full CMake/CTest/Python output](evidence/p2/aa6f500-post-sync-build-test.txt)
- [Latest commit `739fe95` board sync and validation](evidence/p2/739fe95-board-sync-verification.txt), including the [native build/tests](evidence/p2/739fe95-board-build-test.txt) and [two-model comparator compile/CLI check](evidence/p2/739fe95-board-comparator-build.txt)
- [Latest queue-policy commit `b73e3da` board sync and validation](evidence/p2/b73e3da-board-sync-verification.txt), including [build/tests](evidence/p2/b73e3da-board-build-test.txt) and [CLI checks](evidence/p2/b73e3da-board-cli-check.txt)
- INT8 conversion, calibration hashes, ten held-out tensor/detection comparisons, and input hashes: [conversion transcript](evidence/p2/yolov8n-rk3576-int8-conversion.txt), [calibration record](evidence/p2/int8-calibration-manifest.txt), [holdout input hashes](evidence/p2/int8-holdout-input-sha256.txt)
- Native worker A/B raw records for all three interleaved repetitions: [latest-frame queue summary](evidence/p2/int8-queue1-summary.jsonl) and [preserve-all queue summary](evidence/p2/int8-queue150-summary.jsonl), with per-run stdout/stderr/temperature metadata alongside them
- [Failure-probe manifest and summary](evidence/p2/p2-failure-probes-manifest.txt), with per-case stdout/stderr, file hashes, sizes and exit status in `evidence/p2/failure-*`
- [SQLite recovery board test log](evidence/p2/sqlite-recovery-board-python-tests.txt), including the simulated unavailable-database and retry sequence
- [SQLite recovery manifest](evidence/p2/sqlite-recovery-manifest.txt), with environment, limits, and local/board source hashes
- [SQLite write-failure board test log](evidence/p2/sqlite-write-failure-board-tests.txt) and [manifest](evidence/p2/sqlite-write-failure-manifest.txt), documenting the temporary-file `RLIMIT_FSIZE` exercise and recovery
- [Queue-policy A/B manifest and summary](evidence/p2/queue-policy-ab-manifest.txt) with per-run output and temperature samples in `evidence/p2/queue-policy-*`
- Per-frame letterbox and model-output records are alongside those summaries in `evidence/p2/`.
- Reusable probes: [GStreamer MPP decoder benchmark](scripts/benchmark_gstreamer_mpp_decode.py), [OpenCV FFmpeg decode benchmark](scripts/benchmark_opencv_cpu_decode.cpp), [RGA operation benchmark](scripts/benchmark_rga.cpp), [worker-matched RGA letterbox benchmark](scripts/benchmark_rga_letterbox.cpp), [queue-policy A/B runner](scripts/benchmark_queue_policy_ab.sh), [worker failure-path probe](scripts/probe_worker_failure_paths.sh), [RKNN input/output comparison](scripts/compare_rknn_input_outputs.cpp), [two-model RKNN comparison](scripts/compare_rknn_models.cpp), [Toolkit2 converter](scripts/convert_onnx_to_rknn.py), and [detection-set comparison](scripts/compare_rknn_detection_sets.py).
