# P2 acceptance report — RK3576 media and preprocessing probes

- **Date:** 2026-11-05
- **Board:** KICKPI K7 / RK3576 / Ubuntu 24.04.3 aarch64
- **Status:** MPP decode and RGA preprocessing were exercised on the board. No optimized end-to-end worker path is accepted yet.

## Test media and environment

The tests used the first 150 frames of OpenCV's `vtest.avi`, encoded as a 768×576 H.264 Constrained Baseline MP4 at 10 FPS and repeated ten times to form 1,500 frames. The test file SHA-256 is `9b37472e231a56c33a5731f6d4d93c72904c7a5def7d0fde10cc60e04d094126`. The board reports GStreamer 1.24.2, Rockchip MPP GStreamer plugin 1.14.4, MPP 1.5.0, RGA 2.2.0, and RGA API 1.10.1. The test used the existing `mppvideodec`, `videoconvert`, OpenCV FFmpeg, and `librga` installations; it did not install or change board packages.

## MPP decode results

The GStreamer MPP pipeline decoded all 1,500 frames into NV12 on each of three measured runs after one warmup run. The measured segment ended at a fakesink; it did not resize, convert to RGB, run RKNN, or import DMA-BUFs. Median throughput was **1,683.2 frames/s**, with no GStreamer bus warnings in the benchmark records. The MPP element held `/dev/mpp_service` while running.

For a BGR-output comparison, `videoconvert` after MPP produced a median **162.7 frames/s**. OpenCV's FFmpeg software path produced BGR at a median **505.1 frames/s** on the same clip and decoded all 1,500 frames in every run. The GStreamer BGR result includes its conversion/filter path; the NV12-only 1,683.2 frames/s figure must not be compared directly with the BGR result or presented as an end-to-end acceleration ratio. Under these test conditions, the MPP-to-BGR GStreamer path was slower than the OpenCV FFmpeg path.

## RGA preprocessing results

The first operation-level probe resized BGR 768×576 to RGB 640×480. One synchronous `improcess` operation measured 1.345 ms versus 2.331 ms for OpenCV resize plus color conversion; mean absolute pixel difference was 0.197 and maximum difference was 7. This is not the worker's full 640×640 letterbox.

A second probe matched the worker's 640×640 letterbox for 768×576 input: resize to 640×480, pad 80 pixels at the top and bottom with value 114, and convert BGR to RGB. Three batches of 1,000 calls after 30 warmups measured median **2.810 ms/frame** for the current OpenCV path and **2.342 ms/frame** for RGA, about **1.20×** at this isolated stage. The RGA measurement includes CPU filling the 640×640 padding canvas. `improcess` was synchronous; source and destination were CPU virtual addresses. The benchmark process repeatedly held `/dev/rga`, and the call returned output pixels. No DMA-BUF or zero-copy path was used.

The RGA pixels differed slightly from OpenCV (mean absolute difference 0.148–0.202 on the six sampled frames; maximum difference 6–7). With the same FP RKNN model, CPU and RGA inputs were compared on frames 0, 15, 45, 75, 100, and 125. The output tensors changed slightly (minimum cosine similarity across the reported tensor comparisons: 0.999880). At confidence thresholds 0.345 and 0.50, the six detection sets matched at IoU 0.50. At the worker's default confidence 0.25, five frames matched fully; frame 125 produced 10 CPU detections versus 11 RGA detections, with 10 matched. Detection sets were decoded from the saved tensors by the project's Python comparison helper; the application worker itself was not run with RGA preprocessing. The unmatched result may be a false positive or a true detection; there is no ground truth in this sample. These are preprocessing-sensitivity results, not detection-accuracy results.

## Decision and remaining P2 gates

- Keep OpenCV CPU preprocessing as the worker default. The isolated RGA stage is about 0.47 ms faster per frame in this test, but that does not establish a meaningful end-to-end gain, and a threshold-sensitive detection difference appeared at confidence 0.25.
- Do not integrate MPP into the live worker based on the NV12/fakesink throughput. The measured GStreamer BGR path was slower than OpenCV FFmpeg, and no MPP-to-RKNN buffer path was measured.
- No zero-copy, quantized model, asynchronous inference, labeled accuracy, peak throughput, or end-to-end speedup is claimed. A production RGA/MPP candidate still needs controlled worker A/B runs across representative media, event/output comparison, and labeled evaluation before it can replace the CPU path.

## Evidence

- [MPP NV12/fakesink benchmark](evidence/p2/mpp-nv12-benchmark.json), [MPP device-FD probe](evidence/p2/mpp-device-fd-probe-samples.txt), [MPP plus BGR conversion](evidence/p2/mpp-bgr-benchmark.json), and [OpenCV FFmpeg BGR baseline](evidence/p2/opencv-cpu-decode-benchmark.jsonl)
- [RGA operation-level comparison](evidence/p2/rga-640x480-operation-benchmark.jsonl), [three exact-letterbox repetitions](evidence/p2/rga-letterbox-640x640-repetitions.jsonl), and [observed `/dev/rga` process descriptors](evidence/p2/rga-device-fd-probe-samples.txt)
- [Six-frame preprocessing and tensor/detection comparison summary](evidence/p2/rga-cpu-vs-rga-six-frame-summary.json), [frame 125 threshold-sensitive detection details](evidence/p2/rga-frame125-detection-details.json), and six per-frame tensor reports
- [Preprocessed tensor SHA-256 values](evidence/p2/rga-preprocessed-input-sha256.txt), [board probe compilation/CLI check](evidence/p2/p2-probe-build-check.txt), and [WSL Python syntax check](evidence/p2/p2-wsl-python-compile-check.txt)
- Per-frame letterbox and model-output records are alongside those summaries in `evidence/p2/`.
- Reusable probes: [GStreamer MPP decoder benchmark](scripts/benchmark_gstreamer_mpp_decode.py), [OpenCV FFmpeg decode benchmark](scripts/benchmark_opencv_cpu_decode.cpp), [RGA operation benchmark](scripts/benchmark_rga.cpp), [worker-matched RGA letterbox benchmark](scripts/benchmark_rga_letterbox.cpp), [RKNN input/output comparison](scripts/compare_rknn_input_outputs.cpp), and [detection-set comparison](scripts/compare_rknn_detection_sets.py).
