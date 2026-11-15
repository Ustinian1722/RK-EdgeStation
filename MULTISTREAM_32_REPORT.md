# RK3576 Single-Worker Multistream and PPE Integration Report

Date: 2026-11-09  
Branch: `research/multistream-32-industrial-ai`  
Base: `5481145116bd9929b666f81bbbbf247b6ed6e49b`  
Board: KICKPI K7 / RK3576, Ubuntu 24.04, Linux 6.1.75, GStreamer 1.24.2, RKNN Runtime 2.0.0b0  
Model: Constructelligence construction-site-safety v0.3 FP RKNN, SHA-256 `70060dcd34d1a0c212639e786bc6bdb1e31c1c6dd885ff6d8c6a40f889f34645`

## Result

The research branch now has a configurable single-process multistream Worker. Each configured camera gets its own capture/reconnect thread and bounded queue; one shared inference manager selects ready queues with weighted fair scheduling and uses one RKNN model/context. The board accepted 1, 2, 4, 8, 16, and 32 distinct H.264 RTSP mounts in one Worker process. All 18 clean runs connected every configured mount and exited successfully.

The shared model path remained the limit: aggregate processing stayed near 4 FPS while the test input grew from about 4.6 to 144 decoded FPS. The 32-source run connected 32/32 sources, but processed only about 0.16 FPS per camera with a 95.1% drop rate. This demonstrates bounded concurrency and scheduler operation; it does **not** establish useful full-rate PPE coverage for 32 cameras.

## Implementation

- `src/multistream.cpp/.hpp` parse a bounded JSON camera config (1–32 cameras), validate unique IDs and source/task settings, redact RTSP credentials in display output, and implement a smooth weighted fair scheduler.
- `src/main.cpp` keeps the existing single-camera path and adds a multi-camera path with one capture thread and per-camera queue/state for each enabled source, independent reconnect epochs, and one serialized RKNN inference engine. The tested defaults were Latest queues with capacity 1. A failed source does not stop healthy cameras.
- `python/edge_station/server.py` adds camera list/status and algorithm readiness endpoints, filtered camera/algorithm event reads, atomic local config updates, and per-camera enable/disable. The server binds to `127.0.0.1`; the Web UI lists camera state without opening dozens of browser streams. Fire/smoke remains `not_ready` because no licensed and accuracy-validated RKNN model is integrated.
- `python/edge_station/storage.py` adds `camera_id` and `algorithm_id` through an in-place SQLite migration. Events and evidence remain separated by camera.
- `deploy/cameras.example.json` documents the local configuration shape. `scripts/rtsp_multisource_test_server.py`, `scripts/run_multistream_benchmark.py`, and `scripts/summarize_multistream_benchmark.py` provide a moving RTSP fixture, board telemetry collection, and results aggregation.

This matrix used `--video-memory-mode copy` and the existing shared RKNN input mode. It was a scheduling/concurrency test, not a DMA-BUF optimization comparison. The model and video pipeline remain selectable; the original single-camera CLI is preserved.

## Board multistream experiment

The development host served separate synthetic H.264 RTSP mounts (`/cam-01` … `/cam-32`), each with a moving test pattern at 320×240 and 5 FPS. The board connected through an SSH reverse tunnel bound to board loopback; it did not receive 32 physical camera feeds over Ethernet. Each condition ran for about 12 seconds with a one-second aligned telemetry sample. There were three repetitions per condition in interleaved order. Every run used the same FP RKNN model, Worker binary, thresholds, Latest queue capacity 1, and model/decode path.

“Per-camera FPS” below is aligned to the full Worker run window (processed frames divided by run duration); the min–median–max span is across cameras, then averaged across repetitions. Drop rate is the Worker's dropped frames divided by its decoded frames. The previous separate-process 1/2/4 results in [the PPE scaling report](CONSTRUCTELLIGENCE_PPE_SCALING_REPORT.md) use a different topology and are not pooled with this single-process experiment.

| RTSP sources | Connected | Aggregate decoded FPS | Aggregate processed FPS, mean ± SD | Per-camera FPS, min–median–max | Drop rate | Inference calls per camera, min / median / max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 4.58 | 3.93 ± 0.08 | 3.93–3.93 (3.93) | 12.95% | 48 / 49 / 49 |
| 2 | 2/2 | 9.20 | 3.87 ± 0.07 | 1.91–1.96 (1.94) | 56.14% | 23 / 24 / 25 |
| 4 | 4/4 | 18.20 | 3.86 ± 0.01 | 0.89–1.05 (0.97) | 76.85% | 11 / 12 / 13 |
| 8 | 8/8 | 36.30 | 3.95 ± 0.03 | 0.48–0.56 (0.48) | 86.84% | 6 / 6 / 7 |
| 16 | 16/16 | 72.42 | 4.01 ± 0.01 | 0.24–0.32 (0.24) | 92.28% | 3 / 3 / 4 |
| 32 | 32/32 | 144.06 | 4.09 ± 0.01 | 0.08–0.19 (0.16) | 95.14% | 1 / 2 / 3 |

| Sources | `rknn_run` P50 / P95 ms* | Decode-to-result P50 / P95 ms* | Worker CPU, one-core equivalent | System CPU, mean across cores | Worker RSS peak, mean / max MiB | Min system memory available MiB | Max SoC temperature | NPU utilization / frequency |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 226 / 259 | 312 / 425 | 45.2% | 6.4% | 125.5 / 131.1 | 7,017 | 25.9 °C | unavailable / 300 MHz |
| 2 | 224 / 260 | 324 / 422 | 46.3% | 6.7% | 125.9 / 131.6 | 7,011 | 25.9 °C | unavailable / 300 MHz |
| 4 | 225 / 258 | 325 / 410 | 50.0% | 7.5% | 127.0 / 132.2 | 7,003 | 25.9 °C | unavailable / 300 MHz |
| 8 | 215 / 247 | 304 / 416 | 54.6% | 7.9% | 131.4 / 134.2 | 6,985 | 25.9 °C | unavailable / 300 MHz |
| 16 | 215 / 233 | 303 / 377 | 65.0% | 9.9% | 130.7 / 132.2 | 6,951 | 26.8 °C | unavailable / 300 MHz |
| 32 | 209 / 217 | 298 / 350 | 83.2% | 13.3% | 138.2 / 138.7 | 6,857 | 27.8 °C | unavailable / 300 MHz |

`*` The timing columns are the mean, over three repetitions, of the median across cameras of each camera's reported percentile. At 16 sources each camera completed only 3–4 inferences per short run; at 32 it completed 1–3. Therefore the high-count P95 values are descriptive snapshots, not stable tail-latency estimates. The run- and camera-level call counts are retained beside the latency values in the raw summaries.

Worker CPU is measured in one-core-equivalent percent; system CPU is the mean of per-core utilization over the sampling interval. Worker peak RSS rose from about 125 MiB to 138 MiB. Minimum available system RAM stayed above 6.8 GiB. The highest sampled thermal-zone temperature was 27.8 °C and no cooling-device state became nonzero during these short runs. This is not a long-duration thermal qualification.

The NPU busy/load counter is **unavailable**. Before testing, `/sys/class/devfreq/27700000.npu/load` reported `100@300000000Hz` while `/sys/class/devfreq/27700000.npu/device/power/runtime_status` was `suspended`; this fails the idle sanity check. The monitor retained raw values but did not treat NPU frequency, IRQs, or `rknn_run` duration as utilization. Current NPU frequency remained 300 MHz in these samples. The flat throughput, 209–226 ms `rknn_run` P50, increasing Worker CPU and rising drop rate are indirect evidence that serialized inference and per-source ingest compete for a roughly four-frame-per-second budget; NPU saturation itself is not directly proven.

Latest-frame scheduling bounded latency by replacing stale queued work, but did so by dropping increasingly many decoded frames. It does not make a 5-FPS PPE detector equivalent to a 5-FPS camera when it processes 0.16 FPS per stream at 32 sources. No tested configuration here is a suitable 32-camera full-rate industrial safety deployment.

## Dynamic PPE and model evidence

The existing 34-image Roboflow `construction-site-safety` v1 Test set and its hashed labels were already present in the project evidence, so this continuation reused that evaluation and did not need another dataset download. The FP RKNN result was AP50 0.4504 and AP50:95 0.2790 on the 34-image set, with the native label map and preprocessing documented in the [Constructelligence evaluation report](CONSTRUCTELLIGENCE_PPE_SCALING_REPORT.md). At confidence 0.25, key-category recall was Person 1.000 (4 boxes), Helmet/Hardhat 0.750 (32), Vest 1.000 (2), No-Helmet 0.571 (7), and No-Vest 0.333 (3). The very small Person/Vest/No-Vest ground-truth counts make those per-class estimates unstable. The model-card result of 0.531/0.337 used a different evaluation protocol with pseudo-label additions and is not a matched comparison.

A separate controlled 20-second board API/Worker smoke used the existing real moving construction clips: the 34-second Haringey recording and the 5.24-second `Men at work in building site` clip. Both inputs connected; each received 30 inferences. The Worker decoded 446 frames and dropped 384, and SQLite passed `integrity_check`. It recorded 22 ROI-enter, 2 dwell, and 11 PPE event rows (7 `no_vest`, 4 `no_helmet`). The short clip looped and reached stream epoch 2, so these are replay event rows, not counts of unique people or independent event-accuracy results. A representative evidence frame is included below; it is one model-generated event image, not an independently annotated ground-truth sample.

The board-local HTTP control smoke also disabled `zone-b`, confirmed it stopped while `zone-a` continued, then replaced it with `zone-c` through `PUT /api/cameras/config`; `zone-c` connected and began processing. Its sanitized state snapshots are in [`api-config-control.json`](evidence/multistream-32-20261109/api-config-control.json). The first control harness later left its service alive because it terminated an intermediate shell; the run is used only as functional API evidence. The service was found, stopped, and the clean performance matrix was rerun separately.

![Board PPE event evidence frame from the CC0 construction clip](evidence/multistream-32-20261109/dynamic-api-clean-20261109/ppe-2-ppe-violation.jpg)

The source attribution and license notes remain in [the earlier dynamic evaluation](DYNAMIC_VIDEO_EVALUATION.md): the Haringey clip is CC BY-SA 4.0 and the short `Men at work` clip is CC0 1.0. Neither is an official companion video for the Constructelligence model. The weights are AGPL-3.0; Roboflow v1 Test images/labels are CC BY 4.0. Model weights, dataset images, and source videos are not committed.

## Reproducibility and retained data

- `scripts/rtsp_multisource_test_server.py` serves up to 32 independent moving H.264 mounts using GStreamer/OpenH264. It is a synthetic transport/decoder fixture, not PPE data.
- `scripts/run_multistream_benchmark.py` runs a board Worker and saves one-second telemetry (`/proc/stat`, process CPU/RSS, memory, CPU/NPU frequency, temperature, cooling state) plus the complete Worker JSON stream. It records `npu_utilization: unavailable`.
- `scripts/summarize_multistream_benchmark.py` builds the checked-in `summary.csv`, `summary.json`, and `resource-curves.svg` from the clean run folder.
- The 18 clean runs, aligned raw telemetry, compressed Worker records, model/config/Worker hashes, and per-camera counters are in [`evidence/multistream-32-20261109/`](evidence/multistream-32-20261109/). The clean test matrix contains 3 runs at each of 1, 2, 4, 8, 16, and 32 sources.
- An earlier 18-run attempt is preserved in [`confounded-prior/`](evidence/multistream-32-20261109/confounded-prior/), but excluded: its API smoke was left alive by a harness that killed an intermediate shell instead of the Python server, so a second Worker ran concurrently. The issue was identified from board `/proc` state and SQLite timestamps; both residual processes were stopped, the empty-board idle probe was repeated, and all reported scale results above were then rerun. See the accompanying [exclusion note](evidence/multistream-32-20261109/confounded-prior/NOTE.md).

Python service/storage tests ran 16 cases: 15 passed and one POSIX-only `RLIMIT_FSIZE` case was skipped on Windows. On RK3576, the worker and C++ tests built in an isolated project staging directory and CTest passed 2/2. The live multistream matrix passed 18/18 Worker exits; the controlled dynamic API smoke exited cleanly with no leftover Worker and a valid SQLite database.

## Assessment

This is a validated single-process multistream software path and a useful scheduling/resource-observation portfolio artifact. It is **not** a validated 32-camera PPE capacity solution. The main measured constraint is a shared, serialized FP RKNN Worker at about 4 aggregate inferences per second; at 32 live sources, CPU and queues remain operational but almost all camera frames are dropped. NPU busy percentage cannot be stated with this driver, and high-source-count P95 estimates need longer per-camera observation windows before they can support a service-level claim.
