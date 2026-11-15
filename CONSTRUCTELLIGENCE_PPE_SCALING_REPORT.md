# Constructelligence PPE RKNN and Camera-Scale Evaluation

Date: 2026-11-07
Branch: `research/dynamic-video-dmabuf-20261107`
Worker build: RK3576 build from the current worktree, RKNN Runtime 2.0.0b0
Model: Constructelligence construction-site-safety-hazards v0.3, non-INT8 RKNN
Model SHA-256: `70060dcd34d1a0c212639e786bc6bdb1e31c1c6dd885ff6d8c6a40f889f34645`

## Model and Test Split

The selected release is YOLOv8s v0.3 with a 640×640 RGB letterbox input and a raw `[1,18,8400]` output (4 box channels and 14 class scores). The board RKNN reports FP16 input/output tensor attributes; the Worker requests floating-point output conversion before its minimal YOLOv8 PPE decoder. The model remains non-INT8. The decoder maps the released model's native classes to the project's existing PPE taxonomy and ignores unsupported classes. [Model card and license](https://huggingface.co/constructelligence/construction-site-safety-hazards/blob/main/README.md)

The test split is Roboflow `construction-site-safety` v1 (`original_raw-images`): 34 images and 34 original YOLO label files, downloaded through the official dataset export. The version page lists 398 images total, including these 34 test images, and CC BY 4.0. [Roboflow v1 page](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/1)

Evaluation used the native label names and a checked map: `person→Person`, `hardhat→Hardhat`, `no-hardhat→NO-Hardhat`, `safety vest→Safety Vest`, `no-safety vest→NO-Safety Vest`, `no-mask→NO-Mask`, `gloves→Gloves`, `safety shoes→Safety Shoes`, `excavator→EXCAVATORS`, `wheel loader→wheel loader`, and `dump truck→dump truck`. Classes absent from the dataset labels were not scored. Input preprocessing matched the release contract: RGB, 640×640, grey 114 letterbox; candidate floor 0.001 and class-aware NMS IoU 0.7. AP uses 101-point interpolation and averages only native classes with at least one ground-truth box.

| Evaluation | AP50 | AP50:95 | Scope |
|---|---:|---:|---|
| RK3576 RKNN v0.3 | 0.4504 | 0.2790 | 34 images; 9 native classes with ground truth |
| Host ONNX v0.3 | 0.4520 | 0.2803 | Same files, labels and evaluation code |
| Model-card source result | 0.531 | 0.337 | Same 34-image source, with the model authors' evaluation labels/protocol |

The RKNN and ONNX results are close on this split. The authors' source result is higher, but it is not an exact protocol match: the model card states that people were added with COCO YOLOv8s pseudo-labels for two incompletely labelled sources. The downloaded v1 labels contain only 4 `Person` boxes. Therefore the table is a comparison, not a claim of a controlled replication.

| Native class | GT boxes | AP50 | AP50:95 | Recall at confidence 0.25 | Precision at confidence 0.25 |
|---|---:|---:|---:|---:|---:|
| Person | 4 | 0.143 | 0.110 | 1.000 | 0.095 |
| Hardhat | 32 | 0.683 | 0.260 | 0.750 | 0.686 |
| Safety Vest | 2 | 0.417 | 0.134 | 1.000 | 0.200 |
| NO-Hardhat | 7 | 0.843 | 0.444 | 0.571 | 0.571 |
| NO-Safety Vest | 3 | 0.174 | 0.089 | 0.333 | 0.111 |

The operating-point precision for Person is strongly affected by the source's sparse Person labels. Safety Vest and NO-Safety Vest have only 2 and 3 boxes, so their estimates are very unstable. The low violation-class recall, especially NO-Safety Vest, means this model is not a safety-rated compliance detector.

The isolated, unpaced RKNN evaluation processed 34 images at 3.88 FPS. `rknn_run` P50/P95 was 184.6/215.8 ms, process CPU was 52.2% of one core (6.5% of the 8-core system), and peak RSS was 157,908 KiB. These are a model-evaluation workload, not real-time video results.

## Dynamic Video Worker Results

The Haringey clip is a real moving-camera construction-site recording by Acabashi, released under CC BY-SA 4.0. The separate short `Men at work in building site` clip is by Akum20, CC0 1.0. Neither is an official model-card video. [Haringey source](https://commons.wikimedia.org/wiki/File:Building_construction_Moira_Close_Broadwater_Farm_Haringey_2025_16.webm) · [CC0 source](https://commons.wikimedia.org/wiki/File:Men_at_work_in_building_site.webm)

| Clip | Pipeline | Decoded / processed / dropped | Worker FPS | End-to-end P95 | PPE and event observation |
|---|---|---:|---:|---:|---|
| Haringey, 34 s, 1280×720, 15 FPS | MPP + RGA + retained DMA-BUF + shared RKNN input | 510 / 147 / 363 | 4.269 | 325 ms | 4 Person-only ROI entries, 3 dwell transitions, 0 PPE violation alerts, 1 unknown PPE person; Worker track log records helmet and vest detections |
| Men at work, 5.24 s, 640×1138, 15 FPS | MPP + mapped NV12 copy + RGA + shared RKNN input | 79 / 24 / 55 | 4.300 | 321 ms | 7 Person ROI entries and 3 PPE violation events: 2 `no_vest`, 1 `no_helmet`; alerts were associated to Person tracks and coincide with visibly uncovered workers in sampled frames |

For the second clip, retained DMA-BUF mode failed before decoding: the GStreamer MPP buffer did not satisfy the Worker's valid-FD, contiguous linear NV12 plane-layout requirements. The failure and stderr are retained in the evidence folder. The mapped-copy fallback completed the same video and emitted PPE events; it copied 86,305,920 source NV12 bytes over the 79 decoded frames. The first clip did pass the retained DMA-BUF route and reported zero explicit source-copy bytes.

These short replays are functional diagnostics. The clips do not have dense independent PPE annotations, and the second clip drops 69.6% of frames at this model's rate. Event precision/recall and real-camera alarm quality are not established.

## Simulated 1/2/4 Camera Scaling

Each camera is a separate Worker process replaying the same 34-second Haringey file at its 15 FPS media clock, with its own state directory. All runs used the same model SHA, GStreamer/MPP + RGA + retained DMA-BUF, shared RKNN input, confidence 0.25, Latest queue capacity 1, and a 1.0-second telemetry sample interval. This is process-level replay concurrency, not single-process multistream scheduling. One short run was recorded at each scale.

| Processes | Per-process FPS | Total FPS / inference calls | Drops per stream | `rknn_run` P50 / worst P95 | Worst end-to-end P95 | System CPU avg / max | Mean per-process CPU (core-equivalent) | Max aggregate Worker RSS | System RAM used, avg / max | NPU busy / freq | Max SoC temp |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| 1 | 4.269 | 4.269 / 147 | 71.2% | 219.9 / 267.2 ms | 325.4 ms | 6.4 / 11.2% | 46.1% | 130 MiB | 754 / 762 MiB | unavailable / 300 MHz | 27.8°C |
| 2 | 4.514, 4.512 | 9.026 / 310 | 69.6% | 206–208 / 247.0 ms | 310.3 ms | 12.3 / 25.0% | 44.1%, 44.2% | 257 MiB | 868 / 882 MiB | unavailable / 300 MHz | 28.7°C |
| 4 | 3.027, 2.857, 2.884, 2.856 | 11.624 / 403 | 79.4–80.6% | 323.6–325.8 / 379.2 ms | 446.7 ms | 19.7 / 78.2% | 33.4%, 42.1%, 33.2%, 40.7% | 522 MiB | 1,100 / 1,121 MiB | unavailable / 300 MHz | 33.3°C |

The 2-process run delivered 2.11× the 1-process throughput while keeping each stream near 4.5 FPS. Four processes reached 2.72× total throughput, but per-stream throughput fell to about 2.9 FPS, inference P95 increased to 367–379 ms, and drops rose to about 80%. Latest-frame preserved lower-age work by discarding queued frames; none of these configurations met the 15 FPS source rate.

Per-process latency values retained from the same runs: for 2 processes, `rknn_run` P95 was 247.0/242.4 ms and end-to-end P95 was 309.8/310.3 ms; for 4 processes these were 367.4/377.3/377.8/379.2 ms and 440.3/446.7/430.8/424.4 ms. Sampled per-process RSS peaked at 129.7 MiB for one Worker, 128.3 MiB per Worker at two, and 130.4 MiB per Worker at four. The CSV retains each CPU core and process's one-second samples.

The kernel exposes `/sys/class/devfreq/27700000.npu/load`, but the idle sanity check returned `100@300000000Hz` while NPU runtime power status was `suspended`. The monitor therefore records **NPU utilization: unavailable** and never treats that value or frequency as a busy counter. The sampled current-frequency node remained at 300 MHz. Per-core CPU values, worker CPU/RSS, memory, all available thermal zones, cooling-device states, and CPU policy frequencies are retained at one-second cadence. Cooling states remained zero and no thermal throttling was observed in these short runs; this does not establish long-duration thermal behavior. The 4-way latency growth is indirect evidence of shared-resource contention, not proof of NPU saturation.

Telemetry files and curves: [resource curves](evidence/constructelligence-ppe-scaling/board/resource-curves.svg), [per-sample CSV](evidence/constructelligence-ppe-scaling/board/camera-scale-monitor.csv), [summary JSON](evidence/constructelligence-ppe-scaling/board/camera-scale-summary.json), and the raw per-run `monitor.jsonl` and `summary.json` files under `evidence/constructelligence-ppe-scaling/board/`. Dynamic event and track logs are under `evidence/constructelligence-ppe-scaling/dynamic/`.

## Reproducibility, Licensing, and Outcome

The Roboflow export manifest stores the ZIP and per-file SHA-256 hashes; the board model report stores the model, preprocessing, timing, and prediction hashes. The 34 source images, labels, video files, and `.rknn` weight file remain local and are not committed. The Constructelligence model card declares the weights AGPL-3.0 and training data CC BY 4.0; the Roboflow v1 dataset is CC BY 4.0. Review the model's AGPL terms before redistributing or offering a network service using these weights.

The minimal C++ Worker adapter for the released `[1,18,8400]` floating output and its unit test were built and passed on RK3576. The FP model ran through the existing Worker pipeline without changing the broader MPP/RGA/RKNN architecture. The current evidence supports the 2-process configuration as the best tested balance for this workload; it does not prove performance for live independent cameras or support a single-process multistream claim.
