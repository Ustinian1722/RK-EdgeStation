# Build and deploy

Status: native RK3576 build/tests and still/moving project-model detection-to-event/API smoke runs pass. Full P1 acceptance remains open for tracker continuity, numerical model comparison, labeled accuracy, and reliable NPU-utilization evidence.

## Current supported development path

The board is aarch64 Ubuntu 24.04.3 with GCC/G++ 13.3 and CMake 3.28.3, so native C++17/CMake builds are available without adding a cross toolchain. The Windows host has no C++ compiler; the application is built on the board. Toolkit2 is only needed to create or convert a project model. The app's build and inference path uses the board's existing RKNN C runtime.

Board's dedicated project path is `/home/kickpi/projects/RK-EdgeStation`. It was created for this project without touching existing user project directories.

## Planned reproducible build interface

From the project root on the board, build and run the local checks with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
PYTHONPATH=python python3 -m unittest discover -s tests/python -v
```

The same sequence is available as `bash scripts/build_and_test_board.sh`; it passed on the target board on 2026-11-05 after the ROI event and media-time changes. Launch a local MP4/model run with `bash scripts/run_demo.sh MODEL.rknn INPUT.mp4`. The UI listens on `127.0.0.1:8080` by default. The worker supports `--dwell-ms N` (default 5000) and `--roi-debounce-frames N` (default 2); the Python service forwards both options. Dwell time follows monotonicized media timestamps, using nominal FPS or decode elapsed time when timestamps are invalid. Model files and run state stay outside Git. The first project model was converted in the isolated WSL Toolkit2 environment and run from the board workspace; see [MODEL_CONVERSION.md](MODEL_CONVERSION.md) and [P1_ACCEPTANCE_REPORT.md](P1_ACCEPTANCE_REPORT.md).

The worker discovers board OpenCV and RKNN headers/libraries and links dynamically to those installed libraries. The current P1 reference path uses OpenCV CPU decode/preprocessing and paces file input at its reported media rate when valid; MPP/RGA integration is deferred until its measured comparison in P2. Python's standard library provides the local HTTP server and SQLite store.

## Deployment controls

- Deploy only to `/home/kickpi/projects/RK-EdgeStation`.
- Use the task-specific SSH key under the user's `.ssh` directory; do not copy private keys or passwords to the board or repo.
- Do not run `sudo`, modify groups, alter system libraries, replace kernel/drivers, or flash storage.
- The management service binds to `127.0.0.1`; no LAN listener is enabled by default.
- The board runtime reports 2.0.0b0. The official RKNN C API smoke test passed even though `/dev/rknpu*` nodes were not visible.
- Toolkit2 2.0.0b0 conversion is available in the isolated Ubuntu 22.04 x86-64 WSL2 distro, but no Rockchip silicon or runtime is installed on the host. Conversion is optional once a compatible `.rknn` exists. Model inference and the deployed application run on RK3576.
- The project YOLOv8n path completed 16/16 repeated still-image frames and generated five stored events. The moving sample API run after tracker refinement completed with 129 processed, 21 dropped, 1,021 detections, and 29 stored events. In a separate three-run, full-150-frame comparison, the original greedy IoU tracker emitted 33 enter events per run and the motion-predicted candidate emitted 29; this is not an identity-accuracy result. Two low-confidence `skis` predictions remain.
- The queue-capacity-150 tracker comparison preserved every input frame but accumulated a 3.5–4.6 s p95 end-to-end delay. Keep queue capacity 1 for the live freshness path until scheduling is profiled; the larger queue was used only to compare tracker behavior on identical frames.
- ROI enter/leave transitions require the configured number of consecutive processed frames inside/outside. Dwell emits once per confirmed visit. Current occupancy counts confirmed tracks; when a track expires while inside, the count decreases but no synthetic leave event is emitted. The board event run used a queue of 150 for reproducible frame-by-frame behavior and is not a live latency result.
- During repeated-still queue probes, the worker processed 88/150 and 91/150 frames at capacity 1. One run recorded shared NPU/IOMMU interrupt counts +5,096 during inference and 0 change during a two-second idle check. In the dynamic run, a later same-boot read was lower than the captured sample, so IRQ evidence remains exploratory. `/dev/rknpu*` remains absent, per-core utilization is not exposed, and devfreq stayed at 300 MHz.
- systemd installation and a LAN-bound API are P3 tasks and require review of their exact unit/configuration before activation.

## Deployment evidence

Every deployable release will record source commit, compiler/CMake versions, dependency versions, model SHA-256, command line, configuration, board identity, and run log. Build success alone is not inference success.
