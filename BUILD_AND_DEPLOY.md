# Build and deploy

Status: P1 worker and local dashboard source added; native RK3576 build and unit checks pass; project-model end-to-end run is pending.

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

The same sequence is available as `bash scripts/build_and_test_board.sh`. It passed on the target board on 2026-11-05. Launch a local MP4/model run with `bash scripts/run_demo.sh MODEL.rknn INPUT.mp4`. The UI listens on `127.0.0.1:8080` by default. Model files and run state stay outside Git. The actual project-model run remains pending because conversion has not completed.

The worker discovers board OpenCV and RKNN headers/libraries and links dynamically to those installed libraries. The current P1 reference path uses OpenCV CPU decode/preprocessing and paces file input at its reported media rate when valid; MPP/RGA integration is deferred until its measured comparison in P2. Python's standard library provides the local HTTP server and SQLite store.

## Deployment controls

- Deploy only to `/home/kickpi/projects/RK-EdgeStation`.
- Use the task-specific SSH key under the user's `.ssh` directory; do not copy private keys or passwords to the board or repo.
- Do not run `sudo`, modify groups, alter system libraries, replace kernel/drivers, or flash storage.
- The management service binds to `127.0.0.1`; no LAN listener is enabled by default.
- The board runtime reports 2.0.0b0. The official RKNN C API smoke test passed even though `/dev/rknpu*` nodes were not visible.
- Full conversion is a host software task: the Windows host needs a compatible Linux x86-64 Python environment for this pinned Toolkit2 wheel, but no Rockchip silicon. Model inference and the deployed application run on RK3576.
- systemd installation and a LAN-bound API are P3 tasks and require review of their exact unit/configuration before activation.

## Deployment evidence

Every deployable release will record source commit, compiler/CMake versions, dependency versions, model SHA-256, command line, configuration, board identity, and run log. Build success alone is not inference success.
