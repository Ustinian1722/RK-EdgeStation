# Build and deploy

Status: P0 inventory only; a project build has not yet been performed.

## Current supported development path

The board is aarch64 Ubuntu 24.04.3 with GCC/G++ 13.3 and CMake 3.28.3, so native C++17/CMake builds are available without adding a cross toolchain. The host Windows installation has Git, Python 3.12/3.13, and Ninja but no CMake or C++ compiler. Toolkit2 model conversion is a separate host-side toolchain concern.

Board's dedicated project path is `/home/kickpi/projects/RK-EdgeStation`. It did not exist at audit time. It will be created for this project without touching `/home/kickpi/soc_cpp`, `/home/kickpi/rklstmt`, or other existing directories.

## Planned reproducible build interface

After P1 source exists, the root will expose commands equivalent to:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRK_EDGE_ENABLE_RKNN=ON
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

The CPU-only CI configuration will set `RK_EDGE_ENABLE_RKNN=OFF` and use deterministic mocks. A board build will discover system CMake packages/headers and link to the installed RKNN/MPP/RGA libraries without copying those binaries into the repository.

## Deployment controls

- Deploy only to `/home/kickpi/projects/RK-EdgeStation`.
- Use the task-specific SSH key under the user's `.ssh` directory; do not copy private keys or passwords to the board or repo.
- Do not run `sudo`, modify groups, alter system libraries, replace kernel/drivers, flash storage, or enable a network listener during early phases.
- Board runtime currently reports 2.0.0b0 and no `/dev/rknpu` node; record this before any runtime test.
- systemd installation and a LAN-bound API are P3 tasks and require review of their exact unit/configuration before activation.

## Deployment evidence

Every deployable release will record source commit, compiler/CMake versions, dependency versions, model SHA-256, command line, configuration, board identity, and run log. Build success alone is not inference success.
