# Hardware audit — KICKPI K7 / RK3576

Date: 2026-11-05. Source of truth: direct read-only SSH observations from `kickpi@192.168.31.223`. The raw captured command output is in [evidence/p0/hardware-audit.txt](evidence/p0/hardware-audit.txt). SSH public-key authentication was configured in the account's `.ssh` directory and verified in batch mode; the private key remains outside this repository. No system packages, drivers, kernel, boot files, groups, or existing projects were changed.

## Board and OS

| Item | Observed value | Evidence / qualification |
|---|---|---|
| Board | Rockchip RK3576 KICKPI K7 Board | `/proc/device-tree/model` |
| Compatible | `rockchip,rk3576-evb1-v10 rockchip,rk3576` | Device tree |
| OS / kernel | Ubuntu 24.04.3 LTS; Linux 6.1.75, vendor build | `/etc/os-release`, `uname -a` |
| Architecture | aarch64, 64-bit; Debian arch `arm64` | `uname -m`, `getconf`, `dpkg` |
| CPU | 8 cores: 4 Cortex-A53 and 4 Cortex-A72 | `lscpu` |
| CPU frequencies | A53 408–2016 MHz; A72 408–2208 MHz | `lscpu`; current frequency is load-dependent |
| RAM | 8,105,812 kB total; 7.7 GiB reported; no swap | `/proc/meminfo`, `free -h` |
| Root storage | eMMC `mmcblk2`, 58.2 GB; root ext4 58 GB with about 44 GB free | `lsblk`, `df -hT /` |

## NPU and RKNN

- Kernel platform driver `RKNPU` is bound to `27700000.npu`; NPU interrupts and devfreq are present.
- NPU devfreq is currently 300 MHz; available rates are 300, 400, 500, 600, 700, 800, 900, and 950 MHz.
- `/lib/librknnrt.so` reports **RKNPU Runtime 2.0.0b0**, build `35a6907d79@2024-03-24T10:31:14`. `/usr/bin/rknn_server` is present. Both files are outside dpkg ownership.
- `rknn-toolkit2` and `rknn-toolkit-lite2` Python packages are absent on the board.
- `/dev/rknpu` and `/dev/rknpu0` are absent even though the platform driver is bound. The official RK3576 C API sample nevertheless created a context and completed inference; the runtime reported driver 0.9.7. Its device access path has not been traced.
- The RKNPU driver version was not exposed through the readable debugfs paths. The successful RKNN runtime probe reports driver version 0.9.7.
- Do not replace the runtime, load a driver, or modify kernel/firmware as a workaround without a separate review; preserve the board image.

## Media, cameras, and graphics

| Component | Observed | What remains unproven |
|---|---|---|
| MPP | packages `librockchip-mpp1` and `-dev` 1.5.0-1; `/dev/mpp_service`; GStreamer `mppvideodec` factory exists for H.264/H.265 | No video has yet been decoded through the hardware path |
| RGA | `librga2` and `-dev` 2.2.0-1; `/dev/rga` | No format/scale operation has yet been executed |
| DMA-BUF | `/dev/dma_heap/system` exists | No shared-buffer import or zero-copy path has been tested |
| GStreamer | framework 1.24.2; Rockchip package 1.14-4 | Pipeline negotiation and sustained decode still need a board test |
| OpenCV | development package 4.6.0 | CPU video baseline not run yet |
| V4L2 | platform CIF, ISP and VPSS nodes; media devices 0–8 | `/dev/video0` fails with “No such device”; no physical sensor or USB camera was identified |

Many `/dev/video*` entries represent the board's platform pipeline, not proof of an attached camera. The first P1 source will be a local MP4; RTSP follows. USB/MIPI input stays optional until a real sensor is connected and capture succeeds.

## Build tools, load, temperature, and permissions

- GCC/G++ 13.3.0, CMake 3.28.3, Make, pkg-config 1.8.1, Python 3.12.3, OpenCV 4.6.0, GStreamer 1.24.2, and `v4l-utils` 1.26.1 are installed.
- GPU is currently 300 MHz; available 300–900 MHz. VOP is 702 MHz; DMC is 528 MHz. These are snapshot values, not performance limits.
- At audit time, SoC temperature was 24.1 °C and NPU/GPU temperature 23.1 °C. A Type-C source power-supply entry exists, but no board power measurement interface was found.
- User `kickpi` belongs to `video`, `audio`, and `sudo`; not `render`. `/dev/rga` and `/dev/mpp_service` are group `video`; render-node access must be checked before a DRM/RGA path.

## Host toolchain

- Host: Windows 11 build 26100, x64.
- Git 2.51.2 and GitHub CLI are installed; `gh auth status` reports an active login with `repo` scope. No token value is recorded.
- Default Python is 3.13.5; Python 3.12.10 is also present. Ninja is installed. Host CMake, MSVC `cl`, Clang, and GCC are absent.
- WSL 2.6.2 is installed but has no registered Linux distribution. D: had about 14 GiB free at inspection; C: had about 1.2 GiB free. Do not place a large conversion environment on C:.
- Native aarch64 builds can run on the board. Toolkit2 conversion needs a supported Linux x86-64 environment and Python version.

## Decision

The official RKNN C API sample proves that the installed runtime can run an RK3576 model, despite the absence of visible `/dev/rknpu*` nodes. Its access path remains undocumented in this audit; no system change was needed. MPP/RGA still need actual decode/resize tests. Current official Model Zoo 2.5.0 requires RKNPU SDK 2.5.0 or newer, while this board reports 2.0.0b0; use compatible older artifacts and do not silently update the board runtime. Refer to [MODEL_CONVERSION.md](MODEL_CONVERSION.md) for the project model conversion gate.
