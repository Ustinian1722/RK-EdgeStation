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
- P1 follow-up: the project YOLOv8n model loaded and produced detections/events through the board's RKNN C Runtime. One repeated-still 150-frame run recorded the aggregate counts on the `/proc/interrupts` lines labeled `27702000.iommu, 27700000.npu` rising from 7,281 to 12,377 (+5,096), then no change during a two-second idle follow-up. During a later moving-video run, 58 samples rose from 12,377 to 19,626, but a same-boot read at 09:31:45Z showed 19,098. This unresolved counter discrepancy makes the IRQ series exploratory; it is not used as direct utilization evidence. The lines combine IOMMU/NPU and do not expose per-core utilization.
- In a separate 90-sample probe, NPU devfreq stayed at 300 MHz and `/sys/class/devfreq/27700000.npu/load` read `100@300000000Hz`; the same value persisted during a two-second idle check, so it is not treated as a utilization percentage. `/dev/rknpu*` remains absent and the runtime access path is not traced. Raw samples are in [evidence/p1](evidence/p1/).
- Do not replace the runtime, load a driver, or modify kernel/firmware as a workaround without a separate review; preserve the board image.

## Media, cameras, and graphics

| Component | Observed | What remains unproven |
|---|---|---|
| MPP | packages `librockchip-mpp1` and `-dev` 1.5.0-1; `/dev/mpp_service`; GStreamer `mppvideodec` factory exists for H.264/H.265 | No video has yet been decoded through the hardware path |
| RGA | `librga2` and `-dev` 2.2.0-1; `/dev/rga` | No format/scale operation has yet been executed |
| DMA-BUF | `/dev/dma_heap/system` exists | No shared-buffer import or zero-copy path has been tested |
| GStreamer | framework 1.24.2; Rockchip package 1.14-4 | Pipeline negotiation and sustained decode still need a board test |
| OpenCV | development package 4.6.0 | A generated MP4 software encode/decode check and the project CPU decode path passed; MPP hardware decode remains untested |
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
- WSL 2.6.2 is installed. The isolated `RK-EdgeStation-Toolkit2` Ubuntu 22.04 x86-64 distro now resides at `D:\WSL\RK-EdgeStation-Toolkit2-Fresh`; it uses Python 3.10.12 and Toolkit2 2.0.0b0. The earlier partial VHDX is preserved under `D:\WSL\RK-EdgeStation-bootstrap\partial-distro-backup`.
- Toolkit2 conversion is optional host tooling: no Rockchip silicon or Runtime is required on Windows/WSL. Native build and board inference use the RK3576's installed compiler/runtime. The current converter is the pinned Linux x86-64 wheel.

## Decision

The official sample and project YOLOv8n worker both completed through the installed RKNN C Runtime despite absent `/dev/rknpu*` nodes. The platform driver and devfreq are visible, but the runtime access path and accelerator utilization are not traced; the P1 clock sample stayed at 300 MHz. No system change was needed. MPP/RGA still need actual hardware decode/resize tests. Current official Model Zoo 2.5.0 requires RKNPU SDK 2.5.0 or newer, while this board reports 2.0.0b0; use compatible older artifacts and do not silently update the board runtime. See [MODEL_CONVERSION.md](MODEL_CONVERSION.md) and [P1_ACCEPTANCE_REPORT.md](P1_ACCEPTANCE_REPORT.md).
