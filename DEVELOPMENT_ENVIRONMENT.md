# Development environments

Status: the Windows/WSL Toolkit2 setup and the native RK3576 build path were re-verified on 2026-11-05.

## Which machine does what

| Environment | Use |
|---|---|
| Windows | Edit the project at `D:\RK-EdgeStation`, keep local model/video artifacts under the ignored `.work` directory, and connect to the board over SSH. |
| WSL2 RK-EdgeStation-Toolkit2 | Optional x86-64 ONNX-to-RKNN conversion and offline model diagnostics. It has Ubuntu 22.04, Python 3.10.12, and Rockchip Toolkit2 2.0.0b0. |
| KICKPI K7 / RK3576 | Build the C++ application with the board's GCC, CMake, OpenCV, and installed RKNN C Runtime; run inference, event storage, and the local UI. |

For deployment and inference, RK3576 carries its own RKNN C Runtime; the host needs no Rockchip silicon or Runtime. Once a compatible `.rknn` model exists, the application can be built and run entirely on RK3576. The host-side WSL Toolkit2 is an optional engineering tool for converting new model versions and offline diagnostics.

## Open and check the WSL environment

From Windows PowerShell:

```powershell
wsl -d RK-EdgeStation-Toolkit2 --cd /mnt/d/RK-EdgeStation
```

Inside WSL, verify the Toolkit2 imports and Python dependencies:

```bash
rknn-python -c 'import sys, rknn.api, onnx, torch; print(sys.version); print(rknn.api.__file__); print("onnx", onnx.__version__, "torch", torch.__version__)'
rknn-python -m pip check
```

The 2026-11-05 verification reported Python 3.10.12, ONNX 1.14.1, PyTorch 2.0.1+cpu, rknn.api under /opt/rknn-toolkit2-2.0.0b0, and no broken requirements. See [model conversion](MODEL_CONVERSION.md) for conversion settings and recorded model hashes.

## Repair and preserve the distro

The registered distro is installed at `D:\WSL\RK-EdgeStation-Toolkit2-Fresh`. To re-apply cached dependencies and verify Toolkit2 without re-importing the distro, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\WSL\RK-EdgeStation-bootstrap\Install-RK-EdgeStation.ps1
```

The workstation-specific bootstrap guide is `D:\WSL\RK-EdgeStation-bootstrap\README.md`. It documents the offline wheelhouse, proxy setup, and fresh-import script. `Import-RK-EdgeStation.ps1` refuses to overwrite an existing distro path. Keep the older partial VHDX at `D:\WSL\RK-EdgeStation-bootstrap\partial-distro-backup` intact unless its contents are deliberately reviewed.

Toolkit2 is licensed Rockchip SDK software. Keep its archive, wheel, and dependencies outside Git and use them only under the authorized compatible-product development terms. Current SDK and wheel artifacts are in the ignored local .work area; do not commit or redistribute them.

## Native RK3576 build

SSH to the board, enter /home/kickpi/projects/RK-EdgeStation, and use the commands in [build and deploy](BUILD_AND_DEPLOY.md). The board-side build and inference path uses the installed runtime; no host-side RKNN Runtime installation is part of it.
