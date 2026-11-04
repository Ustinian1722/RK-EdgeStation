# Model conversion and validation

Status: the official SDK-bundled RK3576 MobileNet sample was run successfully on the board as a P0 smoke test. No project-specific model has been converted or run.

## Version selection

- Board runtime: `librknnrt.so` 2.0.0b0, build dated 2024-03-24.
- Official Model Zoo v2.0.0 explicitly added RK3576 and requires RKNPU2 SDK >=2.0.0.
- Official current Model Zoo v2.5.0 requires SDK >=2.5.0; it must not be paired with this board runtime without a separately authorized and verified board update.
- Candidate first model: official RK3576 YOLOv8n from Model Zoo v2.0.0. YOLO11n was added later and is not the conservative first model for this runtime.
- The Toolkit2 v2.0.0-beta0 source tree includes a preconverted RK3576 MobileNet `.rknn`. It was compiled against the installed runtime and ran on the board; details and hashes are in [P0_ACCEPTANCE_REPORT.md](P0_ACCEPTANCE_REPORT.md) and `evidence/p0/official-rknn-demo-meta.txt`.

## Conversion environment constraint

The pinned Toolkit2 v2.0.0-beta0 converter provides Linux x86-64 wheels, including Python 3.10. The Windows host has Python 3.12 and 3.13. A local Ubuntu 22.04 WSL registration was left partial by a failed setup; `wsl --list`, `wsl --status`, and an unregister attempt all stalled, so no usable Linux x86-64 environment is currently available. The board is aarch64 and has no Toolkit2 Python package. For this pinned converter, a compatible isolated Linux x86-64 environment is still required; the host needs no Rockchip silicon. Deployment, native C++ build, RKNN inference, media processing, and the app can all run on the RK3576 board when a compatible `.rknn` model is supplied. A preconverted model also avoids installing RK tools on the host.

The RKNN Toolkit license is a Rockchip SDK agreement: using the SDK implies acceptance and limits the grant to development/testing of compatible Rockchip applications. This is not a general open-source dependency license. The user explicitly authorized use of Toolkit2 under its terms. No SDK wheel has been installed or run by this project yet; the P0 demo used the already-converted model bundled with the official SDK source. A 12 MiB backup of the partial WSL virtual disk is retained outside the repository while the conversion environment remains unavailable.

## Planned conversion record

For every model, retain outside Git:

- upstream model URL, license, source revision and source SHA-256;
- export framework/tool versions and ONNX SHA-256;
- Toolkit2 version, Python version, OS and conversion script/config;
- target platform (`rk3576`), input dimensions/layout/color order, mean/std, optimization level, quantization mode, calibration sample manifest, and conversion log;
- resulting `.rknn` SHA-256 and embedded model/toolkit/target metadata;
- numerical comparison against the source graph and board runtime output.

FP16 and INT8 comparisons must use the same model graph and input samples. Report output matching and detection-level effects separately; successful conversion alone does not establish model accuracy.
