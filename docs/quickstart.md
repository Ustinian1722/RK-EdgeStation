# Quick start

## 1. Prerequisites

The verified target is KICKPI K7 / RK3576 (aarch64 Ubuntu 24.04.3, vendor Linux 6.1.75, RKNN Runtime 2.0.0b0). CMake requires OpenCV 4 development files and RKNN C headers/runtime. The optional default-enabled hardware backends also require GStreamer development libraries (including app, video and allocators), an installed Rockchip MPP GStreamer plugin and RGA headers/library. Use a board/vendor-supported runtime; do not substitute unrelated SDK versions.

Provide a compatible RKNN file and a video you have permission to use. See [model conversion](../MODEL_CONVERSION.md), [PPE training](../TRAINING_PPE.md) and [third-party notices](../THIRD_PARTY_NOTICES.md). Model family and preprocessing must match; there is no generic arbitrary-model loader.

## 2. Board build

From the cloned repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DEDGE_ENABLE_GSTREAMER=ON -DEDGE_ENABLE_RGA=ON
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
```

An OpenCV/CPU preprocessing build can disable the optional GStreamer and RGA interfaces, but still requires RKNN and OpenCV. It is not a hardware-free inference implementation.

## 3. Single-stream service

Default COCO-model reference path:

```bash
bash scripts/run_demo.sh /path/to/coco.rknn /path/to/input.mp4
```

PPE example with the hardware pipeline, using the Python service directly to select the model family:

```bash
PYTHONPATH=python python3 -m edge_station.server \
  --worker build/bin/rk_edge_worker \
  --model /path/to/yolox-ppe.rknn --model-family yolox-ppe \
  --video /path/to/input.mp4 --state-dir state/ppe-demo \
  --video-backend gstreamer-mpp --preprocess-backend rga \
  --queue-policy latest --queue-capacity 1 --port 8080
```

The server accepts RTSP in place of the local video. Do not commit camera credentials. Open http://127.0.0.1:8080 on the board; for remote access, use an SSH port forward from your workstation (replace BOARD_USER and BOARD_HOST):

```bash
ssh -L 8080:127.0.0.1:8080 BOARD_USER@BOARD_HOST
```

Then open the same address locally. Avoid directly exposing the service to untrusted networks.

## 4. Optional DMA-BUF worker

The low-copy path is a Worker-level experimental option, not an automatic fallback or service default:

```bash
build/bin/rk_edge_worker --model /path/to/yolox-ppe.rknn \
  --model-family yolox-ppe --video /path/to/input.mp4 \
  --video-backend gstreamer-mpp --preprocess-backend rga \
  --video-memory-mode dmabuf --rknn-input-mode shared
```

For incompatible DMA-BUF layout, rerun with `--video-memory-mode copy --rknn-input-mode api`. See [implementation](../ZERO_COPY_IMPLEMENTATION.md).

## 5. Multistream research mode

Copy the [example config](../deploy/cameras.example.json) to an ignored local state directory, set camera sources and task options, then run:

```bash
mkdir -p state
cp deploy/cameras.example.json state/cameras.json
# Edit state/cameras.json with authorized camera sources.
PYTHONPATH=python python3 -m edge_station.server \
  --worker build/bin/rk_edge_worker \
  --model /path/to/yolox-ppe.rknn --model-family yolox-ppe \
  --camera-config state/cameras.json --state-dir state/multi-demo \
  --video-backend gstreamer-mpp --preprocess-backend rga --port 8080
```

Use a fresh state directory for independent experiments. The previous 32-source matrix used another FP model and a shared serial context; these commands are illustrative, not a promise of 32-channel real-time inference.

## 6. Host-only checks

On Linux with Python 3 and a C++17 compiler:

```bash
python3 -m pip install -r requirements-test.txt
bash scripts/test_host.sh
```

This runs hardware-independent C++ core/multistream and Python unit tests. It does not build or validate the RKNN Worker. For board tests and service installation see the [deployment reference](../BUILD_AND_DEPLOY.md).
