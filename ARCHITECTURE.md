# Architecture

> This document describes the historical copy-path baseline. For the current candidate, including optional DMA-BUF and multistream, see [current architecture](docs/architecture.md). The original text below is retained as experimental provenance.

Status: the selectable GStreamer/MPP decoder and RGA preprocessing path are integrated into the RK3576 C++ worker and have passed three-repeat board matrices. OpenCV/FFmpeg plus CPU preprocessing remains the reference fallback. The implementation copies decoded frames between stages; DMA-BUF import and zero-copy are not implemented. Exact end-to-end tradeoffs and validation limits are in [FLAGSHIP_OPTIMIZATION_REPORT.md](FLAGSHIP_OPTIMIZATION_REPORT.md).

## End-to-end path

```text
MP4 / H.264 RTSP
       |
  VideoSource
  |          |
OpenCV      GStreamer C API
FFmpeg      qtdemux / rtspsrc + depay
  |         mppvideodec -> NV12 appsink
  |                    |
  +------ BGR / owned packed NV12 copy
                       |
               bounded frame queue
                 latest or FIFO
                       |
        CPU/OpenCV or synchronous RGA
        resize + color + 114 letterbox
          |                         |
       BGR/RGB uint8 NHWC model input
                       |
                 RKNN Runtime
                       |
            output fetch + postprocess
                       |
        tracking + PPE/ROI event state
                       |
       evidence + SQLite + REST/local UI
```

The tested GStreamer build is 1.24.2 with Rockchip MPP plugin 1.14.4 and MPP 1.5 on the KICKPI K7 / RK3576. The board does not provide `h264parse`, so the MP4 path uses `filesrc ! qtdemux ! capsfilter(video/x-h264, parsed=true, stream-format=avc, alignment=au) ! mppvideodec`; H.264 RTSP uses `rtspsrc ! rtph264depay ! capsfilter ! mppvideodec`. The decoded output is negotiated as NV12 and read through `appsink`.

## Source and buffer ownership

- `src/video_source.hpp/.cpp` defines `VideoSource`, `DecodedFrame`, the OpenCV/FFmpeg backend, and the GStreamer/MPP backend. `--video-backend opencv|gstreamer-mpp` selects the source. The OpenCV path owns a copied BGR frame. The GStreamer path maps the sample using `GstVideoFrame`, respects each plane's stride, row-copies visible Y and UV pixels into an owned contiguous NV12 `cv::Mat`, copies the PTS, then unmaps and unreferences the sample.
- `appsink` is bounded to one sample and does not silently drop frames. The application queue applies the selected bounded FIFO or latest-frame policy and reports queue drops. This lets the offline preserve-all runs account for each decoded frame.
- MPP-to-worker therefore includes a full NV12 host-memory copy. In this build, padding/stride bytes are discarded during row copy. The GStreamer buffer is released before inference starts.
- No DMA-BUF fd is passed from MPP to RGA or RKNN. The RGA API receives host virtual-address buffers, writes a host `cv::Mat`, and RKNN receives that CPU pointer through `rknn_inputs_set`. Do not describe this as zero-copy.
- Local-file PTS is mapped through the GStreamer segment. RTSP uses monotonic receive/decode time as its event timeline. A successful RTSP reconnect increments `stream_epoch`; queued frames from an older epoch are discarded and tracker/event temporal state resets.

## Preprocessing contracts

`src/preprocess.cpp` provides `--preprocess-backend cpu|rga`.

- CPU reference: BGR source or NV12-to-BGR conversion, OpenCV linear resize, pad value 114, then model-specific channel order.
- RGA candidate: synchronous `improcess` combines resize, color conversion and letterbox into a CPU-addressed destination. NV12 is the direct source when the decoder backend supplies it. OpenCV BGR input remains accepted for backend-isolation tests.
- COCO YOLOv8n receives RGB, NHWC `uint8`, raw 0–255. PPE YOLOX-S receives BGR, NHWC `uint8`, raw 0–255. The PPE contract is unchanged by the RGA path.
- `scripts/compare_preprocess_dumps.py` compares the source sequence/PTS, tensor bytes, float outputs when representations match, and same-class detection boxes for two saved first-frame runs.

Both paths currently allocate/copy into the RKNN input tensor. RGA runs synchronously (`IM_SYNC`); no overlap between RGA and inference is claimed.

## Worker scheduling and event behavior

The C++ worker has one source/producer thread and one inference/consumer thread connected by a finite queue. Queue capacity and `latest|fifo` policy are configurable. Latest replaces the oldest queued frame; FIFO preserves queued order and rejects a newly decoded frame while full. Metrics include decoded/processed/dropped counts, stage latency percentiles, queue wait, decode-to-result age, process CPU, RSS, copied source bytes, model-input bytes, and nominal media rate. Offline unpaced and media-clock paced runs are distinct modes (`--input-pacing none|media`).

The event layer retains PPE/person association, ROI entry/leave/dwell, persistent danger-zone intrusion, and sustained image-region forklift/person co-occurrence, plus de-duplication, cooldown, severity, evidence images, SQLite history and local REST/UI. The co-occurrence event is an image-space cue; monocular video does not provide a physical-distance estimate.

## Build gates and known limits

- CMake options `EDGE_ENABLE_GSTREAMER` and `EDGE_ENABLE_RGA` allow the portable OpenCV/CPU build to remain available. The RK3576 build used both options enabled with the board's installed libraries; no kernel, driver, package or runtime replacement was made.
- `/dev/mpp_service` and `/dev/rga` are present. The board's RKNN C Runtime successfully runs the worker even though `/dev/rknpu*` device nodes are absent. Per-core NPU utilization is not exposed by the current evidence.
- MPP-to-NV12 stage throughput is high, but MPP-to-BGR was slower than OpenCV/FFmpeg on the tested clip. Worker-level results are reported separately; the NV12 fakesink benchmark is not a system FPS claim.
- RGA reduced COCO preprocessing time in the matched worker run, but the end-to-end FP FPS change was small because RKNN input transfer and inference dominate. For the already model-sized PPE input, RGA did not improve FP throughput. Use the measured workload-specific end-to-end result, not the isolated preprocessing speedup, to choose a deployment path.
- The current PPE clips repeat fixed development images. They verify model-to-event wiring only; independent dynamic PPE accuracy, identity continuity, real cameras, and physical risk-distance evaluation remain open.
