# Architecture

Status: P0 design freeze based on the hardware audit. The design deliberately separates portable CPU behavior from Rockchip-specific acceleration so that each hardware stage can be measured and disabled independently.

## Deployment boundary

```text
MP4 / RTSP / verified camera
            |
      VideoSource
            |
      bounded frame queue
            |
   Decode + preprocess ---- CPU/OpenCV reference
      |             |
   MPP decode     RGA transform (only when verified)
      |             |
      +------ RKNN input adapter
                  |
             RKNN Runtime
                  |
         detection postprocess
                  |
          tracker + ROI rules
                  |
        event/evidence service
           /             \
     SQLite/logs      local REST/UI
                  |
       metrics + recovery state
```

Every block is in one RK3576 Linux deployment. No Jetson, remote inference service, or second compute node participates in the target data path.

## Process and module boundaries

- `core`: timestamps, frame/image ownership, result/event types, configuration validation, bounded queue, cancellation, and metrics.
- `video`: source interface, OpenCV CPU fallback, GStreamer/MPP candidate, V4L2 camera adapter enabled only after sensor verification.
- `preprocess`: deterministic CPU/OpenCV reference and optional RGA backend. Input/output pixel formats, strides, alignment, and copy count are explicit.
- `inference`: `IInferenceEngine` interface; RKNN implementation owns context and output buffers with RAII. Mock implementation is for host CI only.
- `vision`: detector decoding, stable-ID tracker, ROI geometry, event state machine, evidence extraction.
- `runtime`: bounded workers, FIFO/latest-frame policies, deadlines, backpressure, reconnect and cancellation.
- `management`: loopback REST API, browser assets, model/source/ROI CRUD, authentication, and health routes.
- `storage`: SQLite schema/migrations and structured event/evidence metadata.
- `optlab`: model/conversion metadata, accuracy/output comparison, stage timers, resource sampling, report generation.

## Frame and result contracts

Each frame carries source ID, sequence, decode/presentation timestamp, monotonic ingest time, dimensions, pixel format, stride, and an owned buffer handle. A buffer handle specifies whether it is CPU memory, an MPP frame, an RGA allocation, or a DMA-BUF fd; ownership and release are explicit.

Inference results retain source/frame sequence and monotonic start/end times. Tracking updates and events are accepted only for the same current stream epoch and valid frame sequence. Reconnect increments the stream epoch to invalidate delayed work from the previous connection.

## Scheduling behavior

- Queue capacity is configured and finite.
- FIFO preserves every accepted frame until full; latest-frame mode evicts an older queued frame when a newer frame arrives.
- Every evicted, expired, timed-out, or failed frame increments a distinct counter.
- A stream epoch, cancellation token, and bounded worker shutdown prevent stale results after reconnect/restart.
- Queue wait, inference duration, end-to-end latency, and frame age use monotonic clocks and are reported separately.

## Media and memory policy

1. Establish pixel-for-pixel/box-coordinate behavior with CPU decode and preprocessing.
2. Add MPP decode and RGA transform behind the same interfaces.
3. Benchmark each stage under the same video, format, dimensions, and thread settings.
4. Add DMA-BUF import only if allocator compatibility, cache synchronization, fd lifetime, stride, and runtime input binding are demonstrated on RK3576.

The audit found `/dev/mpp_service`, `/dev/rga`, and a system DMA heap, but not `/dev/rknpu`. The official RKNN sample passed; MPP/RGA and the full video-to-event chain still require their own board tests.

## Management and safety defaults

- API binds to `127.0.0.1` by default; explicit configuration is required for LAN binding.
- API routes validate source URLs, model paths, ROI geometry, and resource limits.
- Event records retain model/rule version and evidence time so a reviewer can reconstruct why the event occurred.
- Service shutdown stops sources, cancels workers, releases RKNN outputs/contexts, flushes event state, and closes SQLite.
- The service never changes clocks/governors or installs/replaces drivers automatically.
