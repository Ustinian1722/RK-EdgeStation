# DMA-BUF / shared RKNN input implementation

## Status

The production Worker now has a selectable Z2 path for the installed RK3576 stack:

```text
GStreamer mppvideodec GstSample (MPP GstMemory FD, NV12)
  → RGA import of borrowed MPP FD
  → NV12-to-model-color, resize and letterbox in RGA
  → RGA import of RKNN-owned input-memory FD
  → rknn_mem_sync(BIDIRECTIONAL)
  → rknn_run / outputs_get
```

The path is **partial low-copy**. It removes the visible NV12 CPU row-copy from inference input and directs RGA output into the RKNN input allocation. It does not remove the CPU postprocess/output boundary, event-image conversion, or any internal driver/runtime copy that cannot be inspected here. The default remains Z0 (`copy` + `api`).

## Selection and fallback

```bash
# Stable/default: MPP sample → packed NV12 CPU copy → RGA → rknn_inputs_set
rk_edge_worker --model yolox-s-ppe-int8.rknn --model-family yolox-ppe \
  --video sample.mp4 --video-backend gstreamer-mpp --preprocess-backend rga \
  --video-memory-mode copy --rknn-input-mode api

# Candidate: retained MPP DMA-BUF → RGA → RKNN input-memory FD
rk_edge_worker --model yolox-s-ppe-int8.rknn --model-family yolox-ppe \
  --video sample.mp4 --video-backend gstreamer-mpp --preprocess-backend rga \
  --video-memory-mode dmabuf --rknn-input-mode shared
```

`dmabuf` requires GStreamer/MPP + RGA + shared RKNN input. `shared` requires RGA. The Worker rejects unsupported options and fails the frame on invalid metadata or API status; it never silently switches buffers or formats. Switch to `copy` + `api` explicitly to use the proven fallback.

## Ownership and synchronization

1. **MPP frame acquisition.** `GstMppVideoSource::read` requests `video/x-raw,format=NV12`. It checks for exactly one `GstMemory`, a `gst_is_dmabuf_memory` allocator, a valid FD, `GstVideoMeta`, contiguous plane offsets, two equal strides, an offset-zero memory view and a buffer size. It stores the borrowed FD and a `shared_ptr<void>` whose deleter calls `gst_sample_unref`. The FD is never closed by `DecodedFrame`.
2. **Queue/lifetime.** `DecodedFrame` moves through the bounded queue. Latest-frame eviction destroys the evicted sample reference. The active inference frame retains its sample through the synchronous RGA `IM_SYNC` operation. A 10-cycle board probe verified the FD remains valid after source pipeline close while a frame owns the sample, and closes when that frame releases its sample. MPP/system-heap process-global FDs rose by five on first initialisation and stayed flat across the remaining nine cycles.
3. **RKNN allocation.** `RknnYoloEngine` queries `RKNN_QUERY_NATIVE_INPUT_ATTR`, requires one NHWC image with matching width/height/three channels, sets the established raw image input type to UINT8, allocates `size_with_stride` using `rknn_create_mem`, checks its virtual address/FD/size and binds it with `rknn_set_io_mem`. The memory is destroyed before `rknn_destroy`.
4. **Z1 virtual output.** RGA writes through the RKNN allocation's virtual-address view. The memory is synchronised with `RKNN_MEMORY_SYNC_TO_DEVICE` before inference, because that route writes through a CPU-addressable mapping.
5. **Z2 FD output.** RGA imports both MPP source and RKNN destination FDs, wraps both as handles, fills the letterbox region and performs a synchronous resize/color conversion into the destination. Both handles are released and checked before `rknn_mem_sync(RKNN_MEMORY_SYNC_BIDIRECTIONAL)` and inference. The installed Runtime header defines bidirectional sync as both CPU/device consistency directions. The NPU runs serially in the existing Worker, so the single RKNN input buffer is not reused by concurrent in-flight frames.
6. **Event evidence.** When the Worker emits an event, `frame_to_bgr` maps the retained GstSample, copies the visible Y/UV rows into a temporary NV12 matrix, converts to BGR, annotates and JPEG-encodes. This intentional CPU copy is outside inference input and is not included in `source_copy_bytes_total`.
7. **Outputs.** Existing `rknn_outputs_get`/release and CPU postprocessing remain unchanged. No output DMA-BUF path was attempted.

## Model input contracts

| Model | RGA output | Shared memory allocation | Preserved behavior |
|---|---|---:|---|
| YOLOX-S PPE INT8 | BGR NHWC UINT8, raw 0–255 | 1,228,800 B | Matches existing PPE preprocessing contract. Board probe produced byte-identical input and equal outputs/detections against Z0 on the captured frame. |
| YOLOX-S PPE FP | BGR NHWC UINT8, raw 0–255 | 2,457,600 B stride allocation from queried native attributes | Preserves the validated raw UINT8 submit contract despite the model's native FP16 input representation. The exact input/output parity probe passed. |
| YOLOv8n COCO | RGB NHWC UINT8, raw 0–255 | Queried at runtime | Uses the existing RGB contract. This task's detailed RGA→RKNN direct-FD parity and dynamic Worker runs use the PPE model; COCO Z2 is not promoted by these results. |

The RGA letterbox color remains 114 and `INTER_LINEAR` is explicit. Any unsupported channel order, dimensions, plane layout or memory size fails before inference.

## Measurements and validation

- RK3576 source compiled `rk_edge_worker`, `gst_dmabuf_probe`, `rknn_shared_input_probe` and `gst_mpp_lifecycle_probe`; board CTest passed 2/2 (`edge_core_tests`, `edge_media_tests`).
- `gst_mpp_lifecycle_probe` completed ten open/read/close cycles. Every borrowed MPP FD was valid before close, remained valid while its `DecodedFrame` retained the GstSample, and was closed after sample release. Process FD count: 3 before first MPP open, 8 after the first cycle, and 8 after each later cycle.
- PPE INT8 shared-input probe, three repeated uses of one captured frame with the final bidirectional Z2 sync: B0, Z1 and Z2 input hashes equal; all three RKNN output hashes equal; detections exact-equal; Z2 MPP and RKNN FD imports succeeded. This checks input and output parity for the fixed captured frame, not temporal accuracy.
- The C++ media test compares caller-owned CPU preprocessing output byte-for-byte with the existing reference path, checks invalid destination rejection, and confirms CPU preprocessing refuses a DMA-BUF-only frame without a mapping. Board CTest passes this alongside the existing video/plane checks; RGA/RKNN DMA-BUF behavior is covered by the separate board probes. Final build and CTest output is preserved in [`board-build-ctest.txt`](evidence/dynamic-video/board-build-ctest.txt).
- Full 34-second dynamic H.264 video completed under Z0/Z1/Z2. Z2 reported zero explicit inference-ingress source-copy bytes. Z0/Z1/Z2 each reached EOS with exit status 0.
- The worker does not count internal DMA migration, CPU cache-maintenance, MPP pool traffic, RGA/NPU internal movement, RKNN input conversion or RKNN output transfer as explicit `memcpy` bytes. The API does not expose enough instrumentation to determine those counts.

## Known limits and rejected alternatives

- The sample's ordinary NV12 caps do not contain a DRM format/modifier. The accepted buffer layout is bounded to the exact MPP/GStreamer/RGA combination tested; arbitrary tiled/compressed modifier paths are not claimed.
- Requesting the `memory:DMABuf` caps feature caused negotiation failure on this plugin. Ordinary caps still produced `GstMemory` backed by a valid MPP DMA-BUF FD. The implementation validates runtime memory, not pad-template capability.
- A small exploratory `imcopy` using an imported source handle plus a virtual-address destination failed with librga 1.10.1's “only handles or no handles” constraint. The production Z2 operation imports both ends and passed. This failed mixed-memory probe is retained in the evidence.
- An earlier isolated DMA-BUF probe showed GStreamer leaves several process-global MPP/dma-heap descriptors open after pipeline teardown. The production lifecycle probe shows these descriptors do not grow across ten open/close cycles and borrowed frame FDs are released; they remain until process exit. This is not a claim of zero process-global resource use.
- No `rknn_create_mem_from_fd` path is needed: RKNN allocates the input and exports its FD; RGA writes into that memory. The untested MPP Mode 3 external-pool rewrite and third-party GStreamer RGA plugin were not adopted.
- RTSP DMA-BUF reconnect was not exercised in this task. The tested lifecycle is repeated local MP4 source open/close on RK3576; the production Worker retains the existing RTSP reconnect path but its exact DMA-BUF reconnect behavior remains a limitation.

See [ZERO_COPY_BENCHMARK_REPORT.md](ZERO_COPY_BENCHMARK_REPORT.md) for Z0/Z1/Z2 results and [DYNAMIC_VIDEO_EVALUATION.md](DYNAMIC_VIDEO_EVALUATION.md) for the real-motion test data and event review.
