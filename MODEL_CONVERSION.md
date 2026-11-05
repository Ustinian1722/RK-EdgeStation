# Model conversion and validation

Status: the official RK3576 MobileNet P0 sample and the project YOLOv8n P1 smoke run passed through the board's RKNN C Runtime. Conversion and run artifacts are local validation evidence; model weights remain outside Git.

## Version selection

- Board runtime: `librknnrt.so` 2.0.0b0, build dated 2024-03-24.
- Official Model Zoo v2.0.0 explicitly added RK3576 and requires RKNPU2 SDK >=2.0.0.
- Official current Model Zoo v2.5.0 requires SDK >=2.5.0; it must not be paired with this board runtime without a separately authorized and verified board update.
- Candidate first model: official RK3576 YOLOv8n from Model Zoo v2.0.0. YOLO11n was added later and is not the conservative first model for this runtime.
- The Toolkit2 v2.0.0-beta0 source tree includes a preconverted RK3576 MobileNet `.rknn`. It was compiled against the installed runtime and ran on the board; details and hashes are in [P0_ACCEPTANCE_REPORT.md](P0_ACCEPTANCE_REPORT.md) and `evidence/p0/official-rknn-demo-meta.txt`.

## Conversion environment constraint

An isolated WSL2 distro is installed at `D:\WSL\RK-EdgeStation-Toolkit2-Fresh`: Ubuntu 22.04 x86-64, Python 3.10.12, and Toolkit2 2.0.0b0 (`2.0.0b0+9bab5682`). The project's dependency imports and `pip check` pass. Bootstrap and repair instructions are in `D:\WSL\RK-EdgeStation-bootstrap\README.md`.

The Windows/WSL host is used only for optional ONNX-to-RKNN conversion and diagnostics. It needs neither Rockchip silicon nor a board runtime. The application, native build, media input, RKNN C Runtime inference, event processing, and local UI run on RK3576. The current Toolkit2 wheel is x86-64; the board's aarch64 image has no Toolkit2 Python package, so conversion for this pinned version is done on the host or another compatible x86-64 Linux system.

The RKNN Toolkit license is a Rockchip SDK agreement: using the SDK implies acceptance and limits the grant to development/testing of compatible Rockchip applications. This is not a general open-source dependency license. The user explicitly authorized use of Toolkit2 under its terms. SDK files and wheels stay outside Git. The earlier partial WSL VHDX is preserved separately at `D:\WSL\RK-EdgeStation-bootstrap\partial-distro-backup\ubuntu-22.04-partial-ext4.vhdx`.

## Project YOLOv8n conversion and board smoke

- Model source: Model Zoo v2.0.0 (`9bed8a56de595046651d20fa6e107884f2011e65`), using the optimized graph linked by its `examples/yolov8/model/download_model.sh`: `https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8/yolov8n.onnx`. This is the Model Zoo optimized export, not the original Ultralytics graph. The Model Zoo attributes the pretrained weights to `airockchip/ultralytics_yolov8`; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The ONNX and RKNN weights are retained only in ignored `.work` directories.
- ONNX SHA-256: `0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5`; input `[1,3,640,640]`; opset 12; nine outputs arranged as three box/class/score-sum groups at 80×80, 40×40, and 20×20. The ONNX CPU probe is in [the ONNX probe](evidence/p1/yolov8n-rk3576-fp-onnx-cpu-probe.txt). A later same-input numerical and detection-set comparison against RKNN is recorded below; it covers two frames only and does not establish model equivalence or accuracy.
- Conversion: Toolkit2 `2.0.0b0+9bab5682`, Ubuntu 22.04 x86-64 WSL2, Python 3.10.12, PyTorch `2.0.1+cpu`; `target_platform=rk3576`, mean `[0,0,0]`, std `[255,255,255]`, `fp` / `do_quantization=False`. No calibration dataset was used. The converter warned that opset 19 is recommended while this source graph is opset 12; conversion nevertheless completed and the resulting model ran on the board.
- Command: `rknn-python examples/yolov8/python/convert.py <ONNX> rk3576 fp <OUTPUT.rknn>` from the v2.0.0 Model Zoo checkout. The captured conversion output is [yolov8n-rk3576-fp-conversion.log](evidence/p1/yolov8n-rk3576-fp-conversion.log).
- RKNN artifact: 8,317,332 bytes; SHA-256 `2162e0cf6a71270c092353d94abb391ac111910ce178547ef80dd89b225dab0b`; kept at `.work/conversion/yolov8n-rk3576-fp/yolov8n-rk3576-fp.rknn` and not committed.
- Board result: KICKPI K7 / RK3576, Runtime 2.0.0b0, driver 0.9.7 from the board runtime probe. The worker loaded one 640×640, 3-channel input and nine outputs (`quantized=false`). On a generated 16-frame, 4-FPS MP4 made by repeating the Model Zoo `bus.jpg`, all 16 frames completed; five entry events (one bus and four people) were saved and returned from `/api/events`. The live API reported 3.986 processed FPS, paced by the 4-FPS file, last `rknn_run` plus output-fetch interval 94.654 ms, and end-to-end p50/p95 149.770/189.583 ms. This is a smoke observation from 16 repeated still frames, not an accuracy or peak-throughput result; see [P1_ACCEPTANCE_REPORT.md](P1_ACCEPTANCE_REPORT.md).
- A 150-frame, 15-FPS repeated-still queue probe completed 88 frames and dropped 62 at capacity 1. A second run completed 91 and dropped 59; the shared `/proc/interrupts` lines labeled for `27702000.iommu` and `27700000.npu` increased by 5,096 during inference and by 0 during a two-second idle comparison. This supports RKNPU activity during the RKNN worker run. The devfreq stayed at 300 MHz and its `load` string remained `100@300000000Hz` after idle, so no utilization percentage or frequency-scaling result is claimed. `/dev/rknpu*` is absent and the runtime-to-device path is not traced.
- A moving-video run on the first 150 frames of `vtest.avi` completed with 130/150 processed, 20 dropped, 1,033 detections, and 42 persisted events. After a tracker association refinement, a capacity-1 API smoke completed with 129/150 processed and 29 events. In a separate controlled comparison, all 150 frames were processed in three repetitions per tracker: the greedy-IoU baseline emitted 33 enter events per run and the motion-predicted candidate emitted 29, with 1,194 detections in every run. This suggests fewer repeated entries, but without identity labels it is not an accuracy result. Two low-confidence `skis` predictions remain. The 58-sample IRQ series rose during inference, but a later same-boot read was lower than its captured in-run value; it is retained as exploratory data only.

Raw board build, run, API, stress, frequency, and dynamic-video evidence is under `evidence/p1/`. The same-input ONNX CPU vs board RKNN comparison is exploratory and limited to two single frames; it does not validate detection or tracking accuracy against labels.

## Exploratory same-input numerical comparison

The comparison used the saved post-letterbox RGB uint8 NHWC tensor captured from the RK3576 worker. ONNX Runtime 1.16.0 consumed that same image as float32 RGB `/255`, transposed to NCHW. RKNN output attributes reported FP16, while `rknn_outputs_get(want_float=1)` returned float32 buffers for analysis. Both outputs were decoded with confidence 0.25 and NMS 0.45.

- On the Model Zoo bus still, 5/5 thresholded detections matched at IoU 0.50; mean matched IoU was 0.999324 and mean absolute confidence difference was 0.002519.
- On OpenCV `vtest` frame 100, 8/8 detections matched; mean matched IoU was 1.000000 and mean absolute confidence difference was 0.002329.
- The three box-regression outputs had cosine similarity 0.9999930-0.9999964 on these samples. Raw class-score outputs differed more in low-score regions: RKNN outputs had a minimum of 0.001831 where ONNX outputs included exact zeros. The score-sum (`ReduceSum`) outputs had mean absolute differences of 0.1455-0.1462. The observed class-score floor contributes to this difference, but its exact cause is not isolated.
- Independent Python postprocessing matched the C++ worker detections on both frames, with all confidences within 0.00000004; vtest boxes differed by about one pixel due to floating-point DFL/postprocessing precision.

This is a two-frame numerical and thresholded detection-set observation, not labeled accuracy, general equivalence, or a confidence-boundary test. The low-score class and score-sum differences need investigation across a larger representative set. Reports are [bus frame](evidence/p1/yolov8n-rk3576-fp-onnx-rknn-bus.json), [vtest frame 100](evidence/p1/yolov8n-rk3576-fp-onnx-rknn-vtest100.json), and [comparison summary](evidence/p1/yolov8n-rk3576-fp-onnx-rknn-summary.txt).

### Reproducing the diagnostic

On RK3576, add `--dump-first-rknn-output DIR` to a normal worker invocation. This optional flag writes the first post-letterbox RGB input, the first nine float32 RKNN output buffers, tensor metadata, and C++ detections. It is disabled by default and currently supports non-quantized `fp` models only. Copy the resulting `DIR` to a host path accessible from WSL, then run the comparison script in the Toolkit2 environment:

```bash
rknn-python /mnt/d/RK-EdgeStation/scripts/compare_onnx_rknn_outputs.py \
  --onnx-model /mnt/d/RK-EdgeStation/.work/models/yolov8n-model-zoo-v2.0.0.onnx \
  --dump-dir /mnt/d/RK-EdgeStation/.work/onnx-rknn-comparison/raw \
  --output /mnt/d/RK-EdgeStation/.work/onnx-rknn-comparison/comparison.json
```

Raw input/output tensors and model weights are diagnostics only and stay in ignored `.work` directories; they are not committed.

## Conversion record to retain outside Git

For every model, retain outside Git:

- upstream model URL, license, source revision and source SHA-256;
- export framework/tool versions and ONNX SHA-256;
- Toolkit2 version, Python version, OS and conversion script/config;
- target platform (`rk3576`), input dimensions/layout/color order, mean/std, optimization level, quantization mode, calibration sample manifest, and conversion log;
- resulting `.rknn` SHA-256 and embedded model/toolkit/target metadata;
- numerical comparison against the source graph and board runtime output.

FP16 and INT8 comparisons must use the same model graph and input samples. Report output matching and detection-level effects separately; successful conversion alone does not establish model accuracy.
