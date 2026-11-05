# Third-party notices

## RKNN Model Zoo YOLOv8 postprocessing

- Source: [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo), tag `v2.0.0`, commit `9bed8a56de595046651d20fa6e107884f2011e65`.
- License: Apache-2.0. Original copyright and license headers are retained in each copied source/header file.
- Included files: `src/third_party/rknn_model_zoo_v2.0.0/utils/common.h`, `utils/image_utils.h`, and `yolov8/postprocess.cc`, `postprocess.h`, `yolov8.h`.
- Data file: `model/coco_80_labels_list.txt` from the same revision. No model weights, sample image, runtime, or Toolkit2 binary is included.
- Local integration change: diagnostic/error `printf` calls in `postprocess.cc` now write to stderr so the worker's stdout remains a machine-readable JSONL stream. Detection and postprocessing logic is unchanged.

The project uses the upstream postprocessing routine for compatibility with the official YOLOv8 RKNN graph. The capture, tracker, event, evidence, API, and storage code is project code.

## PPE fine-tuning dependency

- Candidate source: [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), Apache-2.0. Local experiment source is cloned under ignored `.work/yolox-upstream`; no YOLOX implementation files are copied into this repository. The project-specific experiment configuration and ONNX exporter are original project files.
- Candidate data: [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe), CC BY 4.0. It is attributed in the COCO exporter and experiment instructions. Do not commit the dataset archive or derived weights.
- The YOLOX Model Zoo raw-head path was converted with Toolkit2 2.0.0b0 for `rk3576` and executed through RKNN Runtime 2.0.0b0 on the KICKPI K7. The custom training code, exporter, and C++ postprocessor are project-authored; no YOLOX implementation files or Model Zoo binaries are copied into the repository. The candidate weights remain local and are not redistributed.

## YOLOv8 pretrained model

The official Model Zoo example identifies `airockchip/ultralytics_yolov8` as the model source. That repository presents AGPL-3.0 and Enterprise licensing options. The pretrained model is used only as a local P1 validation artifact and is kept outside Git. Before redistribution or a commercial product release, select a model license compatible with that use or replace the model and revalidate the pipeline. See the project [open-source review](OPEN_SOURCE_REFERENCES.md).
