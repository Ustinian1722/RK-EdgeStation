# Third-party notices

## RKNN Model Zoo YOLOv8 postprocessing

- Source: [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo), tag `v2.0.0`, commit `9bed8a56de595046651d20fa6e107884f2011e65`.
- License: Apache-2.0. Original copyright and license headers are retained in each copied source/header file.
- Included files: `src/third_party/rknn_model_zoo_v2.0.0/utils/common.h`, `utils/image_utils.h`, and `yolov8/postprocess.cc`, `postprocess.h`, `yolov8.h`.
- Data file: `model/coco_80_labels_list.txt` from the same revision. No model weights, sample image, runtime, or Toolkit2 binary is included.
- Local integration change: diagnostic/error `printf` calls in `postprocess.cc` now write to stderr so the worker's stdout remains a machine-readable JSONL stream. Detection and postprocessing logic is unchanged.

The project uses the upstream postprocessing routine for compatibility with the official YOLOv8 RKNN graph. The capture, tracker, event, evidence, API, and storage code is project code.

## YOLOv8 pretrained model

The official Model Zoo example identifies `airockchip/ultralytics_yolov8` as the model source. That repository presents AGPL-3.0 and Enterprise licensing options. The pretrained model is used only as a local P1 validation artifact and is kept outside Git. Before redistribution or a commercial product release, select a model license compatible with that use or replace the model and revalidate the pipeline. See the project [open-source review](OPEN_SOURCE_REFERENCES.md).
