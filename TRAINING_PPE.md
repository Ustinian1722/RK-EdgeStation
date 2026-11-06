# PPE detector experiment

## Scope and data

This is a separate 12-class PPE/person/forklift detector experiment. It does not
replace or relabel the existing COCO YOLOv8n baseline. The candidate is pinned to
the Ultralytics Assets `v0.0.0` Construction-PPE archive, distributed under CC BY
4.0. The archive hash, embedded license, official November 2026 dataset page,
and matching YAML are recorded in the [dataset audit](evidence/ppe/construction-ppe-audit.json).
Preserve the attribution in `scripts/export_construction_ppe_coco.py` and in any
dataset-derived model card. Its 1,000 images are split 800/200 with 14,863 annotations.
The release says human-collected and generated images are mixed, but the public
manifest has no image-level origin field, so the generated-image fraction is
unknown. The release has no separate test split; its validation metrics are
development metrics and are not independent deployment accuracy.

The hash and label audit is reproducible with:

```powershell
python scripts/audit_construction_ppe.py .work/construction-ppe/extracted/construction-ppe `
  --archive .work/construction-ppe/construction-ppe.zip `
  --output evidence/ppe/construction-ppe-audit.json
```

Create COCO annotation files next to the dataset's images:

```powershell
python scripts/export_construction_ppe_coco.py `
  .work/construction-ppe/extracted/construction-ppe `
  .work/construction-ppe/extracted/construction-ppe/annotations
```

The archive, images, annotations, checkpoints, and ONNX/RKNN binaries stay under
`.work` and are excluded from Git.

## Reproducible event-replay inputs

After exporting the COCO annotations, regenerate positive and compliant-negative
fixed-scene clips from validation image IDs 90 and 200:

```powershell
python scripts/generate_ppe_replay_videos.py `
  --output-dir .work/ppe-event-replays
```

The generator verifies the person/PPE labels before encoding two 640×640 H.264
clips at 15 FPS for 150 frames. Its `replay-manifest.json` records source-image
and video hashes, expected PPE-event labels, FFmpeg version, and encoding
arguments. This creates repeatable rule inputs, not real moving footage. Board
results must be collected again for newly generated video hashes; the archived
metrics in the acceptance report refer to the separately hashed original clips.

Create 640-pixel training images and matching COCO boxes in the ignored cache:

```powershell
python scripts/prepare_ppe_training_cache.py `
  .work/construction-ppe/extracted/construction-ppe `
  .work/construction-ppe/training-640 --max-side 640
```

## Training

The base detector is YOLOX-S from the Apache-2.0 Megvii YOLOX project, rather
than Ultralytics' AGPL training stack. The experiment used upstream commit
`6ddff4824372906469a7fae2dc3206c7aa4bbaee`. Keep the local clone pinned to that
revision. Install its Python package in
an isolated environment with the existing CUDA-enabled PyTorch and OpenCV, plus
`pycocotools`; do not install its old pinned Torch requirements over the local
CUDA stack.

From PowerShell, with that environment active (the project loop avoids the
legacy trainer's backward-pass memory failure on this host):

```powershell
$env:RK_EDGE_PPE_DATASET = (Resolve-Path `.work/construction-ppe/training-640`).Path
$env:PYTHONPATH = (Resolve-Path `.work/yolox-upstream`).Path + ";" + $env:PYTHONPATH
$env:OMP_NUM_THREADS = "1"
python scripts/train_yolox_ppe.py `
  --exp training/exp/yolox_s_construction_ppe.py `
  --pretrained `.work/construction-ppe/yolox_s.pth` --batch-size 1
```

If a run is interrupted after saving `latest_ckpt.pth`, add `--resume` to
continue from the next completed epoch. Validation uses the installed standard
`pycocotools` evaluator on Windows and records aggregate plus per-class AP50 and
AP50:95 in `epoch_metrics.jsonl`.

The planned 40-epoch/640-pixel/batch-1/seed-42 configuration is a constrained
first fine-tune for the 8 GiB RTX 5060 and this small dataset, not a claimed
optimum.
The COCO-pretrained CSPDarknet feature extractor is frozen; the feature-pyramid
neck and 12-class detection head are adapted to PPE. Full-model backpropagation
exceeded available GPU memory on this host, and FP16 also failed during
backpropagation. The project loop uses FP32, disables the CUDA foreach SGD
temporary that exceeded available memory, and omits the extra EMA model to
bound memory. Mosaic augmentation is disabled because its Windows preprocessing
path hit host allocation failures; single-image HSV and flip augmentation
remain. No full-network fine-tuning is claimed.
The Windows data loader uses zero worker processes because spawn-based workers
failed while serializing the COCO annotation object. Python, NumPy, and Torch
RNGs are seeded, while strict cuDNN determinism is disabled because its
benchmark/deterministic combination failed to select a convolution engine on
this host; the run is therefore reproducible at the RNG/initialization level,
not guaranteed bitwise identical. Keep the
pretrained model, seed, YOLOX commit, exact command, hardware/software versions,
best checkpoint hash, and per-class AP50/AP50:95 in the resulting experiment
record. Do not select a deployment threshold on the same validation results and
then describe them as untouched test accuracy.

The completed run logged 27 full validation epochs (indices 0–26) before the
host allocation/resume path stopped short of the 40-epoch budget. Best was
epoch index 26: AP50:95 47.223% and AP50 72.505%. Per-class scores and all
logged epochs are in [PPE_ACCEPTANCE_REPORT.md](PPE_ACCEPTANCE_REPORT.md) and
[`evidence/ppe/training-epoch-metrics.jsonl`](evidence/ppe/training-epoch-metrics.jsonl).
The best checkpoint SHA-256 is
`e38fb877f9959fbc104407be31000135e7164b23aab7b01229b7017538e4d195`.

## ONNX and RKNN conversion

Export the fine-tuned checkpoint to the three pre-decode YOLOX head maps used by
the RKNN Model Zoo example:

```powershell
python scripts/export_yolox_onnx.py `
  --exp training/exp/yolox_s_construction_ppe.py `
  --checkpoint `.work/YOLOX_outputs/yolox_s_construction_ppe/best_ckpt.pth` `
  --output `.work/construction-ppe/yolox_s_construction_ppe.onnx`
```

The exporter checks that outputs are `[1,17,80,80]`, `[1,17,40,40]`, and
`[1,17,20,20]`, where 17 is 4 box channels + 1 objectness channel + 12 class
channels. The graph was converted with the authorized, isolated Toolkit2
2.0.0b0 environment for `rk3576`. YOLOX training/validation uses BGR raw
0–255. From the repository root inside `RK-EdgeStation-Toolkit2`, convert
with the matching configuration:

```bash
rknn-python scripts/convert_onnx_to_rknn.py \
  --onnx-model .work/construction-ppe/yolox_s_construction_ppe.onnx \
  --output .work/construction-ppe/yolox_s_construction_ppe_std1_fp.rknn \
  --target-platform rk3576 --dtype fp \
  --input-mean 0,0,0 --input-std 1,1,1
```

The resulting 20.7 MB model ran through the RKNN C Runtime on the KICKPI K7.
On a saved board input, ONNX/RKNN detection sets matched at IoU 0.5 across
confidence thresholds 0.10, 0.25, and 0.50. This checks graph/runtime
consistency for one image; it does not prove dataset accuracy. The fixed-input
results and hashes are in [PPE_ACCEPTANCE_REPORT.md](PPE_ACCEPTANCE_REPORT.md).

## License and provenance

- RK-EdgeStation source remains Apache-2.0.
- Dataset: the pinned November 2026 `ultralytics/assets` v0.0.0 archive is CC BY
  4.0; retain attribution and license link. The current Chinese dataset page
  and the archive README/LICENSE match its 1,000-image, 12-class schema. The
  English dataset page currently describes a different 2025 schema (1,416
  images, 11 classes, AGPL-3.0); do not use that page to identify the archive
  used for this experiment.
- YOLOX source: Apache-2.0; it is an external training dependency, not copied
  into this repository.
- Do not train this route with Ultralytics model/training code: its AGPL terms
  are incompatible with silently keeping this project under Apache-2.0.
- RKNN Toolkit2 is a licensed Rockchip SDK, not an open-source project
  dependency. Keep its binaries and docs outside Git.
