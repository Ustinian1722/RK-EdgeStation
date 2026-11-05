"""YOLOX-S experiment for the attributed Construction-PPE dataset.

The external YOLOX source is kept under .work and loaded through PYTHONPATH;
this file contains only project-specific configuration.
"""

from __future__ import annotations

import os
from pathlib import Path

from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 12
        self.seed = 42
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.multiscale_range = 0
        self.max_epoch = 40
        self.warmup_epochs = 2
        self.no_aug_epochs = 6
        self.eval_interval = 2
        # Windows spawn-based workers repeatedly serialize the large COCO
        # annotation object; zero workers keeps memory bounded on this host.
        self.data_num_workers = 0
        self.save_history_ckpt = False
        self.ema = False
        self.basic_lr_per_img = 0.01 / 64.0
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.enable_mixup = False
        self.hsv_prob = 0.5
        self.flip_prob = 0.5
        self.degrees = 5.0
        self.translate = 0.08
        self.mosaic_scale = (0.5, 1.5)
        self.shear = 1.0
        self.exp_name = "yolox_s_construction_ppe"

        dataset = os.environ.get("RK_EDGE_PPE_DATASET")
        if not dataset:
            raise RuntimeError("set RK_EDGE_PPE_DATASET to the extracted dataset root")
        self.data_dir = str(Path(dataset).resolve())
        self.output_dir = str(Path(self.data_dir).parents[1] / "YOLOX_outputs")
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"

    def get_model(self):
        import random
        import numpy as np
        import torch

        # Preserve repeatable Python/NumPy/Torch RNG initialization while using
        # cuDNN's non-deterministic workspace path. Strict deterministic mode
        # plus benchmark=True failed to find a convolution engine on this host.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        model = super().get_model()
        # Freeze CSPDarknet but train the feature pyramid and 12-class head.
        # This keeps 640-pixel training within the available GPU memory.
        for parameter in model.backbone.backbone.parameters():
            parameter.requires_grad_(False)
        return model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="images/train",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="images/val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )
