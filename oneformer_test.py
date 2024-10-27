# !pip3 install git+https://github.com/cocodataset/panopticapi.git --quiet - ok
# !pip3 install git+https://github.com/mcordts/cityscapesScripts.git --quiet - failed
#
# import sys, os, distutils.core
# !git clone 'https://github.com/facebookresearch/detectron2'
# dist = distutils.core.run_setup("./detectron2/setup.py")
# !python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])} --quiet
# sys.path.insert(0, os.path.abspath('./detectron2'))
######
#@title 3. Import Libraries and other Utilities
######
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="oneformer")

# Import libraries
import numpy as np
import cv2
import torch
import imutils

# Import detectron2 utilities
from detectron2.config import get_cfg
from OneFormer.detectron2.projects.DeepLab.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from OneFormer.demo.defaults import DefaultPredictor
from OneFormer.demo.visualizer import Visualizer, ColorMode

import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import OneFormer Project
from OneFormer.oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    # add_dinat_config,
    add_convnext_config,
)

######
# @title 4. Define helper functions
######
cpu_device = torch.device("cpu")
SWIN_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
                 "coco": "configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
                 "ade20k": "configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml", }

DINAT_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
                  "coco": "configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
                  "ade20k": "configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml", }


def setup_cfg(dataset, model_path, use_swin):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    # add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    if use_swin:
        cfg_path = SWIN_CFG_DICT[dataset]
    else:
        cfg_path = DINAT_CFG_DICT[dataset]
    cfg_path = os.path.abspath(cfg_path)
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    return cfg


def setup_modules(dataset, model_path, use_swin):
    cfg = setup_cfg(dataset, model_path, use_swin)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
    )
    # if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
    #     from cityscapesscripts.helpers.labels import labels
    #     stuff_colors = [k.color for k in labels if k.trainId != 255]
    #     metadata = metadata.set(stuff_colors=stuff_colors)

    return predictor, metadata


def panoptic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    out = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to(cpu_device), segments_info, alpha=0.5
    )
    return out


def instance_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "instance")
    instances = predictions["instances"].to(cpu_device)
    out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)
    return out


def semantic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "semantic")
    out = visualizer.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=0.5
    )
    return out


TASK_INFER = {"panoptic": panoptic_run,
              "instance": instance_run,
              "semantic": semantic_run}


######
#@title A. Initialize Model
######
# download model checkpoint
use_swin = True
model_dir = os.path.abspath("checkpoints/oneformer")
if not use_swin:
  if not os.path.exists(f"{model_dir}/250_16_dinat_l_oneformer_ade20k_160k.pth"):
    subprocess.run('wget https://shi-labs.com/projects/oneformer/ade20k/250_16_dinat_l_oneformer_ade20k_160k.pth', shell=True)
  predictor, metadata = setup_modules("ade20k", "250_16_dinat_l_oneformer_ade20k_160k.pth", use_swin)
else:
  if not os.path.exists(f"{model_dir}/250_16_swin_l_oneformer_ade20k_160k.pth"):
    subprocess.run('wget https://shi-labs.com/projects/oneformer/ade20k/250_16_swin_l_oneformer_ade20k_160k.pth', shell=True)
  predictor, metadata = setup_modules("ade20k", f"{model_dir}/250_16_swin_l_oneformer_ade20k_160k.pth", use_swin)


######
#@title B. Display Sample Image. You can modify the path and try your own images!
######

# change path here for another image
img = cv2.imread("data/sample_images/room_example.jpg")
img = imutils.resize(img, width=640)


######
#@title C. Run Inference (CPU)
#@markdown Specify the **task**. `Default: panoptic`. Execution may take upto 2 minutes
######
###### Specify Task Here ######
task = "panoptic" #@param
##############################
for task, _ in TASK_INFER.items():
    out = TASK_INFER[task](img, predictor, metadata).get_image()
    cv2.imshow("image", out[:, :, ::-1])
    cv2.waitKey(0)
    cv2.imwrite(f"{task}_result.png", out)
