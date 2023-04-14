import os.path as osp

import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from pycocotools.coco import COCO

from transforms import get_transform


ann = COCO('/Users/KSHITIJ/Desktop/CLF/University related/Y4/Final Year Project/Tiger Datasets/All datasets/atrw_anno_detection_train/json/_annotations.coco.json')
print(ann.keys)