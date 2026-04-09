import torch
import sys
import os
sys.path.append("/home/yangz/NViT-master/nvit/Code_Paper2_Implementation")
sys.path.append("/home/yangz/4D-Humans")

from hmr2.configs import dataset_eval_config
from hmr2.datasets import create_dataset
import argparse

cfg_eval = dataset_eval_config()
ds_name = '3DPW-TEST'
dataset_cfg = cfg_eval[ds_name]
print("keypoint_list:", dataset_cfg.KEYPOINT_LIST)
