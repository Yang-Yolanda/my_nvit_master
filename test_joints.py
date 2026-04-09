import sys
sys.path.append("/home/yangz/4D-Humans")
from hmr2.configs import dataset_eval_config
cfg = dataset_eval_config()

print("\n--- 3DPW-TEST ---")
print("KEYPOINT_LIST:", cfg['3DPW-TEST'].KEYPOINT_LIST)

print("\n--- H36M-VAL-P2 ---")
print("KEYPOINT_LIST:", cfg['H36M-VAL-P2'].KEYPOINT_LIST)

from hmr2.utils import pose_utils
print("\nLook in hmr2 configs to see if joint names are defined.")
