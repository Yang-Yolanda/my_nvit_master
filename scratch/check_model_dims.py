import torch
import sys
from pathlib import Path

# Setup paths
sys.path.append('/home/yangz/NViT-master')
sys.path.append('/home/yangz/4D-Humans')

from nvit.utils.model_io import load_model_from_ckpt

ckpt_path = "/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
device = "cpu" # Use CPU to avoid CUDA errors if driver is stuck

try:
    model = load_model_from_ckpt(ckpt_path, device=device)
    print(f"Model type: {type(model)}")
    
    # Check keypoint counts
    if hasattr(model, 'smpl_head'):
         if hasattr(model.smpl_head, 'joint_regressor'):
             print(f"Joint regressor shape: {model.smpl_head.joint_regressor.shape}")
         else:
             print("smpl_head found, but no joint_regressor attribute.")
    
    # Try a dummy forward pass
    dummy_img = torch.zeros(1, 3, 256, 256)
    batch = {'img': dummy_img}
    with torch.no_grad():
        out = model(batch)
        print(f"pred_keypoints_3d shape: {out['pred_keypoints_3d'].shape}")
        print(f"pred_keypoints_2d shape: {out['pred_keypoints_2d'].shape}")

except Exception as e:
    print(f"Error: {e}")
