#!/home/yangz/.conda/envs/4D-humans/bin/python

import torch
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT

def verify_joints():
    model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    print(f"SMPL NUM_BODY_JOINTS: {cfg.SMPL.NUM_BODY_JOINTS}")
    
    # Forward dummy
    dummy_x = torch.zeros(1, 3, 256, 192)
    with torch.no_grad():
        out = model.backbone(dummy_x)
        print(f"Backbone Out Shape: {out.shape}")
        
        smpl_params, cam, _ = model.smpl_head(out)
        print(f"Global Orient Shape: {smpl_params['global_orient'].shape}")
        print(f"Body Pose Shape: {smpl_params['body_pose'].shape}")
        
        total_joints = smpl_params['global_orient'].shape[1] + smpl_params['body_pose'].shape[1]
        print(f"Total Model Joints: {total_joints}")

if __name__ == "__main__":
    verify_joints()
