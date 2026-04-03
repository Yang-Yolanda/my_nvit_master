#!/home/yangz/.conda/envs/4D-humans/bin/python
import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import pickle
import joblib

# Add paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append('/home/yangz/4D-Humans')
sys.path.append('/home/yangz/4D-Humans/PHALP-master')

from model_manager import ModelManager
from phalp.trackers.PHALP import PHALP as PHALP
from phalp.configs.base import FullConfig
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn

# --- Dummy Autoencoder for PHALP Compatibility ---
class DummyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, en=True):
        # x is assumed (B, 2048)
        # Return embedding. PHALP default is 512 dim usually?
        # Let's try 512.
        return torch.zeros((x.shape[0], 512), device=x.device)

# --- Wrapper to make Pruned Model look like HMR2Predictor ---
class PrunedHMRWrapper(nn.Module):
    def __init__(self, checkpoint_path, cfg):
        super().__init__()
        print(f"🚀 Loading Pruned Model Wrapper: {checkpoint_path}")
        self.mm = ModelManager({'device': 'cuda'})
        self.mm.load_model(checkpoint_path=checkpoint_path)
        self.model = self.mm.model
        self.model.eval()
        self.cfg = cfg
        
        # Expose SMPL for PHALP
        if hasattr(self.model, 'smpl'):
            self.smpl = self.model.smpl
        else:
            if hasattr(self.model, 'module'):
                self.smpl = self.model.module.smpl
            else:
                self.smpl = None 
        
        # Mock Autoencoder for Appearance Tracking
        self.autoencoder_hmar = DummyAutoencoder()

        self.img_size = 256
        self.focal_length = 5000.

    def forward(self, x):
        batch = {
            'img': x[:, :3, :, :],
            'mask': (x[:, 3, :, :]).clip(0, 1)
        }
        
        with torch.no_grad():
            out = self.model(batch)
        
        B = x.shape[0]
        uv_image = torch.zeros((B, 4, 256, 256), device=x.device)
        uv_vector = torch.zeros((B, 2048), device=x.device) 
        
        return {
            'pose_smpl': out['pred_smpl_params'],
            'pred_cam': out['pred_cam'],
            'pred_keypoints_3d': out['pred_keypoints_3d'],
            'pred_vertices': out['pred_vertices'],
            'pred_cam_t': out['pred_cam_t'],
            'uv_image': uv_image,
            'uv_vector': uv_vector
        }

# --- Custom Tracker ---
class PrunedTracker(PHALP):
    def __init__(self, cfg, pruned_ckpt):
        self.pruned_ckpt = pruned_ckpt
        super().__init__(cfg)
        
    def setup_hmr(self):
        print("🔧 Setting up Pruned HMR...")
        self.HMAR = PrunedHMRWrapper(self.pruned_ckpt, self.cfg)

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Pruned model path')
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--out_folder', type=str, default='diagnosis_out')
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'])
    parser.add_argument('--save_video', action='store_true', help='Render and save video visualization')
    parser.add_argument('--save_npz', action='store_true', help='Save results as NPZ')
    args = parser.parse_args()

    cfg = OmegaConf.structured(FullConfig)
    cfg.video.source = args.video_path
    cfg.video.output_dir = args.out_folder
    
    if args.save_video:
        cfg.render.enable = True 
        cfg.render.type = 'human_mesh'
    else:
        cfg.render.enable = False

    print(f"�� Starting Pruned Tracking on {args.video_path}")
    tracker = PrunedTracker(cfg, args.checkpoint)
    
    try:
        outputs = tracker.track()
    except Exception as e:
        print(f"❌ Tracking Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    pkl_path = None
    if isinstance(outputs, tuple):
        _, pkl_path = outputs
    else:
        vid_name = os.path.basename(args.video_path).split('.')[0]
        pkl_path = os.path.join(args.out_folder, 'results', f'{vid_name}.pkl')

    if pkl_path and os.path.exists(pkl_path):
        print(f"✅ Tracking Complete. Results at {pkl_path}")
        convert_to_npz(pkl_path, args.out_folder)
    else:
        print("⚠️ Could not locate output .pkl file.")

def convert_to_npz(pkl_path, out_folder):
    print("🔄 Converting PHALP results to Diagnosis NPZ...")
    try:
        data = joblib.load(pkl_path)
    except Exception as e:
        print(f"❌ Failed to load PKL: {e}")
        return
    
    img_names = []
    joints = []
    poses = []
    betas = []
    
    # Sort frames
    frames = sorted(data.keys())
    
    for frame in frames:
        people = data[frame]
        if not people: continue
        
        # Take the first person
        p = people[0]
        
        img_names.append(frame)
        
        if '3d_joints' in p:
             joints.append(p['3d_joints'])
        if 'pose' in p:
             poses.append(p['pose'])
        if 'betas' in p:
             betas.append(p['betas'])
             
    if len(joints) > 0:
        npz_name = os.path.join(out_folder, 'diagnosis_result.npz')
        np.savez_compressed(
            npz_name,
            img_name=img_names,
            jointPositions=np.array(joints),
            poses=np.array(poses),
            betas=np.array(betas)
        )
        print(f"💾 Diagnosis NPZ saved: {npz_name}")
    else:
        print("⚠️ No valid data extracted for NPZ.")

if __name__ == '__main__':
    main()
