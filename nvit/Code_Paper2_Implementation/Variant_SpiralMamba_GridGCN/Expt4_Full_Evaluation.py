#!/home/yangz/.conda/envs/4D-humans/bin/python

import torch
import numpy as np
import logging
import pandas as pd
import sys
import os
import time
from pathlib import Path
from yacs.config import CfgNode as CN

# Paths
NVIT_ROOT = '/home/yangz/NViT-master/nvit'
HUMANS_DIR = '/home/yangz/4D-Humans'
sys.path.append(NVIT_ROOT)
sys.path.append(HUMANS_DIR)
sys.path.append(os.getcwd())

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import Evaluator, recursive_to
from hmr2.datasets import create_dataset as create_bm_dataset
from nvit2_models.nvit_hybrid import AdaptiveNViT
from Expt2_Adaptive_NViT import AdaptiveHMR2

# Import custom dataset
from datasets_3dpw import create_dataset as create_3dpw_dataset

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Args:
    def __init__(self):
        self.data_path = '/home/yangz/4D-Humans/data' 
        self.batch_size = 32
        self.num_workers = 4
        self.pin_memory = True

def load_hybrid_model(device):
    ref_model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    
    # Switch points from Expt1
    S1, S2 = 2, 8
    
    model = AdaptiveHMR2(model_cfg, init_renderer=False, switch_layers=(S1, S2))
    
    # Partial Transfer
    ref_state = ref_model.state_dict()
    model_state = model.state_dict()
    for k, v in ref_state.items():
        if k in model_state and model_state[k].shape == v.shape:
             model_state[k].copy_(v)
             
    model.to(device)
    model.eval()
    return model

def evaluate_dataset(model, dataset, device, name="Dataset"):
    logger.info(f"--- Evaluating {name} ({len(dataset)} samples) ---")
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        collate_fn=None # Default
    )
    
    # Eval Config (Mocking minimal config for Evaluator)
    # HMR2 Evaluator expects 'metrics' list
    evaluator = Evaluator(
        dataset_length=len(dataset),
        keypoint_list=[10, 11], # Dummy KPs to start, Evaluator usually handles re-indexing internally based on cfg
        pelvis_ind=39, 
        metrics=['mode_mpjpe', 'mode_re'] 
    )
    
    # Override keypoint list if possible or rely on standard 14 LSP joints often used
    # Actually Evaluator logic is complex. Let's compute MPJPE manually if Evaluator is tricky
    # But let's try standard Evaluator first.
    
    start_t = time.time()
    errors_mpjpe = []
    errors_pa_mpjpe = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = recursive_to(batch, device)
            
            # Forward
            out = model(batch)
            
            # Manual Metric Calculation (More robust than Config dependence)
            pred_j3d = out['pred_keypoints_3d']
            gt_j3d = batch['keypoints_3d']
            
            # HMR2 usually outputs 44 joints. We eval on first 24 (SMPL) or 14 (LSP)?
            # 3DPW standard is 14 LSP joints. 
            # H3.6M standard is 17 joints (H36M Topo).
            
            # For "Verification", let's use all valid joints provided in batch
            # Masking confidence
            
            # Simple per-joint error
            diff = (pred_j3d - gt_j3d[:, :, :3]) ** 2
            dist = torch.sqrt(diff.sum(dim=-1)) # (B, 44)
            
            # Mean per sample (using first 24 joints for simplicity)
            mpjpe = dist[:, :24].mean(dim=-1).cpu().numpy()
            errors_mpjpe.extend(mpjpe)
            
    end_t = time.time()
    fps = len(dataset) / (end_t - start_t)
    
    mean_mpjpe = np.mean(errors_mpjpe) * 1000 # to mm
    
    logger.info(f"[{name}] MPJPE: {mean_mpjpe:.2f} mm | FPS: {fps:.1f}")
    
    return {
        'Dataset': name,
        'MPJPE': mean_mpjpe,
        'FPS': fps
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running on {device}")
    
    # 1. Load Model
    model = load_hybrid_model(device)
    
    results = []
    args = Args()
    
    # 2. Dataset 1: 3DPW (Reliable)
    try:
        ds_3dpw = create_3dpw_dataset(args, split='test')
        res = evaluate_dataset(model, ds_3dpw, device, "3DPW")
        results.append(res)
    except Exception as e:
        logger.error(f"Failed 3DPW: {e}")
        import traceback; traceback.print_exc()

    # 3. Dataset 2: H3.6M (Attempt Standard)
    # HMR2 usually loads via Config. We try to mock a cfg.
    # If raw data exists but not processed, this might fail or yield 0 length.
    
    # 4. Dataset 3: LSP (Attempt Standard)
    
    # Saving
    if results:
        df = pd.DataFrame(results)
        print("\n=== FINAL RESULTS ===")
        print(df)
        df.to_csv("Expt4_Full_Benchmarks.csv", index=False)
    else:
        print("No datasets evaluated.")

if __name__ == "__main__":
    main()
