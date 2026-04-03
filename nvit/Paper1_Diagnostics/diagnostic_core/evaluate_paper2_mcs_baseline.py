#!/home/yangz/.conda/envs/4D-humans/bin/python

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.append("/home/yangz/NViT-master/nvit")
sys.path.append("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/diagnostic_core")
sys.path.append("/home/yangz/4D-Humans")

from diagnostic_engine import HMR2Wrapper
from motion_diagnostic import MotionCoherenceEngine
from hmr2.models import load_hmr2
from hmr2.models.discriminator import Discriminator
from datasets_3dpw import create_dataset

def run_baseline_mcs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    print("Loading HMR2 Baseline...")
    ckpt_path = "/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
    model, _ = load_hmr2(checkpoint_path=ckpt_path) 
    wrapper = HMR2Wrapper(model).to(device).eval()
    
    # 2. Load Discriminator
    print("Initializing Discriminator...")
    disc = Discriminator().to(device).eval()
    
    # 3. Setup Dataset
    print("Loading 3DPW Evaluation Set...")
    class Args:
        data_path = "/home/yangz/4D-Humans/data"
        batch_size = 8
        num_workers = 4
    dataset = create_dataset(Args(), split='test')
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    
    # 4. Initialize Engine
    engine = MotionCoherenceEngine(discriminator=disc, device=str(device))
    
    # 5. Evaluate
    print("Evaluating MCS on 20 batches...")
    all_metrics = []
    
    for i, batch in enumerate(tqdm(loader)):
        if i >= 20: break
        
        batch = wrapper.to_device(batch, device)
        with torch.no_grad():
            output = wrapper(batch)
            metrics = engine.evaluate_mcs(output)
            all_metrics.append(metrics)
            
    # 6. Aggregate
    avg_metrics = {}
    for k in all_metrics[0].keys():
        avg_metrics[k] = np.mean([m[k] for m in all_metrics])
        
    print("\n=== Baseline MCS Results (HMR2) ===")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save results
    res_path = Path("nvit/Paper1_Diagnostics/diagnostic_core/results/paper2_baseline_mcs.json")
    res_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(res_path, 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    print(f"\nResults saved to {res_path}")

if __name__ == "__main__":
    run_baseline_mcs()
