#!/home/yangz/.conda/envs/4D-humans/bin/python

import torch
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import time

# Paths
NVIT_ROOT = '/home/yangz/NViT-master/nvit'
HUMANS_DIR = '/home/yangz/4D-Humans'
sys.path.append(NVIT_ROOT)
sys.path.append(HUMANS_DIR)
sys.path.append(os.getcwd())

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from nvit2_models.nvit_hybrid import AdaptiveNViT
from Expt2_Adaptive_NViT import AdaptiveHMR2
from datasets_3dpw import create_dataset
from hmr2.utils import Evaluator, recursive_to

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def measure_gflops(model, device):
    try:
        from thop import profile
        dummy = torch.randn(1, 3, 256, 192).to(device)
        macs, params = profile(model, inputs=({'img': dummy}, ), verbose=False)
        return macs / 1e9, params / 1e6
    except ImportError:
        logger.warning("thop not found")
        return 0, 0
    except Exception as e:
        logger.warning(f"GFLOPs measurement failed: {e}")
        return 0, 0

def evaluate_model(model, dataloader, device, name="Model", max_samples=100):
    model.eval()
    dataset_cfg = dataloader.dataset.dataset_cfg if hasattr(dataloader.dataset, 'dataset_cfg') else None
    
    # Eval Config
    from hmr2.configs import dataset_eval_config
    if dataset_cfg is None: dataset_eval_config()['3DPW-TEST']
    if dataset_cfg is None: dataset_cfg = dataset_eval_config()['3DPW-TEST']
    
    evaluator = Evaluator(
        dataset_length=int(1e8),
        keypoint_list=dataset_cfg.KEYPOINT_LIST,
        pelvis_ind=39,
        metrics=['mode_mpjpe']
    )
    
    start_time = time.time()
    count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_samples: break
            
            # Move to device
            batch = recursive_to(batch, device)
            # Forward
            out = model(batch)
            # Eval
            evaluator(out, batch)
            count += 1
            
    end_time = time.time()
    throughput = count / (end_time - start_time)
    
    metrics = evaluator.get_metrics_dict()
    mpjpe = metrics['mode_mpjpe']
    
    # GFLOPs
    gflops, params = measure_gflops(model, device)
    
    logger.info(f"[{name}] MPJPE: {mpjpe:.2f}mm | GFLOPs: {gflops:.2f}G | Params: {params:.1f}M | FPS: {throughput:.1f}")
    
    return {
        'Name': name,
        'MPJPE': mpjpe,
        'GFLOPs': gflops,
        'Params': params,
        'FPS': throughput
    }

def main():
    logger.info("Starting Expt3: Comparative Study...")
    device = torch.device('cuda')
    
    # 1. Load Baseline (Reference)
    ref_model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    ref_model.to(device)
    
    # 2. Config & Data
    class Args:
        batch_size = 1 # Batch 1 for accurate inference timing/flops
        num_workers = 0
        pin_mem = True
        data_path = '/home/yangz/4D-Humans/data' 
        
    dataset = create_dataset(Args(), split='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 3. Models to Compare
    results = []
    
    # A. Baseline
    results.append(evaluate_model(ref_model, dataloader, device, name="Baseline (ViT)", max_samples=50))
    
    # Helper to Build Variants
    def build_variant(s1, s2, name):
        logger.info(f"Building {name} (Switches: {s1}, {s2})...")
        model = AdaptiveHMR2(model_cfg, init_renderer=False, switch_layers=(s1, s2))
        
        # Load Weights (Partial Load)
        ref_state = ref_model.state_dict()
        model_state = model.state_dict()
        
        # Transfer what matches
        for k, v in ref_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k].copy_(v)
                
        model.to(device)
        return evaluate_model(model, dataloader, device, name=name, max_samples=50)

    # B. Pure Mamba (After Patch Embed, all Mamba)
    # Switch 1 = 0 (Start Mamba immediately). Switch 2 = 32 (Never GCN)
    # NOTE: ViT typically has patch embed -> blocks. 
    # AdaptiveNViT: if i < s1: ViT. 
    # So if s1=0, layer 0 is Mamba.
    results.append(build_variant(0, 32, "Pure Mamba Backbone"))

    # C. Pure GCN
    # Switch 1 = 0, Switch 2 = 0 (All GCN)
    # results.append(build_variant(0, 0, "Pure GCN Backbone")) 
    # Warning: Pure GCN might be unstable or fail drastically without global attention init? 
    # Let's try "Early GCN" (ViT L0-1 -> GCN)? Or purely GCN.
    # User Hypothesis: "Full Replacement A (All-GCN): Fails at Touching"
    results.append(build_variant(0, 0, "Pure GCN Backbone"))

    # D. Proposed Hybrid
    # Based on Expt1: L0 ViT Peak, L7 KTI Peak.
    # So Switch 1 = 1 (L0 is ViT), Switch 2 = 8 (L1..7 Mamba, L8+ GCN)?
    # Expt1 Log said: Max Ent @ L0. Max KTI @ L7.
    # Wait, if Max Ent is L0, we should keep L0? Or switch AFTER L0?
    # Logic: Entropy High -> Visual Search -> ViT.
    # Entropy drops -> We found parts -> Switch to Mamba.
    # So we want ViT UNTIL Entropy drops significantly? Or just at peak?
    # Usually peak is early. Let's say keep 2 layers ViT.
    # KTI Peak is L7. This means Structure is Assembly is COMPLETE at L7.
    # So Switch to GCN AFTER L7. i.e. L8.
    
    results.append(build_variant(2, 8, "Adaptive NViT (Proposed)"))
    
    # 4. Save & Plot
    df = pd.DataFrame(results)
    df.to_csv("Expt3_Results.csv", index=False)
    print(df)
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    colors = ['gray', 'orange', 'red', 'green']
    x = np.arange(len(df))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, df['MPJPE'], width, label='MPJPE (mm)', color='tab:blue')
    ax1.set_ylabel('MPJPE (Lower is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Name'], rotation=15)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, df['GFLOPs'], width, label='GFLOPs', color='tab:orange')
    ax2.set_ylabel('GFLOPs (Lower is Better)')
    
    plt.title('NViT Paper 2: Comparative Study')
    fig.tight_layout()
    plt.savefig('Expt3_Comparative_Study.png')
    
    # Write Conclusion Log
    with open("Expt3_Comparative_Study.log", "w") as f:
        f.write(df.to_markdown())
        
        # Analyze
        best_acc = df.loc[df['MPJPE'].idxmin()]
        best_eff = df.loc[df['GFLOPs'].idxmin()]
        
        f.write(f"\n\nCONCLUSION:\n")
        f.write(f"Most Accurate: {best_acc['Name']} ({best_acc['MPJPE']:.2f}mm)\n")
        f.write(f"Most Efficient: {best_eff['Name']} ({best_eff['GFLOPs']:.2f}G)\n")

if __name__ == "__main__":
    main()
