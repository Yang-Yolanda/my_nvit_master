#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import sys
import os
import argparse
from pathlib import Path
import logging
import numpy as np
import cv2
from torchvision.transforms import Normalize

# Ensure we can import from the parent diagnostic_core
sys.path.append(str(Path(__file__).resolve().parent.parent))
from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_single_image_test():
    try:
        from hmr2.models import load_hmr2
    except ImportError:
        print("Error: HMR2/4D-Humans dependencies not found.")
        return

    # 1. Load Model (HMR2)
    print("\n[1/4] Loading HMR2 Model...")
    ckpt_path = '/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
    if not os.path.exists(ckpt_path):
        import glob
        candidates = glob.glob('/home/yangz/.cache/4DHumans/**/epoch=35*.ckpt', recursive=True)
        if candidates:
            ckpt_path = candidates[0]
        else:
            print("Checkpoint not found!")
            return

    model, _ = load_hmr2(ckpt_path)
    model.eval()

    # 2. Setup Diagnostic Lab
    print("[2/4] Setting up Diagnostic Lab...")
    wrapper = get_wrapper(model, 'hmr2')
    lab = ViTDiagnosticLab(wrapper, model_name='hmr2', output_root='results_single')
    
    # 3. Load One Image Manually (Bypass Dataset Config Hell)
    print("[3/4] Loading an image manually...")
    # Using the image found by `find` command
    # I will rely on the tool finding one. 
    # For now, I'll assume one exists or search inside python if I can't pass argument easily.
    # Actually, I'll search here.
    
    import glob
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    candidates = glob.glob(os.path.join(img_dir, '**/*.jpg'), recursive=True)
    if not candidates:
        print("No images found in 3DPW.")
        return
    img_path = candidates[0]
    print(f"Using image: {img_path}")
    
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # HMR2 Preprocessing
    # Resize to 256x256 (HMR2 default input)
    # Then it crops centrally or we just resize. HMR2 usually does 256 then crop 224 or uses 256 directly for ViT?
    # HMR2 ViT uses 224 or 256? Config said IMAGE_SIZE=224 in my previous patch.
    # Standard ViT is 224.
    
    input_size = 256 # HMR2 requires 256x256
    img_resized = cv2.resize(img_cv, (input_size, input_size))
    
    # Normalize
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1) # C, H, W
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_norm = (img_tensor - mean) / std
    
    # Add batch dim
    batch = {'img': img_norm.unsqueeze(0)}
    
    # Move to CPU (avoid OOM with concurrent benchmarks)
    batch = wrapper.to_device(batch, 'cpu')
    model.to('cpu')
    
    # 4. Run Inference
    print("[4/4] Running Inference...")
    with torch.no_grad():
        _ = wrapper(batch)
        
    # 5. Print Results
    print("\n" + "="*50)
    print("      SINGLE IMAGE DIAGNOSTIC REPORT")
    print("="*50)
    print(f"Image: {os.path.basename(img_path)}")
    print("-" * 50)
    
    print(f"{'Layer':<5} | {'Entropy (Focus)':<15} | {'KTI (Structure)':<15} | {'Interpretation'}")
    print("-" * 80)
    
    for i in range(10): 
        if i in lab.layer_metrics:
            if lab.layer_metrics[i]['entropy']:
                ent = lab.layer_metrics[i]['entropy'][0]
                kti = lab.layer_metrics[i]['kti'][0]
                
                if ent > 3.5: focus = "Global"
                elif ent > 2.5: focus = "Regional"
                else: focus = "Local (Joints)"
                
                print(f"{i:<5} | {ent:.4f}          | {kti:.4e}      | {focus}")
            else:
                print(f"{i:<5} | No Data")
        else:
            print(f"{i:<5} | N/A")
            
    print("="*50)
    print("\nDone.")

if __name__ == "__main__":
    run_single_image_test()
