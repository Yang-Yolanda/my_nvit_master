#!/home/yangz/.conda/envs/4D-humans/bin/python

import os
import torch
import numpy as np
import cv2
from PIL import Image
import requests
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys

# Add paths for diagnostic engine and easy_vitpose
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(os.path.join(base_path, 'Paper1_Diagnostics/diagnostic_core'))
sys.path.append(os.path.join(base_path, 'Paper1_Diagnostics/Experiment2_KTI'))
sys.path.append(os.path.join(base_path, 'external_models/PromptHMR/pipeline/detector/ViTPose'))
sys.path.append(os.path.join(base_path, 'external_models/PromptHMR/pipeline/detector/ViTPose/easy_vitpose'))

from diagnostic_engine import ViTDiagnosticLab, ModelWrapper
from evaluate_hamer_kti_real import AttentionCapture # Reuse if needed, but CLIP used out.attentions

# --- ViTPose Animal Topology (AP10K) ---
# 17 Joints
AP10K_JOINTS = [
    'L_Eye', 'R_Eye', 'Nose', 'Neck', 'Root_of_Tail',
    'L_Shoulder', 'L_Elbow', 'L_Front_Paw',
    'R_Shoulder', 'R_Elbow', 'R_Front_Paw',
    'L_Hip', 'L_Knee', 'L_Back_Paw',
    'R_Hip', 'R_Knee', 'R_Back_Paw'
]

# Connections (0-indexed)
AP10K_PARENTS = {
    0: 2,  # L_Eye -> Nose
    1: 2,  # R_Eye -> Nose
    2: 3,  # Nose -> Neck
    3: 5,  # Neck -> L_Shoulder
    3: 8,  # Neck -> R_Shoulder
    3: 4,  # Neck -> Root_of_Tail (Spine)
    5: 6,  # L_Shoulder -> L_Elbow
    6: 7,  # L_Elbow -> L_Front_Paw
    8: 9,  # R_Shoulder -> R_Elbow
    9: 10, # R_Elbow -> R_Front_Paw
    4: 11, # Root_of_Tail -> L_Hip
    4: 14, # Root_of_Tail -> R_Hip
    11: 12, # L_Hip -> L_Knee
    12: 13, # L_Knee -> L_Back_Paw
    14: 15, # R_Hip -> R_Knee
    15: 16, # R_Knee -> R_Back_Paw
}

def get_ap10k_parents():
    parents = [-1] * 17
    for child, parent in AP10K_PARENTS.items():
        parents[child] = parent
    return parents

# --- Model Loading Logic ---

class ViTPoseAnimalWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        # ViTPose architecture usually has a .backbone and .keypoint_head
        self.backbone = model.backbone
        
    def get_backbone(self):
        return self.backbone
    
    def forward(self, batch):
        # batch['img'] is (B, 3, H, W)
        # ViTPose returns heatmaps
        img = batch['img']
        # ViTPose usually expects 256x192
        return self.model(img)

def download_weights(target_path):
    # Corrected URL for ViTPose-Base AP10K weights
    url = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/ap10k/vitpose-b-ap10k.pth"
    print(f"Downloading ViTPose weights from {url}...")
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def run_evaluation(model_path, image_folder, num_samples=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    from easy_vitpose.vit_models.model import ViTPose
    from easy_vitpose.configs.ViTPose_ap10k import model_base as model_cfg
    
    model = ViTPose(model_cfg)
    if not os.path.exists(model_path):
        download_weights(model_path)
        
    ckpt = torch.load(model_path, map_location='cpu')
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device)
    model.eval()
    
    wrapper = ViTPoseAnimalWrapper(model)
    
    # Define a customized Lab for ViTPose
    class ViTPoseLab(ViTDiagnosticLab):
        def _patch_attention_modules(self):
            # ViTPose backbone is often timm-like but might have nested blocks
            # We'll see if the standard patcher works
            super()._patch_attention_modules()

    lab = ViTPoseLab(wrapper, model_name="Animal", output_root='nvit/Paper1_Diagnostics/Experiment2_KTI/results')
    lab.parents = get_ap10k_parents()
    
    # 2. Setup Data
    # Since we might not have a full dataset, we'll try to find animal images locally or use a placeholder
    image_paths = list(Path(image_folder).glob('**/*.jpg')) + list(Path(image_folder).glob('**/*.png'))
    if not image_paths:
        print(f"No images found in {image_folder}. Downloading a sample...")
        sample_url = "https://raw.githubusercontent.com/ViTAE-Transformer/ViTPose/main/demo/zebra.jpg"
        sample_path = Path("nvit/Paper1_Diagnostics/Experiment2_KTI/sample_animal.jpg")
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(sample_url)
        with open(sample_path, "wb") as f:
            f.write(resp.content)
        image_paths = [sample_path]

    # For KTI validation on a real model, we need GROUND TRUTH keypoints.
    # If we don't have them, we can use the model's own predictions as a "Self-Consistency" check.
    # If KTI is high between attention and pred_keypoints, it means the model's attention 
    # is spatially aligned with its output topology.
    
    lab.groups = {'Control': {'mask_layers': [], 'mode': 'none'}}
    
    count = 0
    for img_path in tqdm(image_paths[:num_samples], desc="Evaluating ViTPose Animal KTI"):
        img_orig = cv2.imread(str(img_path))
        if img_orig is None: continue
        
        # Preprocess
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (192, 256)) # ViTPose standard
        
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get Predictions to use as "GT" for validation
            # (Self-consistency check to see if attention aligns with pred topological nodes)
            heatmaps = model(img_tensor) # (1, 17, 64, 48)
            
            # Extract keypoints from heatmaps
            B, J, H_h, W_h = heatmaps.shape
            kp_2d = []
            for j in range(J):
                hm = heatmaps[0, j]
                idx = hm.flatten().argmax().item()
                y, x = divmod(idx, W_h)
                # Scale to input image size (256x192)
                kp_2d.append([x * (192 / W_h), y * (256 / H_h), hm[y, x].item()])
            
            kp_2d = torch.tensor(kp_2d).unsqueeze(0).to(device)
            
            # Inject into batch for KTI
            batch = {'img': img_tensor, 'keypoints_2d': kp_2d}
            
            # INJECT INTO LAB STATE (Required when bypassing run_experiment)
            lab.current_keypoints = kp_2d
            lab.current_batch_adj = lab.compute_kti_adjacency(batch)
            
            # Forward via lab (triggers hooks)
            lab.wrapper(batch)
            
            # Note: diagnostic_engine computes KTI inside the hooks based on `batch['keypoints_2d']`
            
        count += 1
        if count >= num_samples: break
        
    print(f"Processed {count} samples.")
    # Explicitly save results
    serializable_metrics = {}
    for layer_idx, metrics in lab.layer_metrics.items():
        serializable_metrics[layer_idx] = {
            'entropy': [float(x) for x in metrics['entropy']],
            'kti': [float(x) for x in metrics['kti']],
            'rank': [float(x) for x in metrics['rank']]
        }
    
    # Results are now handled by lab.run_experiment or similar
    # But for manual saving:
    out_dir = Path(lab.output_dir) / 'Control'
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / 'layer_metrics_Control.json'
    import json
    with open(metrics_file, 'w') as f:
        json.dump(serializable_metrics, f)
    
    # Save a small results.csv for consistency
    results_file = out_dir / 'results.csv'
    pd_results = {'Group': ['Control'], 'MPJPE': [0.0], 'Avg_KTI': [np.mean(kti_vals)]}
    import pandas as pd
    pd.DataFrame(pd_results).to_csv(results_file, index=False)
    
    print(f"Metrics saved to {metrics_file}")
    
    # Aggregate and Print
    kti_vals = [np.mean(v['kti']) for v in lab.layer_metrics.values() if v['kti']]
    if kti_vals:
        print(f"Average KTI across all layers: {np.mean(kti_vals):.4f}")
    else:
        print("KTI values seem empty. Checking hooks...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='nvit/Paper1_Diagnostics/Experiment2_KTI/weights/vitpose-b-ap10k.pth')
    parser.add_argument('--image_folder', type=str, default='/home/yangz/4D-Humans/data/3DPW/imageFiles')
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    
    run_evaluation(args.model_path, args.image_folder, args.num_samples)
