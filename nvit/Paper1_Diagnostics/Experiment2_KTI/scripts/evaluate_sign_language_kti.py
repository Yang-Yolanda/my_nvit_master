#!/home/yangz/.conda/envs/4D-humans/bin/python

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# Add paths for HaMeR and Diagnostic Engine
sys.path.append(os.path.abspath('nvit/Paper1_Diagnostics/diagnostic_core'))
sys.path.append(os.path.abspath('nvit/external_models/hamer'))

from diagnostic_engine import ViTDiagnosticLab, ModelWrapper
from evaluate_hamer_kti_real import HaMeRWrapper, AttentionCapture

def run_sign_language_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_path = 'nvit/external_models/hamer/example_data/test1.jpg'
    
    if not os.path.exists(img_path):
        print("Image not found. Generating dummy for logic check...")
        img = np.zeros((256, 256, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

    # Preprocess for HaMeR
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 1. Define Hand Topology (21 Joints)
    # Parents for MediaPipe/HaMeR hand
    HAND_PARENTS = [
        -1, 0, 1, 2, 3, # Thumb
        0, 5, 6, 7,    # Index
        0, 9, 10, 11,  # Middle
        0, 13, 14, 15, # Ring
        0, 17, 18, 19  # Pinky
    ]
    
    # Dummy/Estimated Hand Keypoints for KTI (In a real sign language app, we'd use MediaPipe)
    # We'll use a standardized hand pose for this validation
    kp_2d = torch.zeros((1, 21, 2))
    # ... fill some reasonable values or keep 0 for random baseline test ...
    # Let's use the center of the image and spread branches
    kp_2d[0, 0] = torch.tensor([128, 200]) # Wrist
    # Simplified Finger Spreading
    for i in range(1, 21):
        kp_2d[0, i] = torch.tensor([128 + (i % 5 - 2) * 20, 200 - (i // 5 + 1) * 30])
    
    kp_2d = (kp_2d / 256.0) * 2.0 - 1.0 # Scale to [-1, 1] for Lab
    kp_2d = kp_2d.to(device)

    # 2. Setup HaMeR
    from hamer.models import load_hamer
    from hamer.configs import CACHE_DIR_HAMER
    model, cfg = load_hamer(os.path.join(CACHE_DIR_HAMER, 'hamer_vit_L_cc_pretrain.ckpt'))
    model.to(device)
    model.eval()

    wrapper = HaMeRWrapper(model)
    lab = ViTDiagnosticLab(wrapper, model_name="HaMeR_SignLanguage", output_root='nvit/Paper1_Diagnostics/Experiment2_KTI/results')
    lab.parents = HAND_PARENTS
    lab.current_keypoints = kp_2d
    lab.current_batch_adj = lab.compute_kti_adjacency({'keypoints_2d': kp_2d})
    lab.current_feature_grid = (16, 16) # HaMeR uses 16x16 Vit-L

    # Create capture
    # HaMeR Vit-L blocks are in model.backbone.vit.blocks
    capturer = AttentionCapture(model.backbone.vit.blocks)
    
    print("Evaluating HaMeR on Sign Language Hand...")
    with torch.no_grad():
        _ = model(img_tensor)
        attns = capturer.get_stacked() # (L, 1, H, N, N)
        
        layer_metrics = {}
        for l_idx, attn_map in enumerate(attns):
            kti = lab.calculate_physically_grounded_kti(attn_map, kp_2d)
            layer_metrics[l_idx] = {'kti': [float(kti)]}

    # Save
    out_dir = Path('nvit/Paper1_Diagnostics/Experiment2_KTI/results/SignLanguage_HaMeR')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'layer_metrics_Control.json', 'w') as f:
        json.dump(layer_metrics, f)
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    import json
    run_sign_language_validation()
