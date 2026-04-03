#!/home/yangz/.conda/envs/4D-humans/bin/python

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import cv2

# Add paths for dependencies
sys.path.append(os.path.abspath('nvit/Paper1_Diagnostics/diagnostic_core'))
# easy_vitpose path
sys.path.append(os.path.abspath('external_models/PromptHMR/pipeline/detector/ViTPose'))
sys.path.append(os.path.abspath('external_models/PromptHMR/pipeline/detector/ViTPose/easy_vitpose'))

from diagnostic_engine import ViTDiagnosticLab, ModelWrapper

# Topology for AP10K (Proxy for distals)
HAND_PARENTS = [-1, 0, 1, 3, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # 17 joints

class ViTPoseHandWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)
    def get_backbone(self):
        return self.model.backbone
    def forward(self, batch):
        img = batch['img']
        return self.model(img)

class ViTPoseHandLab(ViTDiagnosticLab):
    def __init__(self, wrapper, **kwargs):
        super().__init__(wrapper, **kwargs)
        self.parents = [-1, 0, 1, 3, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def run_evaluation(model_path, image_path, num_samples=5, out_channels=133):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    from easy_vitpose.vit_models.model import ViTPose
    # Wholebody has 133 joints
    model = ViTPose(dict(
        backbone=dict(
            type='ViT',
            img_size=(256, 192),
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.3,
        ),
        keypoint_head=dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1, ),
            out_channels=out_channels, # Dynamic
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=True,
            post_process='default',
            shift_heatmap=True,
            modulate_kernel=11)
    ))

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    ckpt = torch.load(model_path, map_location='cpu')
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.to(device)
    model.eval()

    wrapper = ViTPoseHandWrapper(model)
    lab = ViTPoseHandLab(wrapper, model_name="ViTPose_Hand_SignLanguage", output_root='nvit/Paper1_Diagnostics/Experiment2_KTI/results')
    lab.current_feature_grid = (16, 12) # 256x192 / 16 = 16x12

    # Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Preprocess
    img_input = cv2.resize(img, (192, 256))
    img_input = torch.from_numpy(img_input).float().permute(2, 0, 1) / 255.0
    img_input = (img_input - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img_input = img_input.unsqueeze(0).to(device)

    # 2. Keypoints (Self-consistency check)
    with torch.no_grad():
        preds = model(img_input)
        # Preds is usually (1, 133, 64, 48)?
        # Get max indices for keypoints
        # Use all 17 available joints
        use_indices = list(range(17))
        
        # Max on heatmaps
        kp_2d = torch.zeros((1, 17, 2)).to(device)
        for i, idx in enumerate(use_indices):
            hm = preds[0, idx]
            pos = torch.argmax(hm)
            r, c = pos // hm.shape[1], pos % hm.shape[1]
            # Map back to 256x192
            kp_2d[0, i] = torch.tensor([c * 4, r * 4]).float().to(device)

    # Inject into Lab
    batch = {'img': img_input, 'keypoints_2d': kp_2d}
    lab.current_keypoints = kp_2d
    lab.current_batch_adj = lab.compute_kti_adjacency(batch)
    
    print("Evaluating Sign Language Hand KTI...")
    with torch.no_grad():
        lab.wrapper(batch) # Forward pass with hooks

    # Save results
    serializable_metrics = {}
    for layer_idx, metrics in lab.layer_metrics.items():
        serializable_metrics[layer_idx] = {
            'kti': [float(x) for x in metrics['kti']]
        }
    
    out_dir = Path(lab.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'layer_metrics_Control.json', 'w') as f:
        json.dump(serializable_metrics, f)
    print(f"Metrics saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='nvit/Paper1_Diagnostics/Experiment2_KTI/weights/vitpose-b-ap10k.pth')
    parser.add_argument('--image_path', type=str, default='nvit/external_models/hamer/example_data/test1.jpg')
    args = parser.parse_args()
    
    # Adjust model to 17 joints for AP10K weights
    def run_wrapper(model_path, image_path):
         run_evaluation(model_path, image_path, out_channels=17)
    
    run_evaluation(args.model_path, args.image_path, out_channels=17)
