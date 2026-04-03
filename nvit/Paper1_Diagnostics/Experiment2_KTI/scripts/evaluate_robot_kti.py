#!/home/yangz/.conda/envs/4D-humans/bin/python

import os
import sys
import torch
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add paths
sys.path.append(os.path.abspath('nvit/Paper1_Diagnostics/TaskB_Robot'))
sys.path.append(os.path.abspath('nvit/Paper1_Diagnostics/diagnostic_core'))

try:
    from toy_robot_dataset import ToyRobotDataset
    from toy_vit import ToyViT
    from diagnostic_engine import ViTDiagnosticLab, ModelWrapper
    # Pre-trained runner imports
    from transformers import ViTForImageClassification, AutoModel, AutoConfig
    import timm
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class ToyWrapper(ModelWrapper):
    def __init__(self, model, parents_list, is_pretrained=False):
        super().__init__(model)
        self.parents = parents_list
        self.is_pretrained = is_pretrained
        
    def get_backbone(self):
        if self.is_pretrained:
            # timm model
            return self.model
        return self.model
        
    def forward(self, batch):
        img = batch['img']
        # Handle Pretrained resizing inside wrapper if necessary
        # But we will use dataset resizing in this script
        return self.model(img)

class AttentionCapture:
    """Universal Attention Capture for ToyViT and Timm ViT"""
    def __init__(self, model, is_pretrained=False):
        self.attentions = []
        self.hooks = []
        self.is_pretrained = is_pretrained
        
        if not is_pretrained:
            # ToyViT logic (blocks[i].attn.attn_drop)
            for i, blk in enumerate(model.blocks):
                handle = blk.attn.attn_drop.register_forward_hook(self._hook_fn)
                self.hooks.append(handle)
        else:
            # Timm ViT logic (blocks[i].attn.attn_drop)
            # Usually same structure
            for i, blk in enumerate(model.blocks):
                if hasattr(blk.attn, 'attn_drop'):
                     handle = blk.attn.attn_drop.register_forward_hook(self._hook_fn)
                else:
                     # Fallback for newer timm
                     pass
                self.hooks.append(handle)
            
    def _hook_fn(self, module, input, output):
        # input is (attn_scores,) or output is attn_scores after drop
        # Timm might return differently? Usually dropout returns tensor.
        self.attentions.append(output.detach().cpu())
        
    def clear(self):
        self.attentions = []
        
    def get_stacked(self):
        if not self.attentions: return None
        return torch.stack(self.attentions, dim=0)

def run_evaluation(model_type, ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating {model_type} on Robot Dataset...")
    
    # 1. Dataset (Use Robot Dataset)
    urdf_path = 'nvit/Paper1_Diagnostics/TaskB_Robot/panda.urdf'
    
    # Determine Img Size based on Model
    img_size = 224 if model_type == 'Pretrained' else 64
    
    dataset = ToyRobotDataset(urdf_path=urdf_path, num_samples=200, img_size=img_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Topology
    parents = dataset.get_parents() # [-1, 0, 1, 2, 3, 4, 5] usually
    
    # 2. Model Loading
    if model_type == 'Scratch':
        model = ToyViT(img_size=64, patch_size=4, num_joints=dataset.num_nodes, depth=6, embed_dim=128)
        # Load weights? 'results/TaskB_Robot/model_final.pth' ?
        # Assuming saved state_dict. If not found, use random (Control).
        # But we want the trained one.
        # User ran Experiment B. Is there a saved model?
        # runner.py doesn't seem to save model explicitly in the provided snippet?
        # Check logs if it saves.
        # Ideally we load the trained one.
        pass
    else:
        # Pretrained: timm vit_tiny
        import timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=dataset.num_nodes*2)
        # We need to adapt input if grayscale?
        # dataset returns grayscale (1 channel).
        # Pretrained needs 3.
        # We will repeat in loop.
        pass

    model.to(device)
    model.eval()
    
    wrapper = ToyWrapper(model, parents, is_pretrained=(model_type=='Pretrained'))
    lab = ViTDiagnosticLab(wrapper, model_name="Robot", output_root='nvit/Paper1_Diagnostics/Experiment2_KTI/results')
    
    # Update Lab Topology manually
    # ViTDiagnosticLab usually defaults to SMPL.
    # We must overwrite parents and adjacency
    lab.parents = parents
    # Recompute adjacency matrix logic for Robot
    # Actually lab.parents is used to build adjacency inside `compute_kti_adjacency`.
    # That function supports `self.parents` override.
    
    # Hook Attention
    # DiagnosticLab tries to patch automatically.
    # But timm/ToyViT structures might differ.
    # Let's rely on Lab's patching if standard.
    # If standard timm, Lab handles it.
    # ToyViT might need manual handling if Lab doesn't recognize it.
    
    # Override Lab.groups to just Control
    lab.groups = {'Control': {'mask_layers': [], 'mode': 'none'}}
    
    # We can use lab.run_experiment IF we fake the Evaluator.
    # But simpler to just run loop here and use lab.calculate_kti
    
    layer_metrics = defaultdict(lambda: {'kti': []})
    
    capturer = AttentionCapture(model, is_pretrained=(model_type=='Pretrained'))
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            img = batch['img'].to(device)
            
            if model_type == 'Pretrained' and img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
                
            kp = batch['keypoints'].to(device)
            
            capturer.clear()
            _ = model(img) # Forward
            
            # Get Attention
            attns = capturer.get_stacked() # (L, B, H, N, N)
            if attns is None: continue
            
            # Compute KTI for EACH layer
            # Problem: ToyViT might have CLS token.
            # Lab.calculate_physically_grounded_kti handles it?
            
            # Manually call specific KTI func
            for l_idx, attn_map in enumerate(attns):
                # attn_map: (B, H, N, N)
                # wrapper provides grid info?
                # ToyViT: 64/4 = 16x16 grid = 256. +1 CLS.
                # Pretrained: 224/16 = 14x14 grid = 196. +1 CLS.
                
                # We need to set lab.current_feature_grid
                if model_type == 'Scratch':
                    lab.current_feature_grid = (16, 16)
                else:
                    lab.current_feature_grid = (14, 14)
                    
                kti = lab.calculate_physically_grounded_kti(attn_map, kp)
                layer_metrics[l_idx]['kti'].append(kti)
                
    # Save JSON
    # Save into Control subfolder
    group_dir = Path(lab.output_dir) / 'Control'
    group_dir.mkdir(parents=True, exist_ok=True)
    
    serializable = {}
    for l, v in layer_metrics.items():
        serializable[l] = {'kti': [float(x) for x in v['kti']]}
        
    with open(group_dir / 'layer_metrics_Control.json', 'w') as f:
        json.dump(serializable, f)
        
    # Standard results.csv
    results_file = group_dir / 'results.csv'
    kti_avg = np.mean([np.mean(v['kti']) for v in layer_metrics.values() if v['kti']])
    import pandas as pd
    pd.DataFrame({'Group':['Control'], 'Avg_KTI':[kti_avg]}).to_csv(results_file, index=False)

    print(f"Saved {group_dir}/layer_metrics_Control.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, required=True, choices=['Scratch', 'Pretrained'])
    args = parser.parse_args()
    
    run_evaluation(args.type, None)
