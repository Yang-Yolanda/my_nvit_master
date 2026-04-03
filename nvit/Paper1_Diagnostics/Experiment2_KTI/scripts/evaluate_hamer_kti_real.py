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
import cv2

# Add paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
hamer_path = os.path.join(base_path, 'external_models/hamer')
sys.path.append(hamer_path)
sys.path.append(os.path.join(base_path, 'Paper1_Diagnostics/diagnostic_core'))

try:
    from hamer.models import HAMER, load_hamer
    from hamer.utils import recursive_to
    from diagnostic_engine import ViTDiagnosticLab, ModelWrapper
    # Import our new loader
    from nvit.Paper1_Diagnostics.Experiment2_KTI.freihand_loader import FreiHANDDataset
except ImportError as e:
    print(f"Import Error: {e}")
    # sys.exit(1)

class HaMeRWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        
    def get_backbone(self):
        return self.model.backbone
        
    def forward(self, batch):
        # batch is {'img': ...} or list of dicts?
        # Dataset returns single sample dict. 
        # We need to collate if batching, but we run BS=1 for simplicity here.
        if isinstance(batch, list):
             # Simple collate
             img = torch.stack([b['img'] for b in batch])
             batch = {'img': img}
        elif 'img' in batch and len(batch['img'].shape) == 3:
             # Add batch dim
             batch['img'] = batch['img'].unsqueeze(0)
             
        return self.model(batch)

class AttentionCapture:
    def __init__(self, model):
        self.attentions = []
        self.hooks = []
        for i, blk in enumerate(model.backbone.blocks):
            handle = blk.attn.attn_drop.register_forward_hook(self._hook_fn)
            self.hooks.append(handle)
            
    def _hook_fn(self, module, input, output):
        self.attentions.append(output.detach().cpu())
        
    def clear(self):
        self.attentions = []
        
    def get_stacked(self):
        if not self.attentions: return None
        return torch.stack(self.attentions, dim=0)

def run_evaluation(data_root, checkpoint_path=None, use_random=False, num_samples=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Validating HaMeR KTI on FreiHAND (Simulated/Real)...")
    
    # 1. Setup Model
    model = None
    model_type = "HaMeR_Random" if use_random else "HaMeR_Trained"
    
    if use_random or not checkpoint_path:
        print("Using Randomly Initialized HaMeR (Control)")
        from hamer.configs import default_config
        from yacs.config import CfgNode as CN
        model_cfg = default_config()
        model_cfg.defrost()
        
        # Manually add missing nodes
        if 'BACKBONE' not in model_cfg.MODEL:
            model_cfg.MODEL.BACKBONE = CN()
            
        model_cfg.MODEL.BACKBONE.TYPE = 'vit'
        model_cfg.MODEL.IMAGE_SIZE = 256
        model_cfg.MODEL.BBOX_SHAPE = [192, 256] 
        # Dummy MANO
        if 'MANO' not in model_cfg: model_cfg.MANO = CN()
        model_cfg.MANO.MODEL_PATH = 'dummy.pkl'
        model_cfg.MANO.MEAN_PARAMS = 'dummy_mean.pkl'
        model_cfg.freeze()
        
        try:
             model = HAMER(model_cfg, init_renderer=False)
        except Exception:
             print("Falling back to DummyHAMER")
             from hamer.models.backbones import create_backbone
             class DummyHAMER(torch.nn.Module):
                def __init__(self, cfg):
                    super().__init__()
                    self.backbone = create_backbone(cfg)
                def forward(self, batch):
                    x = batch['img']
                    if x.shape[3] == 256: obs_x = x[:,:,:,32:-32]
                    else: obs_x = x
                    feats = self.backbone(obs_x)
                    return {}
             model = DummyHAMER(model_cfg)
    else:
        print(f"Loading Checkpoint: {checkpoint_path}")
        model, model_cfg = load_hamer(checkpoint_path)
    
    model.to(device)
    model.eval()
    
    # 2. Setup Dataset
    # Check if download finished by looking for folder
    # Assuming user unzipped to 'nvit/external_models/hamer/datasets/freihand'
    dataset = FreiHANDDataset(data_root, split='evaluation')
    print(f"Dataset Size: {len(dataset)}")
    
    # 3. Setup Lab
    # Topology: OpenPose Hand (21 joints)
    parents = [-1,  0,  1,  2,  3,  0,  5,  6,  7,  0,  9, 10, 11,  0, 13, 14, 15,  0, 17, 18, 19]
    wrapper = HaMeRWrapper(model)
    lab = ViTDiagnosticLab(wrapper, model_name=f"{model_type}_FreiHAND", output_root='nvit/Paper1_Diagnostics/Experiment2_KTI/results')
    lab.parents = parents
    lab.current_feature_grid = (16, 12) # HaMeR standard 256x192
    
    capturer = AttentionCapture(model)
    layer_metrics = defaultdict(lambda: {'kti': []})
    
    # 4. Evaluation Loop
    count = 0
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            if count >= num_samples: break
            
            try:
                sample = dataset[i]
            except Exception as e:
                print(f"Load Error: {e}")
                continue
                
            img_tensor = sample['img'].to(device).unsqueeze(0) # (1, 3, 256, 256)
            gt_kp_2d = sample['keypoints_2d'].to(device).unsqueeze(0) # (1, 21, 2)
            
            capturer.clear()
            _ = model({'img': img_tensor})
            attns = capturer.get_stacked()
            
            if attns is None: continue
            
            for l_idx, attn_map in enumerate(attns):
                kti = lab.calculate_physically_grounded_kti(attn_map, gt_kp_2d)
                layer_metrics[l_idx]['kti'].append(kti)
            
            count += 1
            
    # Save
    output_dir = Path(f'nvit/Paper1_Diagnostics/Experiment2_KTI/results/{model_type}_FreiHAND')
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {l: {'kti': [float(x) for x in v['kti']]} for l, v in layer_metrics.items()}
    with open(output_dir / 'layer_metrics_Control.json', 'w') as f:
        json.dump(serializable, f)
    print(f"Finished. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='nvit/external_models/hamer/datasets/freihand')
    # Use random by default for now until we have weights
    parser.add_argument('--random', action='store_true', default=True) 
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    run_evaluation(args.data_root, args.checkpoint, args.random)
