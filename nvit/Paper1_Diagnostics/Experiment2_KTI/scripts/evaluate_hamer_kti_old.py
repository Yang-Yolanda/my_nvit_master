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
# hamer package is inside external_models/hamer/
sys.path.append(hamer_path)
sys.path.append(os.path.join(base_path, 'Paper1_Diagnostics/diagnostic_core'))

try:
    from hamer.models import HAMER, load_hamer
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
    from diagnostic_engine import ViTDiagnosticLab, ModelWrapper
except ImportError as e:
    print(f"Import Error: {e}")
    # Try to install requirements?
    # sys.exit(1)

class HaMeRWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        
    def get_backbone(self):
        # HaMeR model.backbone is a ViT
        return self.model.backbone
        
    def forward(self, batch):
        # batch is {'img': ...}
        # HaMeR forward_step expects 'img' in batch
        # And it crops internaly to 32:-32?
        # Let's check hamer.py again.
        # Yes, line 105: self.backbone(x[:,:,:,32:-32])
        # So we just pass the batch as is, assuming 'img' is 256x256
        
        # However, we need to ensure input is normalized!
        # HaMeR demo.py normalizes before creating dataset?
        # Actually demo.py ViTDetDataset/utils does normalization usually.
        # Check demo.py line 212: it RE-normalizes for tensorboard logging?
        # It seems it expects normalized input.
        
        # If we pass raw tensor, we handles normalization in harness?
        # We will assume harness/loader provides correct input.
        
        # For KTI hooks to work, we just need to run the backbone
        # model(batch) calls forward_step(train=False)
        return self.model(batch)

class AttentionCapture:
    """Capture attention from HaMeR ViT"""
    def __init__(self, model):
        self.attentions = []
        self.hooks = []
        
        # HaMeR ViT blocks are in model.backbone.blocks
        for i, blk in enumerate(model.backbone.blocks):
            # Hook attn_drop (after softmax)
            handle = blk.attn.attn_drop.register_forward_hook(self._hook_fn)
            self.hooks.append(handle)
            
    def _hook_fn(self, module, input, output):
        self.attentions.append(output.detach().cpu())
        
    def clear(self):
        self.attentions = []
        
    def get_stacked(self):
        if not self.attentions: return None
        return torch.stack(self.attentions, dim=0)

def run_evaluation(image_folder, checkpoint_path=None, use_random=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluting HaMeR on {image_folder}...")
    
    model = None
    model_type = "HaMeR_Random"
    
    if use_random or not checkpoint_path or not os.path.exists(checkpoint_path):
        print("Using Randomly Initialized HaMeR (Control/Test Mode)")
        try:
            from hamer.configs import default_config
            from yacs.config import CfgNode as CN
            
            # Load default config
            model_cfg = default_config()
            model_cfg.defrost()
            
            # Manually add missing nodes
            if 'BACKBONE' not in model_cfg.MODEL:
                model_cfg.MODEL.BACKBONE = CN()
                
            model_cfg.MODEL.BACKBONE.TYPE = 'vit'
            model_cfg.MODEL.IMAGE_SIZE = 256
            model_cfg.MODEL.BBOX_SHAPE = [192, 256] 
            model_cfg.LOSS_WEIGHTS.ADVERSARIAL = 0
            
            # Setup MANO dummy if needed by HAMER
            if 'MANO' not in model_cfg:
                model_cfg.MANO = CN()
            model_cfg.MANO.MODEL_PATH = 'dummy.pkl'
            model_cfg.MANO.MEAN_PARAMS = 'dummy_mean.pkl'
            
            model_cfg.freeze()
            
            # Try initializing real HAMER
            # This is expected to fail if weights/files missing
            model = HAMER(model_cfg, init_renderer=False)
            
        except Exception as e:
            print(f"Failed to initialize real HAMER path (likely missing MANO): {e}")
            print("Falling back to DummyHAMER with Real ViT Backbone")
            
            # Mock HAMER
            from hamer.models.backbones import create_backbone
            class DummyHAMER(torch.nn.Module):
                def __init__(self, cfg):
                    super().__init__()
                    self.backbone = create_backbone(cfg)
                    self.cfg = cfg
                    
                def forward(self, batch):
                    # Run backbone to verify hooking
                    x = batch['img']
                    if x.shape[3] == 256:
                         obs_x = x[:,:,:,32:-32]
                    else:
                         obs_x = x
                    
                    feats = self.backbone(obs_x)
                    
                    # Return dummy output
                    B = x.shape[0]
                    # 21 joints, 2D
                    # KTI needs predicted coords to be in pixel space (matching 256x256 image)
                    pred_keypoints_2d = torch.rand(B, 21, 2, device=x.device) * 256
                    return {'pred_keypoints_2d': pred_keypoints_2d}
             
            model = DummyHAMER(model_cfg)
    else:
        print(f"Loading Checkpoint: {checkpoint_path}")
        model, model_cfg = load_hamer(checkpoint_path)
        model_type = "HaMeR_Trained"

    model.to(device)
    model.eval()
    
    # Define Wrapper
    # Topology: OpenPose Hand (21 joints)
    # 0: Wrist
    # 1-4: Thumb
    # 5-8: Index
    # 9-12: Middle
    # 13-16: Ring
    # 17-20: Pinky
    parents = [-1,  0,  1,  2,  3,  0,  5,  6,  7,  0,  9, 10, 11,  0, 13, 14, 15,  0, 17, 18, 19]
    
    wrapper = HaMeRWrapper(model)
    lab = ViTDiagnosticLab(wrapper, model_name=model_type, output_root='nvit/Paper1_Diagnostics/Experiment2_KTI/results')
    
    # Setup Lab for Hand
    lab.parents = parents
    # Override Grid for HaMeR (256x192 input to backbone -> 16x12 patches)
    # HaMeR code slices 32:-32 from 256 width.
    # Height: 256 / 16 = 16
    # Width: 192 / 16 = 12
    lab.current_feature_grid = (16, 12) 
    
    lab.groups = {'Control': {'mask_layers': [], 'mode': 'none'}}
    
    capturer = AttentionCapture(model)
    layer_metrics = defaultdict(lambda: {'kti': []})
    
    # Load Images
    import glob
    img_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')) + glob.glob(os.path.join(image_folder, '*.png')))
    if not img_paths:
        print("No images found. Exiting.")
        return

    print(f"Found {len(img_paths)} images.")
    
    # Transform
    from torchvision import transforms
    normalize = transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
    
    with torch.no_grad():
        for img_path in tqdm(img_paths):
            # Simple Processing: Resize/Center Crop to 256x256
            img_cv = cv2.imread(img_path)[:,:,::-1] # BGR->RGB
            H, W, _ = img_cv.shape
            
            # Center Crop
            S = min(H, W)
            start_x = (W - S) // 2
            start_y = (H - S) // 2
            img_crop = img_cv[start_y:start_y+S, start_x:start_x+S]
            img_resize = cv2.resize(img_crop, (256, 256))
            
            # To Tensor
            img_tensor = torch.from_numpy(img_resize.copy()).permute(2, 0, 1).float() / 255.0
            img_tensor = normalize(img_tensor).unsqueeze(0).to(device)
            
            batch = {'img': img_tensor}
            
            capturer.clear()
            out = model(batch)
            
            # Get Attention
            attns = capturer.get_stacked() # (L, B, H, N, N)
            
            if attns is None: continue
            
            # Get Keypoints (Prediction as Proxy for Ground Truth)
            # pred_keypoints_2d: (B, 21, 2)
            # It is normalized to pixel coords?
            # demo.py: perspective_projection outputs pixel coords.
            pred_kp = out['pred_keypoints_2d']
            
            # We align prediction to cropped space 256x256
            # pred_kp is in 256x256
            
            for l_idx, attn_map in enumerate(attns):
                # attn_map: (B, H, 192, 192)
                # Grid: (16, 12)
                
                # Check if attn n matches grid
                N = attn_map.shape[-1]
                if N != 16*12:
                     # Maybe CLS token?
                     # HaMeR Vit doesn't seem to have CLS based on code. 
                     pass
                
                kti = lab.calculate_physically_grounded_kti(attn_map, pred_kp)
                layer_metrics[l_idx]['kti'].append(kti)
                
    # Save
    output_dir = Path(f'nvit/Paper1_Diagnostics/Experiment2_KTI/results/{model_type}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    serializable = {}
    for l, v in layer_metrics.items():
        serializable[l] = {'kti': [float(x) for x in v['kti']]}
        
    with open(output_dir / 'layer_metrics_Control.json', 'w') as f:
        json.dump(serializable, f)
    print(f"Saved results to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--image_folder', type=str, default='nvit/external_models/hamer/example_data')
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()
    
    run_evaluation(args.image_folder, args.checkpoint, args.random)
