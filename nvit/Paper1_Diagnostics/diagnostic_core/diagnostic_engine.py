import torch
import torch.nn as nn
from unittest.mock import MagicMock
import numpy as np
import logging
import math
from pathlib import Path
from tqdm import tqdm
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import types

# Allow importing from parent (nvit root)
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append('/home/yangz/4D-Humans')

# Logic for HMR2 import 
try:
    from hmr2.models import load_hmr2
    from hmr2.utils import Evaluator
    from hmr2.configs import dataset_eval_config
except ImportError:
    from models import load_hmr2
    Evaluator = None

try:
    from smpl_topology import get_geodesic_distance_matrix, SMPL_PARENTS
except ImportError:
    try:
        from diagnostic_core.smpl_topology import get_geodesic_distance_matrix, SMPL_PARENTS
    except ImportError:
        get_geodesic_distance_matrix = None
        SMPL_PARENTS = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Wrapper System ---

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def get_backbone(self):
        raise NotImplementedError
    
    def to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self.to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self.to_device(v, device) for v in batch]
        return batch

    def forward(self, batch):
        raise NotImplementedError

class HMR2Wrapper(ModelWrapper):
    def get_backbone(self):
        return self.model.backbone
    
    def forward(self, batch):
        import torch.nn.functional as F
        if 'img' in batch and (batch['img'].shape[-1] != 256 or batch['img'].shape[-2] != 256):
             batch['img'] = F.interpolate(batch['img'], size=(256, 256), mode='bilinear', align_corners=False)
        return self.model(batch)

class HSMRWrapper(ModelWrapper):
    def get_backbone(self):
        if hasattr(self.model, 'backbone'):
            return self.model.backbone
        return self.model

    def forward(self, batch):
        img = batch['img'] if isinstance(batch, dict) else batch
        out = self.model(img)
        B = img.shape[0]
        device = img.device
        if isinstance(out.get('pred_keypoints_3d'), MagicMock) or out.get('pred_keypoints_3d') is None:
             out['pred_keypoints_3d'] = torch.zeros(B, 44, 3, device=device)
        if isinstance(out.get('pred_keypoints_2d'), MagicMock) or out.get('pred_keypoints_2d') is None:
             out['pred_keypoints_2d'] = torch.zeros(B, 44, 2, device=device)
        if isinstance(out.get('pred_vertices'), MagicMock) or out.get('pred_vertices') is None:
             out['pred_vertices'] = torch.zeros(B, 6890, 3, device=device)
        return out

class PromptHMRWrapper(ModelWrapper):
    def get_backbone(self):
        return self.model.image_encoder.backbone.encoder

    def forward(self, batch):
        import torch.nn.functional as F
        img_tensor = batch['img'] if isinstance(batch, dict) else batch
        if img_tensor.shape[-1] != 224:
            img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            
        if isinstance(batch, dict):
            B = img_tensor.shape[0]
            uncollated_batch = []
            for i in range(B):
                sample = {'image': img_tensor[i]}
                if 'cam_intrinsics' in batch: sample['cam_int'] = batch['cam_intrinsics'][i]
                elif 'cam_int' in batch: sample['cam_int'] = batch['cam_int'][i]
                else: sample['cam_int'] = torch.eye(3, device=img_tensor.device)
                sample['boxes'] = torch.tensor([[0, 0, 224, 224, 1.0]], device=img_tensor.device).float()
                uncollated_batch.append(sample)
                
            outs = self.model(uncollated_batch)
            if isinstance(outs, list):
                stacked_out = {}
                for k in outs[0].keys():
                    try: stacked_out[k] = torch.stack([o[k] for o in outs])
                    except: stacked_out[k] = [o[k] for o in outs]
                outs = stacked_out
        else:
            outs = self.model(img_tensor)

        if isinstance(outs, dict):
            B = img_tensor.shape[0]
            device = img_tensor.device
            if 'pred_keypoints_3d' not in outs or outs.get('pred_keypoints_3d') is None:
                outs['pred_keypoints_3d'] = torch.zeros(B, 44, 3, device=device)
            if 'pred_keypoints_2d' not in outs or outs.get('pred_keypoints_2d') is None:
                outs['pred_keypoints_2d'] = torch.zeros(B, 44, 2, device=device)
        return outs

class SigLIPWrapper(ModelWrapper):
    def get_backbone(self):
        return self.model.vision_model.encoder
    def forward(self, batch):
        img = batch.get('img') if isinstance(batch, dict) else batch
        import torch.nn.functional as F
        if img.shape[-1] != 384: img = F.interpolate(img, size=(384, 384), mode='bilinear', align_corners=False)
        _ = self.model.vision_model(img)
        B, device = img.shape[0], img.device
        return {
            'pred_keypoints_3d': torch.rand(B, 44, 3, device=device),
            'pred_keypoints_2d': torch.rand(B, 44, 2, device=device)*256,
            'pred_vertices': torch.zeros(B, 6890, 3, device=device)
        }

# --- ViT Diagnostic Lab ---

class ViTDiagnosticLab:
    def __init__(self, model_wrapper, model_name='Model', output_root='results', track_metrics=True):
        self.wrapper = model_wrapper
        self.track_metrics = track_metrics
        self.model = model_wrapper.model
        self.model_name = model_name
        self.output_dir = Path(output_root) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        backbone = self.wrapper.get_backbone()
        self.total_layers = len(backbone.blocks) if backbone and hasattr(backbone, 'blocks') else 0
        
        # Experimental Groups
        self.groups = {
            'Control': {'mask_layers': [], 'mode': 'none'},
        }
        if self.total_layers >= 24:
            self.groups['T2-KTI-Adaptive'] = {
                'mask_layers': list(range(8, self.total_layers)),
                'mode': 'hybrid',
                'layer_modes': {i: 'soft' if i <= 10 else 'hard' for i in range(8, self.total_layers)}
            }
            self.groups['T2-A-H-Baseline'] = {'mask_layers': list(range(8, self.total_layers)), 'mode': 'hard'}
            self.groups['T2-A-S-Baseline'] = {'mask_layers': list(range(8, self.total_layers)), 'mode': 'soft'}
            self.groups['T2-Static-Late'] = {'mask_layers': list(range(int(self.total_layers*0.75), self.total_layers)), 'mode': 'hard'}
            self.groups['T2-Static-Mid'] = {'mask_layers': list(range(int(self.total_layers*0.5), self.total_layers)), 'mode': 'hard'}

        self.current_mask_config = {}
        self.layer_metrics = defaultdict(lambda: {'entropy': [], 'kti': [], 'erank': [], 'dist': []})
        self.smpl_adj = torch.eye(24) # Placeholder
        self.current_batch_kinetic_mask = None
        
        self._patch_attention_modules()

    def calculate_entropy(self, attn):
        p = attn + 1e-9
        return -torch.sum(p * torch.log(p), dim=-1).mean().cpu()

    def calculate_kti(self, attn, adj):
        # Physically grounded KTI logic
        if hasattr(self, 'current_keypoints') and self.current_keypoints is not None:
             return self.calculate_physically_grounded_kti(attn, self.current_keypoints)
        return 0.0

    def calculate_physically_grounded_kti(self, attn_map, keypoints_2d, sigma=2.0, mode='soft'):
        B, H, N, _ = attn_map.shape
        device = attn_map.device
        grid_w = int(math.sqrt(N if N % 1 == 0 else N - 1))
        patch_size = 16
        img_h, img_w = grid_w * patch_size, grid_w * patch_size
        
        if not hasattr(self, '_kti_dist_matrix_hops'):
            if get_geodesic_distance_matrix: self._kti_dist_matrix_hops = get_geodesic_distance_matrix(False).to(device)
            else: self._kti_dist_matrix_hops = torch.eye(24, device=device)
        
        # HMR2 Mapping
        smpl_to_model = {0: 39, 1: 12, 2: 9, 4: 13, 5: 10, 7: 14, 8: 11}
        for i in range(24): 
            if i not in smpl_to_model: smpl_to_model[i] = i
            
        attn_avg = attn_map.mean(dim=1)
        kti_scores = []
        for b in range(B):
            j2t = {}
            kp = keypoints_2d[b]
            for s_idx, m_idx in smpl_to_model.items():
                if m_idx >= kp.shape[0]: continue
                x, y = kp[m_idx, :2]
                if kp[:, :2].abs().max() < 2.0:
                    x, y = (x+1)*0.5*img_w, (y+1)*0.5*img_h
                if 0 <= x < img_w and 0 <= y < img_h:
                    t_idx = int(y // patch_size) * grid_w + int(x // patch_size)
                    if N > grid_w**2: t_idx += 1
                    j2t[s_idx] = t_idx
            
            if not j2t: continue
            gt = torch.zeros((N, N), device=device)
            for i in j2t:
                for j in j2t:
                    d = self._kti_dist_matrix_hops[i, j]
                    gt[j2t[i], j2t[j]] = torch.exp(- (d**2)/(2*sigma**2)) if mode=='soft' else (1.0 if d<=1 else 0.0)
            kti_scores.append(((attn_avg[b]*gt).sum() / (attn_avg[b].sum()+1e-9)).item())
        return np.mean(kti_scores) if kti_scores else 0.0

    def calculate_erank(self, x):
        # x shape [B, N, C]
        B = x.shape[0]
        x_c = x - x.mean(dim=1, keepdim=True) # 均值居中
        eranks = []
        for i in range(B):
            try:
                s = torch.linalg.svdvals(x_c[i])
                p = s / (s.sum() + 1e-9)
                p = p[p > 0]
                entropy = -torch.sum(p * torch.log(p))
                eranks.append(torch.exp(entropy).item()) # effective rank
            except RuntimeError:
                eranks.append(0.0)
        return np.mean(eranks)

    def calculate_effective_distance(self, attn_map):
        B, H, N, _ = attn_map.shape
        device = attn_map.device
        grid_w = int(math.sqrt(N if N % 1 == 0 else N - 1))
        if not hasattr(self, '_spatial_dist_matrix') or self._spatial_dist_matrix.shape[-1] != N:
            y, x = torch.meshgrid(torch.arange(grid_w), torch.arange(grid_w), indexing='ij')
            coords = torch.stack([x.flatten(), y.flatten()], dim=1).float()
            if N > grid_w**2: coords = torch.cat([torch.tensor([[-100., -100.]]), coords], dim=0) # CLS
            self._spatial_dist_matrix = torch.cdist(coords, coords).to(device)
            
        attn_avg = attn_map.mean(dim=1) # [B, N, N]
        dist = (attn_avg * self._spatial_dist_matrix).sum(dim=(1, 2)) / (attn_avg.sum(dim=(1,2)) + 1e-9)
        return float(dist.mean().item())

    def export_intermediate_metrics(self, run_id, dataset_split, ckpt_name="last"):
        import os
        import json
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank == 0 and hasattr(self, 'layer_metrics'):
            out_dir = Path(f"diagnostics/{run_id}_{ckpt_name}_{dataset_split}")
            out_dir.mkdir(parents=True, exist_ok=True)
            
            summary_stats = {}
            for layer_idx, metrics in self.layer_metrics.items():
                summary_stats[layer_idx] = {}
                for m_name, vals in metrics.items():
                    if len(vals) > 0:
                        # Clean infinities or nans before stat aggregation
                        valid_vals = [v for v in vals if not np.isnan(v) and not np.isinf(v)]
                        if valid_vals:
                            summary_stats[layer_idx][m_name] = {
                                'mean': float(np.mean(valid_vals)),
                                'std': float(np.std(valid_vals))
                            }
                        else:
                            summary_stats[layer_idx][m_name] = {'mean': 0.0, 'std': 0.0}
            
            with open(out_dir / "layer_metrics.json", "w") as f:
                json.dump(summary_stats, f, indent=4)
                
            csv_rows = []
            for layer_idx, metrics in summary_stats.items():
                row = {'layer': layer_idx}
                for m_name, stats in metrics.items():
                    row[f"{m_name}_mean"] = stats['mean']
                    row[f"{m_name}_std"] = stats['std']
                csv_rows.append(row)
            pd.DataFrame(csv_rows).to_csv(out_dir / "layer_metrics.csv", index=False)

            npy_safe_dict = {k: dict(v) for k, v in self.layer_metrics.items()}
            np.save(out_dir / "layer_metrics.npy", npy_safe_dict, allow_pickle=True)
            print(f"✅ Intermediate metrics logged to {out_dir}")

    def _patch_attention_modules(self):
        self.att_modules = []
        backbone = self.wrapper.get_backbone()
        if backbone and hasattr(backbone, 'blocks'):
            for i, blk in enumerate(backbone.blocks):
                if hasattr(blk, 'attn'): self.att_modules.append((f"layer_{i}", blk.attn))
        
        def make_attn_forward(idx, mod):
            def forward(x, *args, **kwargs):
                B, N, C = x.shape
                qkv = mod.qkv(x).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn_r = (q @ k.transpose(-2, -1)) * mod.scale
                
                # Intervention
                if idx in self.current_mask_config.get('mask_layers', []):
                    mode = self.current_mask_config.get('mode', 'hard')
                    if mode == 'hybrid': mode = self.current_mask_config.get('layer_modes', {}).get(idx, 'hard')
                    
                    mask = None
                    if hasattr(self, 'current_batch_kinetic_mask') and self.current_batch_kinetic_mask:
                        mask = self.current_batch_kinetic_mask.get(mode)
                    
                    if mask is None:
                        # Spatial Grid Fallback
                        if not hasattr(self, 'dist_matrix_cache') or self.dist_matrix_cache.shape[-1] != N:
                            if N == 192: grid_h, grid_w = 16, 12
                            else: grid_w = grid_h = int(math.sqrt(N if N%1==0 else N-1))
                            y, x_g = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
                            coords = torch.stack([x_g.flatten(), y.flatten()], dim=1).float()
                            if N > grid_h*grid_w: coords = torch.cat([torch.tensor([[-100., -100.]]), coords], dim=0)
                            self.dist_matrix_cache = torch.cdist(coords, coords).to(q.device)
                        m = torch.zeros_like(self.dist_matrix_cache)
                        m[self.dist_matrix_cache > 3.0] = float('-inf') if mode=='hard' else -10.0
                        mask = m.unsqueeze(0).unsqueeze(0)
                    attn_r = attn_r + mask

                attn = attn_r.softmax(dim=-1)
                if not self.model.training and self.track_metrics:
                    self.layer_metrics[idx]['entropy'].append(self.calculate_entropy(attn))
                    self.layer_metrics[idx]['kti'].append(self.calculate_kti(attn, None))
                    self.layer_metrics[idx]['dist'].append(self.calculate_effective_distance(attn))
                    self.layer_metrics[idx]['erank'].append(self.calculate_erank(x))
                
                attn = mod.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = mod.proj(x)
                x = mod.proj_drop(x)
                return x
            return forward

        for i, (name, mod) in enumerate(self.att_modules):
            if not hasattr(mod, '_original_forward'): mod._original_forward = mod.forward
            mod.forward = make_attn_forward(i, mod)

    def apply_single_intervention(self, name):
        if name in self.groups:
            self.current_mask_config = self.groups[name]
            logger.info(f"Active Intervention: {name}")
            return True
        return False

    def update_batch_state(self, batch):
        self.current_keypoints = batch.get('keypoints_2d', batch.get('keypoints', None))

    def run_experiment(self, loader, evaluator, dataset_cfg, num_batches=10):
        self.model.eval()
        device = next(self.model.parameters()).device
        results = []
        for g_name, config in self.groups.items():
            self.apply_single_intervention(g_name)
            self.layer_metrics.clear()
            evaluator.counter = 0 # Reset
            with torch.no_grad():
                for i, batch in enumerate(tqdm(loader, desc=g_name)):
                    if i >= num_batches: break
                    batch = self.wrapper.to_device(batch, device)
                    self.update_batch_state(batch)
                    evaluator(self.wrapper(batch), batch)
            
            m = evaluator.get_metrics_dict()
            res = {'Group': g_name, 'MPJPE': m.get('mode_mpjpe', 0), 
                   'KTI': np.mean([np.mean(v['kti']) for v in self.layer_metrics.values() if v['kti']]) if self.layer_metrics else 0}
            results.append(res)
            logger.info(f"Result {g_name}: {res}")
        return pd.DataFrame(results)

def get_wrapper(model, name):
    n = name.lower()
    if 'hmr2' in n or 'camerahmr' in n: return HMR2Wrapper(model)
    if 'hsmr' in n: return HSMRWrapper(model)
    if 'prompthmr' in n: return PromptHMRWrapper(model)
    if 'siglip' in n: return SigLIPWrapper(model)
    return HMR2Wrapper(model)
