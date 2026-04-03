#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_experimental_groups(total_layers=32):
    """
    Generate the standard 14 experimental groups used in Paper 1.
    """
    groups = {
        'Control': {'mask_layers': [], 'mode': 'none'},
    }
    
    # --- Tier 1: Depth Sweep (Fixed Mode: Hard) ---
    # Purpose: Titrate the optimal transition point vertically
    depth_points = [0.25, 0.33, 0.50, 0.66, 0.75]
    for p in depth_points:
        split_layer = int(total_layers * p)
        split_layer = max(1, split_layer)
        label = int(p * 100)
        groups[f'Tier1-Depth-{label}%'] = {
            'mask_layers': list(range(split_layer, total_layers)),
            'mode': 'hard',
            'desc': f'Static split at depth {label}%'
        }

    # --- Tier 2: Logic Sweep (Fixed Depth: 75% or 50% - Selection based on Tier 1) ---
    # Purpose: Compare Encoding strategies at a high-performing depth
    sweep_depth = int(total_layers * 0.75) # Defaulting to 75% for sweep
    groups['Tier2-Soft-75%'] = {
        'mask_layers': list(range(sweep_depth, total_layers)),
        'mode': 'soft',
        'desc': 'Soft masking (locality proxy) at back 1/4'
    }
    groups['Tier2-Hybrid-75%'] = {
        'mask_layers': list(range(sweep_depth, total_layers)),
        'mode': 'hybrid',
        'layer_modes': {i: 'soft' if i < (sweep_depth + (total_layers-sweep_depth)//2) else 'hard' for i in range(sweep_depth, total_layers)},
        'desc': 'Hybrid soft-then-hard at back 1/4'
    }
    groups['Tier2-Sandwich-75%'] = {
        'mask_layers': list(range(sweep_depth, total_layers)),
        'mode': 'hybrid',
        'layer_modes': {i: 'soft' if i < (total_layers - 4) else 'hard' for i in range(sweep_depth, total_layers)},
        'desc': 'Sandwich: Primary Soft, Final 4 Hard'
    }

    # --- Tier 3: Architecture Sweep (Paper 2) ---
    # Purpose: Evaluate different Hybrid Combinations (Mamba Variant x GCN Variant)
    groups['Tier3-Spiral-Grid'] = {
        'mode': 'architecture',
        'mamba_variant': 'spiral',
        'gcn_variant': 'grid',
        'desc': 'Round 9 Baseline: Spiral Mamba + Grid GCN'
    }
    groups['Tier3-Spiral-Skeleton'] = {
        'mode': 'architecture',
        'mamba_variant': 'spiral',
        'gcn_variant': 'skeleton',
        'desc': 'Physiological Prior: Spiral Mamba + Skeleton GCN'
    }
    groups['Tier3-ViT-Only'] = {
        'mode': 'architecture',
        'mamba_variant': 'none',
        'gcn_variant': 'none',
        'sl1': 12, 'sl2': 12,
        'desc': 'Baseline: Pure ViT'
    }
    groups['Tier3-Mamba-Only'] = {
        'mode': 'architecture',
        'mamba_variant': 'spiral',
        'gcn_variant': 'guided', # 'guided' in GCN means don't map to joints if sl2=12
        'sl1': 8, 'sl2': 12,
        'desc': 'Ablation: ViT + Mamba'
    }
    groups['Tier3-GCN-Only'] = {
        'mode': 'architecture',
        'mamba_variant': 'none',
        'gcn_variant': 'grid',
        'sl1': 8, 'sl2': 8,
        'desc': 'Ablation: ViT + GCN'
    }

    return groups

    # Removed Redundant Static Hybrids in favor of Phased Tiers above
    return groups

class MaskingPatcher:
    GLOBAL_INSTANCE = None
    """
    Applied patches to HMR2/ViT model to inject Attention Masking during training using PyTorch Hooks.
    """
    def __init__(self, model, mask_config):
        MaskingPatcher.GLOBAL_INSTANCE = self
        self.model = model
        self.mask_config = mask_config
        self.dist_matrix_cache = None
        self.dist_matrix_device = None
        
        self.last_joints2d = None
        self.skel_D = self._build_smpl_skeleton_distance()
        
        # Detect ViT layers
        self.att_modules = []
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'blocks'):
            for i, blk in enumerate(model.backbone.blocks):
                if hasattr(blk, 'attn'):
                    self.att_modules.append((i, blk.attn))
        
        logger.info(f"MaskingPatcher: Found {len(self.att_modules)} Attention layers.")

    def set_joints2d(self, joints2d_xy):
        """Update the 2D joints for the current batch (B, K, 2)"""
        if joints2d_xy is not None:
            self.last_joints2d = joints2d_xy.detach()

    def _build_smpl_skeleton_distance(self):
        # Common SMPL edges (for 24 joints)
        edges = [
            (0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9),
            (7,10), (8,11), (9,12), (9,13), (9,14), (12,15), (13,16), (14,17),
            (16,18), (17,19), (18,20), (19,21), (20,22), (21,23)
        ]
        K = 24
        dist = torch.full((K, K), float('inf'))
        for i in range(K):
            dist[i, i] = 0
        for u, v in edges:
            dist[u, v] = 1
            dist[v, u] = 1
            
        for k in range(K):
            for i in range(K):
                for j in range(K):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        return dist

    def apply(self):
        """Apply the patch to all attention modules."""
        count = 0
        for i, attn_module in self.att_modules:
            # Check if this layer is targeted by config
            mask_type = self.mask_config.get('mode', 'none')
            mask_domain = self.mask_config.get('domain', 'locality')
            
            # Determine specific mode for this layer
            current_mode = 'none'
            if i in self.mask_config.get('mask_layers', []):
                 if mask_type == 'hybrid' or mask_type == 'adaptive':
                     current_mode = self.mask_config.get('layer_modes', {}).get(i, 'none')
                 else:
                     current_mode = mask_type

            if current_mode == 'none':
                continue

            # Original Forward Stash
            if not hasattr(attn_module, '_original_forward'):
                attn_module._original_forward = attn_module.forward
            
            # Define Closure
            def make_forward(idx, mod, mode, domain):
                def forward(x):
                    B, N, C = x.shape
                    qkv = mod.qkv(x).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]

                    attn_r = (q @ k.transpose(-2, -1)) * mod.scale
                    
                    # --- MASK INJECTION START ---
                    import math
                    grid_w = int(math.sqrt(N))
                    if grid_w*grid_w != N and grid_w*grid_w != N-1: grid_w = 14
                    grid_h = grid_w
                    
                    if domain == 'locality':
                        if (self.dist_matrix_cache is None or 
                            self.dist_matrix_device != x.device or 
                            self.dist_matrix_cache.shape[-1] != N):
                            if grid_h * grid_w == N:
                                 y, x_grid = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
                                 coords = torch.stack([x_grid.flatten(), y.flatten()], dim=1).float()
                            elif grid_h * grid_w == N - 1:
                                 y, x_grid = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
                                 coords = torch.stack([x_grid.flatten(), y.flatten()], dim=1).float()
                                 cls_coord = torch.tensor([[-100., -100.]])
                                 coords = torch.cat([cls_coord, coords], dim=0)
                            else:
                                 coords = torch.arange(N).unsqueeze(1).float()
                            dist = torch.cdist(coords, coords)
                            self.dist_matrix_cache = dist.to(x.device)
                            self.dist_matrix_device = x.device
                        
                        dist = self.dist_matrix_cache
                        mask = torch.zeros_like(dist)
                        if mode == 'hard':
                            mask[dist > 3.5] = float('-inf')
                        elif mode == 'soft':
                            mask = - (dist**2) / (2 * 10.0)
                        
                        attn_r = attn_r + mask.unsqueeze(0).unsqueeze(0)
                        
                    elif domain == 'skeleton':
                        if self.last_joints2d is None:
                            raise RuntimeError("skeleton mask domain requires self.last_joints2d. Please call set_joints2d() in forward pass.")
                            
                        # Build Token->Joint Assignment Matrix M
                        # 1. Patch Coordinates
                        y, x_grid = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
                        patch_xy = torch.stack([x_grid.flatten(), y.flatten()], dim=1).float().to(x.device) # (N_patches, 2)
                        
                        # Handle CLS token if present
                        if grid_h * grid_w == N - 1:
                            cls_coord = torch.tensor([[-1000., -1000.]], device=x.device)
                            patch_xy = torch.cat([cls_coord, patch_xy], dim=0) # (N, 2)
                        
                        # 2. Convert Joints from orig pixel (0-256) to Grid space (0-W)
                        u_px = self.last_joints2d # (B, K, 2)
                        # Assume input is 256 crop
                        u_grid = u_px / (256.0 / grid_w) 
                        
                        # Compute M proportional to exp(-dist^2 / 2*t2j_sigma^2)
                        t2j_sigma = self.mask_config.get('t2j_sigma_px', 2.0)
                        p = patch_xy.unsqueeze(0).unsqueeze(1) # (1, 1, N, 2)
                        u = u_grid.unsqueeze(2) # (B, K, 1, 2)
                        dist_sq = ((p - u)**2).sum(dim=-1) # (B, K, N)
                        
                        M = torch.exp(-dist_sq / (2 * t2j_sigma**2)) # (B, K, N)
                        # Normalize over joints K
                        M = M / (M.sum(dim=1, keepdim=True) + 1e-6)
                        
                        # Expand skeleton distance matrix
                        self.skel_D = self.skel_D.to(x.device)
                        D_skel = self.skel_D # (K, K)
                        
                        skel_sigma = self.mask_config.get('skel_sigma', 1.5)
                        hard_hops = self.mask_config.get('hard_hops', 1)
                        
                        if mode == 'hard' or mode == 'hard_1hop_only':
                            A_joint = torch.zeros_like(D_skel)
                            A_joint[D_skel > hard_hops] = -1e4
                            A_tok = torch.einsum('bkn,kl,blm->bnm', M, A_joint, M)
                            mask = A_tok
                        elif mode == 'soft' or mode == 'soft_distance_only':
                            B_joint = -(D_skel**2) / (2 * skel_sigma**2)
                            B_tok = torch.einsum('bkn,kl,blm->bnm', M, B_joint, M)
                            mask = B_tok
                        else:
                            mask = torch.zeros((B, N, N), device=x.device)
                            
                        # Add CLS protection if needed (allow CLS to attend everywhere)
                        if grid_h * grid_w == N - 1:
                            mask[:, 0, :] = 0
                            mask[:, :, 0] = 0
                            
                        attn_r = attn_r + mask.unsqueeze(1) # Broadcast over heads (B, H, N, N)
                        
                    # --- MASK INJECTION END ---
                    
                    attn = attn_r.softmax(dim=-1)
                    attn = mod.attn_drop(attn)
                    
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = mod.proj(x)
                    x = mod.proj_drop(x)
                    return x
                return forward

            # Apply
            attn_module.forward = make_forward(i, attn_module, current_mode, mask_domain)
            count += 1
            
        logger.info(f"MaskingPatcher: Applied masks to {count} layers using config.") 

