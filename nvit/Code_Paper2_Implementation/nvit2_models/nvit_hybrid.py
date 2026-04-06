
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dependency Imports
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("WARNING: mamba_ssm not found. MambaBlock will fail.")

from timm.models.vision_transformer import Block as TimmBlock
from timm.layers import Mlp, PatchEmbed

from .mamba_utils import PatchScanMamba, BidirectionalMambaBlock, MambaBlock, AnatomicalScanMamba
import sys
from pathlib import Path
# Add logic to find smpl_topology
sys.path.append('/home/yangz/NViT-master/nvit/Code_Paper2_Implementation')
from smpl_topology import get_smpl_adjacency_matrix


# ==============================================================================
# 1. Modular Blocks
# ==============================================================================

class ViTBlock(nn.Module):
    """ Wrapped Timm Block for Consistency """
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()
        self.block = TimmBlock(dim=dim, num_heads=num_heads, **kwargs)

    def forward(self, x):
        return self.block(x)

class GridGCNBlock(nn.Module):
    """ 
    Generic GCN Block.
    Can be used for Patch Grid or Joint Skeleton by providing custom adjacency.
    """
    def __init__(self, dim, num_nodes, adj=None, grid_size=None, has_cls=True):
        super().__init__()
        self.dim = dim
        self.num_nodes = num_nodes
        self.has_cls = has_cls
        
        if adj is not None:
            self.register_buffer('adj', adj)
        elif grid_size is not None:
            self.register_buffer('adj', self._build_grid_w_cls_adj(grid_size))
        else:
            # Default Identity if nothing provided
            self.register_buffer('adj', torch.eye(num_nodes + (1 if has_cls else 0)))
        
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        
    def _build_grid_w_cls_adj(self, grid_size):
        gs = grid_size if isinstance(grid_size, (tuple, list)) else (grid_size, grid_size)
        N = gs[0] * gs[1]
        total = N + (1 if self.has_cls else 0)
        adj = torch.eye(total)
        
        y, x = torch.meshgrid(torch.arange(gs[0]), torch.arange(gs[1]), indexing='ij')
        coords = torch.stack([x.flatten(), y.flatten()], dim=1).float()
        dist = torch.cdist(coords, coords)
        mask = (dist <= 1.5).float()
        
        offset = 1 if self.has_cls else 0
        adj[offset:, offset:] = mask
        if self.has_cls:
            adj[0, 1:] = 1.0
            adj[1:, 0] = 1.0
            
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return adj / deg

    def forward(self, x):
        res = x
        x = self.norm(x)
        ax = self.adj @ x 
        out = self.proj(self.act(self.fc(ax)))
        return res + out


# ==============================================================================
# 2. Main Backbone
# ==============================================================================


class HeatmapMapper(nn.Module):
    """
    Experimental Heatmap-Based Mapper.
    Maps Patches -> Joints using Explicit 2D Geometry Guidance.
    Steps:
    1. Predict Heatmaps (B, 24, 14, 14) from Patches (B, 196, C).
    2. Compute Spatial Softmax to get weights (B, 24, 196).
    3. Weighted Sum of Patches: Joints = Weights @ Patches.
    """
    def __init__(self, dim, num_joints=24, grid_size=(14, 14)):
        super().__init__()
        self.num_joints = num_joints
        self.H, self.W = grid_size
        
        # 3x3 Conv to capture local context (ViTPose style)
        self.heatmap_head = nn.Sequential(
            nn.LayerNorm(dim),
            # Reshape handled in forward features... wait, input is (B, 196, D)
            # We need to treat it as image for Conv2d
        )
        self.conv_head = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, num_joints, kernel_size=1)
        )
        
        # [Fix] Zero-initialize the last layer ensures uniform heatmaps at start.
        nn.init.zeros_(self.conv_head[-1].weight)
        nn.init.zeros_(self.conv_head[-1].bias)
        
    def forward(self, x_patches):
        """
        Args:
            x_patches: (B, N_patches, C). N=196. Grid order.
        Returns:
            x_joints: (B, 24, C). Aggregated features.
            heatmaps: (B, 24, 14, 14). For auxiliary loss.
        """
        B, N, C = x_patches.shape
        H, W = self.H, self.W
        
        # 1. Norm & Reshape
        # (B, N, C) -> (B, N, C)
        x_norm = self.heatmap_head(x_patches)
        # (B, N, C) -> (B, C, H, W)
        x_img = x_norm.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 2. Predict Heatmaps (Logits)
        # (B, C, H, W) -> (B, J, H, W)
        logits = self.conv_head(x_img) 
        
        # [Fix] Scale down logits to prevent gradient explosion
        # This reduces the magnitude of gradients flowing back through soft-argmax
        # [Fix] Stronger signal for soft-argmax convergence
        logits = logits
        
        # 2. Reshape for Loss/Softmax
        # (B, J, H, W)
        heatmaps_2d = logits
        
        # 3. Soft Argmax (Get Coords)
        B, J, H, W = logits.shape
        # Add clamping and epsilon for numerical stability (Fix for Step 0 NaN)
        logits_clamped = torch.clamp(logits, min=-11, max=11)
        logits_flat = logits_clamped.reshape(B, J, -1) 
        weights = F.softmax(logits_flat, dim=-1).reshape(B, J, H, W)
        weights = weights + 1e-8
        weights = weights / weights.sum(dim=(-1, -2), keepdim=True)
        
        # Create Grid
        device = logits.device
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        
        # Expected Coords (0..H-1)
        pred_x = (weights * xx).sum(dim=(-1, -2)) 
        pred_y = (weights * yy).sum(dim=(-1, -2)) 
        
        # 4. Feature Sampling (Grid Sample)
        # Normalize to [-1, 1]
        grid_x = (pred_x / max(W - 1, 1)) * 2 - 1
        grid_y = (pred_y / max(H - 1, 1)) * 2 - 1
        
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(1) # (B, 1, J, 2)
        
        sampled_features = F.grid_sample(x_img, grid, align_corners=True) # (B, C, 1, J)
        
        x_joints = sampled_features.squeeze(2).permute(0, 2, 1) # (B, J, C)
        
        # Inject Global Context (GAP of Patches)
        # x_patches: (B, 196, C)
        global_context = x_patches.mean(dim=1, keepdim=True) # (B, 1, C)
        
        # Add or Concat? 
        # Adding preserves dimension (compatible with existing heads)
        x_joints = x_joints + global_context
        
        return x_joints, heatmaps_2d

class AdaptiveNViT(nn.Module):
    """
    Universal Adaptive Backbone.
    Supports ViT, Mamba (Center-Out), and GCN (Grid or Join).
    """
    def __init__(
        self, 
        depth=32,
        embed_dim=1280,
        num_heads=16,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        switch_layer_1=2, 
        switch_layer_2=8, 
        img_size=(256, 192),
        patch_size=16,
        num_joints=24, # Configurable for SMPL, Robotic Arms, etc.
        mamba_variant='spiral', # [New] 'spiral' (Center-Out), 'bi', 'seq'
        gcn_variant='grid' # [New] 'grid', 'forward', 'inverse', 'bi', 'random'
    ):
        super().__init__()
        self.depth = depth
        self.switch_layer_1 = switch_layer_1 
        self.switch_layer_2 = switch_layer_2 
        self.embed_dim = embed_dim
        self.gcn_variant = gcn_variant 
        self.mamba_variant = mamba_variant
        
        # 1. Input Projector
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Mapper: Patches (ViT) -> Joints (Mamba/GCN)
        # [NEW] Use HeatmapMapper. grid_size is derived from patch_embed.
        grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.patch_to_joint = HeatmapMapper(dim=embed_dim, num_joints=num_joints, grid_size=grid_size)
        self.mapped = False # State flag
        
        
        # 2. Architecture Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList()
        
        for i in range(depth):
            if i < switch_layer_1:
                # Stage 1: Global Sight (ViT) [Layers 0-7]
                blk = ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                               qkv_bias=qkv_bias, proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
                tag = "ViT"
            elif i < switch_layer_2 or gcn_variant == 'guided':
                # Stage 2: Spatial Bio-Diffusion (Mamba) [Layers 8-9]
                # In 'guided' mode, we continue Mamba until end.
                # Stage 2: Spatial Bio-Diffusion (Mamba) [Layers 8-9]
                # Operates on Patches (Image Space)
                tag = f"SpatialMamba({mamba_variant})"
                if mamba_variant == 'spiral':
                    # Center-Out / Positive Kinematics
                    # PatchScanMamba handles (B, L, D) -> Spiral -> Mamba -> Unspiral -> (B, L, D)
                    blk = PatchScanMamba(dim=embed_dim, img_size=img_size, patch_size=patch_size, variant='spiral')
                elif mamba_variant == 'reverse_spiral':
                    blk = PatchScanMamba(dim=embed_dim, img_size=img_size, patch_size=patch_size, variant='reverse_spiral')
                elif mamba_variant == 'bi':
                    blk = BidirectionalMambaBlock(dim=embed_dim)
                elif mamba_variant.startswith('kinetic'):
                    # Kinetic Mamba (Anatomical Scan)
                    # Supports 'kinetic', 'kinetic_out', 'kinetic_in', 'kinetic_bi'
                    blk = AnatomicalScanMamba(dim=embed_dim, variant=mamba_variant)
                else:
                    blk = MambaBlock(dim=embed_dim)
            else:
                # Stage 3: Logical Refinement (GCN) [Layers 10-11]
                # Operated on JOINTS (after mapping)
                tag = f"GCN({gcn_variant})"
                
                # Determine Adjacency
                adj = None
                if gcn_variant == 'grid':
                    adj = torch.eye(num_joints + (1 if self.mapped else 0)) 
                elif gcn_variant == 'random':
                    adj = torch.rand((num_joints, num_joints))
                    adj = (adj > 0.5).float()
                elif gcn_variant == 'skeleton':
                     adj = get_smpl_adjacency_matrix(mode='undirected', add_self_loops=True)
                elif gcn_variant == 'directed_out': 
                     adj = get_smpl_adjacency_matrix(mode='outward', add_self_loops=True)
                elif gcn_variant == 'directed_in': 
                     adj = get_smpl_adjacency_matrix(mode='inward', add_self_loops=True)
                else:
                    adj = None
                
                blk = GridGCNBlock(dim=embed_dim, num_nodes=num_joints, adj=adj, has_cls=False)
                
            self.blocks.append(blk)
            logger.info(f"Layer {i}: {tag}")

        self.norm = nn.LayerNorm(embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        
        # Add CLS & Position
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Grid dimensions for reshaping
        Hp = self.patch_embed.img_size[0] // self.patch_embed.patch_size[0]
        Wp = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]
        
        heatmaps = None # Captured from Mapper
        
        map_layer = self.switch_layer_1 if self.mamba_variant.startswith('kinetic') else self.switch_layer_2
        
        for i, blk in enumerate(self.blocks):
            # [NEW] Inject Mapper at map_layer
            # This is where we switch from Patch Domain (ViT/SpatialMamba) to Joint Domain (Mamba/GCN)
            if i == map_layer and not self.mapped:
                # x: (B, 197, C) -> Remove CLS -> (B, 196, C)
                cls_t = x[:, 0:1, :]
                x_patches_raw = x[:, 1:, :] 
                
                # Apply Heatmap Mapper
                # Returns (B, 24, C), (B, 24, 14, 14)
                x_joints, heatmaps = self.patch_to_joint(x_patches_raw) 
                
                # [Topology C] Hybrid Context Strategy
                if "hybrid" in self.mamba_variant:
                    # Save patches for future concatenation
                    self.saved_patches = x_patches_raw
                    x = x_joints
                elif self.gcn_variant != 'guided' or self.mamba_variant.startswith('kinetic'):
                    x = x_joints # Replace with Joints for GCN or Kinematic Mamba
                else:
                    pass # Keep x as Patches for Transformer
                
                self.mapped = True
                
            x = blk(x)
            
        x = self.norm(x)
        
        # [NEW ARCHITECTURE] If map_layer >= depth (e.g., layers 8-10 are all Mamba), Heatmap Mapper was skipped!
        # Guarantee Heatmap processing occurs at the END of the backbone if not mapped yet.
        if not self.mapped:
            cls_t = x[:, 0:1, :]
            x_patches_raw = x[:, 1:, :]
            x_joints, heatmaps = self.patch_to_joint(x_patches_raw)
            self.mapped = True
            
            if "hybrid" in self.mamba_variant:
                self.saved_patches = x_patches_raw
                x = x_joints
            elif self.gcn_variant != 'guided' or self.mamba_variant.startswith('kinetic'):
                x = x_joints
                
        # [Topology C] Concatenation Logic
        if "hybrid" in self.mamba_variant and hasattr(self, 'saved_patches'):
            # x is (B, 24, C) refined joints
            # self.saved_patches is (B, 196, C) spatial evidence
            x_out = torch.cat([self.saved_patches, x], dim=1) 
            del self.saved_patches # Clean up
            self.mapped = False
            return x_out, heatmaps

        if self.gcn_variant == 'guided':
            # Soft Guide Mode: Return Context (Patches) + Guidance (Coords)
            # x is (B, 197, C). Remove CLS -> (B, 196, C)
            cls_t = x[:, 0:1, :]
            x_patches = x[:, 1:, :] 
            
            # Return tuple for specific handling in LightningModule
            self.mapped = False
            return x_patches, heatmaps
            
        elif self.mapped:
            # Output for HMR2 Head: (B, C, 24, 1)
            x_out = x.permute(0, 2, 1).unsqueeze(-1) # (B, C, 24, 1)
            
        else:
            # Fallback (never mapped?)
            x_patches = x[:, 1:, :] 
            B, N, C = x_patches.shape
            x_out = x_patches.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2).contiguous()
        
        # Reset state
        self.mapped = False
        
        return x_out, heatmaps

    def forward(self, x):
        x_out, _ = self.forward_features(x)
        return x_out

    def surgical_freeze(self, freeze_depth=0):
        """
        Freeze the first `freeze_depth` layers of the backbone (ViT stage).
        This is crucial for the "Hybrid-Sandwich" approach where we keep early visual features fixed.
        """
        # Freeze Projector
        if freeze_depth > 0:
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            self.pos_embed.requires_grad = False
            self.cls_token.requires_grad = False
            
        # Freeze Blocks explicitly
        for i, blk in enumerate(self.blocks):
            if i < freeze_depth:
                for p in blk.parameters():
                    p.requires_grad = False
                logger.info(f"Frozen Layer {i}")
            else:
                for p in blk.parameters():
                    p.requires_grad = True


def nvit_hybrid_tiny(**kwargs):
    return AdaptiveNViT(depth=12, embed_dim=192, num_heads=3, **kwargs)

def nvit_hybrid_small(**kwargs):
    return AdaptiveNViT(depth=12, embed_dim=384, num_heads=6, **kwargs)

def nvit_hybrid_base(**kwargs):
    return AdaptiveNViT(depth=12, embed_dim=768, num_heads=12, **kwargs)
