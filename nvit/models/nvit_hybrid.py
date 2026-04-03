
import torch
import torch.nn as nn
from functools import partial
import math

# Reuse components from existing ViT or re-define
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
# Import GCN
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../Code_Paper1_Diagnostics'))
from gcn_layers import KinematicGCNBlock

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity() # Simplification for now
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TokenToJointAggregator(nn.Module):
    """
    Bridge Layer: Transforms dense visual tokens (N=196) into skeletal joints (M=24).
    Implements the "Assembly" concept.
    """
    def __init__(self, in_tokens, out_joints, dim):
        super().__init__()
        # Learnable weight matrix to map Tokens -> Joints
        # "Which tokens correspond to the left elbow?"
        self.aggregator = nn.Parameter(torch.Tensor(out_joints, in_tokens))
        nn.init.xavier_uniform_(self.aggregator)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, N_tokens, Dim]
        # output: [B, N_joints, Dim]
        
        # We need to handle CLS token? Usually we drop CLS or keep it as Root.
        # Assume x includes CLS (N=197) or not.
        # Let's assume input is 196 (patches).
        
        # P = Softmax(W) for partial assignment
        attn = self.aggregator.softmax(dim=-1) # [M, N]
        
        # J = P * T
        out = torch.matmul(attn, x) # [B, M, D]
        out = self.norm(out)
        return out


class LiteMambaBlock(nn.Module):
    """
    Lightweight Implementation of State Space Model (Phase 2: Assembly).
    Simulates "Torso affects Limbs" using 1D Convolutions and Gating.
    Provides the linear complexity/state transition properties without full Mamba/CUDA dependencies.
    """
    def __init__(self, dim, d_state=16, expand=2., kernel_size=4):
        super().__init__()
        self.dim = dim
        self.inner_dim = int(dim * expand)
        
        # 1. Input Projection
        self.in_proj = nn.Linear(dim, self.inner_dim * 2)
        
        # 2. Convolution (Simulates local state transition)
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=kernel_size,
            groups=self.inner_dim,
            padding=kernel_size - 1
        )
        
        # 3. Activation
        self.act = nn.SiLU()
        
        # 4. Output Projection
        self.out_proj = nn.Linear(self.inner_dim, dim)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        
        shortcut = x
        
        # Project
        x_and_res = self.in_proj(x) # [B, N, 2*D]
        x_inner, res = x_and_res.split(self.inner_dim, dim=-1)
        
        # Conv1D (needs [B, C, N])
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :N] # Causality handled by padding/slicing
        x_conv = x_conv.transpose(1, 2)
        
        # Gating (Mamba-style)
        x_gate = self.act(x_conv) * self.act(res)
        
        # Project Out
        out = self.out_proj(x_gate)
        
        return out + shortcut

class NViT_Hybrid(nn.Module):
    def __init__(self, 
                 img_size=(256, 192), 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768, 
                 depth=32,
                 num_heads=12, 
                 phase1_end_layer=12,  # End of ViT (Search)
                 phase2_end_layer=24,  # End of Mamba (Assembly)
                 smpl_adj=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.phase1_end = phase1_end_layer
        self.phase2_end = phase2_end_layer
        
        # --- Phase 1: Search (Vision Transformer) ---
        # "Finding Parts" - Global Attention
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.vit_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads)
            for i in range(phase1_end_layer)
        ])
        
        # --- Transition: Token -> Joint ---
        # Bridge layer: Visual Patches -> Body Joints
        self.num_joints = 24
        self.bridge = TokenToJointAggregator(num_patches + 1, self.num_joints, embed_dim)
        
        # --- Phase 2: Assembly (Mamba / SSM) ---
        # "Assembling Parts" - Linear State Transition (Torso -> Limbs)
        # Using LiteMambaBlock to simulate SSM behavior without CUDA dependency
        self.mamba_blocks = nn.ModuleList([
            LiteMambaBlock(dim=embed_dim)
            for i in range(phase1_end_layer, phase2_end_layer)
        ])
        
        # --- Phase 3: Enforcement (Directed GCN) ---
        # "Global Fine-tuning" - Only Center -> Edge allowed
        if smpl_adj is None:
            # Load default Directed SMPL Adjacency
            # Assuming caller provides it, otherwise identity fallback
            smpl_adj = torch.eye(self.num_joints)
            
        self.gcn_blocks = nn.ModuleList([
            KinematicGCNBlock(embed_dim, embed_dim, smpl_adj)
            for i in range(phase2_end_layer, depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Init
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 1. Phase 1: Global Search (ViT)
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for blk in self.vit_blocks:
            x = blk(x)
            
        # Transition: Bridge to Joints
        # x is [B, 197, D] -> [B, 24, D]
        x_joints = self.bridge(x) 
        
        # 2. Phase 2: Assembly (Mamba)
        for blk in self.mamba_blocks:
            x_joints = blk(x_joints)

        # 3. Phase 3: Enforcement (GCN)
        for blk in self.gcn_blocks:
            x_joints = blk(x_joints)
            
        x_joints = self.norm(x_joints)
        return x_joints

def nvit_hybrid_tiny(**kwargs):
    # Example config: 8 ViT -> 4 Mamba -> 4 GCN (Total 16)
    return NViT_Hybrid(embed_dim=384, num_heads=6, phase1_end_layer=8, phase2_end_layer=12, depth=16, **kwargs)

def nvit_hybrid_base(**kwargs):
    # Standard Config: 12 ViT -> 12 Mamba -> 8 GCN (Total 32)
    return NViT_Hybrid(embed_dim=768, num_heads=12, phase1_end_layer=12, phase2_end_layer=24, depth=32, **kwargs)

