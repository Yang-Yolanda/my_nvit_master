import torch
import sys
from pathlib import Path

# Path to the checkpoint
ckpt_path = '/home/yangz/NViT-master/nvit/Paper1_Diagnostics/checkpoints/ft_T2-A-H-Baseline.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict']

# Check backbone blocks
block_keys = [k for k in state_dict.keys() if 'backbone.blocks' in k]
depth = max([int(k.split('backbone.blocks.')[1].split('.')[0]) for k in block_keys]) + 1

# Check patch_embed weights to infer patch_size
# backbone.patch_embed.proj.weight shape is [out_channels, in_channels, kh, kw]
weight_key = 'backbone.patch_embed.proj.weight'
if weight_key in state_dict:
    weight = state_dict[weight_key]
    out_c, in_c, kh, kw = weight.shape
    patch_size = kh # Should be 16
else:
    patch_size = "unknown"

# Check pos_embed to infer grid
pos_embed_key = 'backbone.pos_embed'
if pos_embed_key in state_dict:
    pos_embed = state_dict[pos_embed_key]
    # shape is (1, N+1, C)
    n_tokens = pos_embed.shape[1]
    has_cls = True # Usually HMR2 pos_embed is N+1
    grid_n = n_tokens - 1
else:
    grid_n = "unknown"
    has_cls = "unknown"

print(f'HMR2 Patch Embed Weight Shape: {state_dict.get(weight_key, torch.zeros(0)).shape}')
print(f'HMR2 Grid Tokens: {grid_n} (e.g., 16x12=192)')
print(f'HMR2 blocks: {depth}')
print(f'Has Pos Embed for CLS: {has_cls}')
