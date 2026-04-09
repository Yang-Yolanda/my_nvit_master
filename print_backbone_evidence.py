import sys
import torch
from pathlib import Path

# Add paths
sys.path.append('/home/yangz/4D-Humans')

from hmr2.models.backbones.vit import ViT

# Mock config or just instantiate with standard params found in vit.py
# img_size=(256, 192), patch_size=16, embed_dim=1280, depth=32, num_heads=16
model = ViT(
    img_size=(256, 192),
    patch_size=16,
    embed_dim=1280,
    depth=32,
    num_heads=16,
    ratio=1,
)
model.eval()

x = torch.zeros(1, 3, 256, 192)
feat, (Hp, Wp) = model.patch_embed(x)

print(f'HMR2 Patch Embed Shape: {feat.shape}')
print(f'HMR2 Grid: Hp={Hp}, Wp={Wp}')
print(f'HMR2 blocks: {len(model.blocks)}')
has_cls = hasattr(model, 'cls_token') and getattr(model, 'cls_token', None) is not None
print(f'Has CLS token: {has_cls}')

# Check pos_embed logic
print(f'Pos Embed Shape: {model.pos_embed.shape}')
