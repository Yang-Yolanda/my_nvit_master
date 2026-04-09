import sys, torch
sys.path.append('/home/yangz/4D-Humans')
from hmr2.models import HMR2
import yaml
from yacs.config import CfgNode
from omegaconf import OmegaConf

try:
    # Just load model from checkpoint to avoid config hell, but map_location=cpu
    model = HMR2.load_from_checkpoint('/home/yangz/NViT-master/nvit/Paper1_Diagnostics/checkpoints/ft_T2-A-H-Baseline.ckpt', map_location='cpu', strict=False).eval()
    x = torch.zeros(1, 3, 256, 192)
    feat, (Hp, Wp) = model.backbone.patch_embed(x)
    print(f'HMR2 Patch Embed Shape: {feat.shape}')
    print(f'HMR2 Grid: Hp={Hp}, Wp={Wp}')
    print(f'HMR2 blocks: {len(model.backbone.blocks)}')
    has_cls = hasattr(model.backbone, 'cls_token') and getattr(model.backbone, 'cls_token', None) is not None
    print(f'Has CLS token: {has_cls}')
except Exception as e:
    print(f"Error: {e}")
