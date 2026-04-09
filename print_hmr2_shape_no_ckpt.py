import sys, torch
sys.path.append('/home/yangz/4D-Humans')
from hmr2.models.backbones.vit import vit
import warnings
warnings.filterwarnings('ignore')

# Provide a dummy cfg (it's not used in vit() function anyway according to source)
model = vit(None)
x = torch.zeros(1, 3, 256, 192)
feat, (Hp, Wp) = model.patch_embed(x)
print(f'HMR2 Patch Embed Shape: {feat.shape}')
print(f'HMR2 Grid: Hp={Hp}, Wp={Wp}')
print(f'HMR2 blocks: {len(model.blocks)}')
has_cls = hasattr(model, 'cls_token') and getattr(model, 'cls_token', None) is not None
print(f'Has CLS token: {has_cls}')
