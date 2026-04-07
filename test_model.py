from omegaconf import OmegaConf
from nvit.Code_Paper2_Implementation.nvit2_models.nvit_hybrid import AdaptiveNViT
import sys

# Test spiral
m = AdaptiveNViT(depth=11, mamba_variant='spiral', gcn_variant='guided')
print('spiral keys:', [k for k in m.state_dict().keys() if 'blocks.8' in k][:3])

# Test bi
m = AdaptiveNViT(depth=11, mamba_variant='bi', gcn_variant='guided')
print('bi keys:', [k for k in m.state_dict().keys() if 'blocks.8' in k][:3])
