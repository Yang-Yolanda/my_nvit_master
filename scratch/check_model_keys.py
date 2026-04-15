import torch
import sys
import os
from pathlib import Path

# Add project paths
PROJ_ROOT = Path('/cpfs_infra/shared/yangz/NViT-master')
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(PROJ_ROOT / 'nvit/Code_Paper2_Implementation'))
sys.path.insert(0, '/cpfs_infra/shared/yangz/4D-Humans')

from omegaconf import OmegaConf
from nvit2_models.guided_hmr2 import GuidedHMR2Module

# Create a dummy config
cfg = OmegaConf.create({
    'MODEL': {
        'BACKBONE': {
            'USE_ADAPTIVE_NVIT': True,
            'depth': 11,
            'switch_layer_1': 8,
            'switch_layer_2': 10,
            'mamba_variant': 'spiral',
            'gcn_variant': 'guided'
        },
        'SMPL_HEAD': {
            'TRANSFORMER_DECODER': {'depth': 3, 'heads': 4, 'mlp_dim': 1024}
        },
        'IMAGE_SIZE': 256
    },
    'LOSS_WEIGHTS': {'HEATMAP': 10.0},
    'TRAIN': {'LR': 1e-5, 'WEIGHT_DECAY': 1e-4},
    'GENERAL': {'LOG_STEPS': 10},
    'EXTRA': {'FOCAL_LENGTH': 5000}
})

model = GuidedHMR2Module(cfg)
keys = list(model.state_dict().keys())
print("First 50 model keys:")
for k in keys[:50]:
    print(k)

# Check specifically for backbone blocks
backbone_block_keys = [k for k in keys if 'nvit_backbone.blocks.0' in k]
print("\nnvit_backbone.blocks.0 keys:")
for k in backbone_block_keys:
    print(k)
