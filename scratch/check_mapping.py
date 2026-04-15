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

# Load actual checkpoint keys
ckpt_path = '/cpfs_infra/shared/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict']

# Create model
cfg = OmegaConf.create({
    'MODEL': {
        'BACKBONE': {
            'USE_ADAPTIVE_NVIT': True,
            'TYPE': 'vit',
            'depth': 12,
            'switch_layer_1': 8,
            'switch_layer_2': 10,
            'mamba_variant': 'spiral',
            'gcn_variant': 'guided'
        },
        'SMPL_HEAD': {
            'TYPE': 'transformer_decoder',
            'IN_CHANNELS': 2048,
            'TRANSFORMER_DECODER': {'depth': 6, 'heads': 8, 'mlp_dim': 1024, 'dim_head': 64, 'dropout': 0.0, 'emb_dropout': 0.0, 'norm': 'layer', 'context_dim': 1280}
        },
        'IMAGE_SIZE': 256
    },
    'LOSS_WEIGHTS': {'HEATMAP': 1.0, 'KEYPOINTS_3D': 5.0, 'KEYPOINTS_2D': 10.0, 'GLOBAL_ORIENT': 5.0, 'BODY_POSE': 5.0, 'BETAS': 0.001, 'ADVERSARIAL:': 0.0},
    'TRAIN': {'LR': 1e-5, 'WEIGHT_DECAY': 1e-4},
    'GENERAL': {'LOG_STEPS': 10},
    'EXTRA': {'FOCAL_LENGTH': 5000}
})

model = GuidedHMR2Module(cfg)
model_state_dict = model.state_dict()
filtered_state_dict = {}

# MAPPING LOGIC (Simulated from train_guided.py)
print("Running Mapping Simulation...")
for k, v in state_dict.items():
    k_new = k
    if k.startswith('backbone.'):
        k_new = k.replace('backbone.', 'nvit_backbone.', 1)
    
    if 'blocks.' in k_new and '.block.' not in k_new:
        parts = k_new.split('.')
        try:
            idx = parts.index('blocks')
            if idx + 1 < len(parts) and parts[idx+1].isdigit():
                parts.insert(idx + 2, 'block')
                k_new = '.'.join(parts)
        except (ValueError, IndexError):
            pass
    
    if k_new in model_state_dict:
        if v.shape == model_state_dict[k_new].shape:
            filtered_state_dict[k_new] = v

total_model_keys = len(model_state_dict)
matched_keys = len(filtered_state_dict)
match_rate = matched_keys / total_model_keys
print(f"Total Model Keys: {total_model_keys}")
print(f"Matched CKPT Keys: {matched_keys}")
print(f"Match Rate: {match_rate:.2%}")

if match_rate > 0.80:
    print("SUCCESS: Match rate is healthy!")
else:
    print("FAILURE: Match rate still too low.")
    # Debug a few missing keys
    missing = [k for k in model_state_dict.keys() if k not in filtered_state_dict]
    print("\nFirst 10 missing keys:")
    for m in missing[:10]:
        print(m)
