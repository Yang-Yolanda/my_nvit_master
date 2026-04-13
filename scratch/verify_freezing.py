import torch
import torch.nn as nn
from omegaconf import OmegaConf
import sys
import os

# Set paths
sys.path.append("/home/yangz/NViT-master/nvit/Code_Paper2_Implementation")
sys.path.append("/home/yangz/4D-Humans")

from nvit2_models.guided_hmr2 import GuidedHMR2Module

# Complete config needed for initialization
cfg = OmegaConf.create({
    'MODEL': {
        'USE_ADAPTIVE_NVIT': True,
        'BACKBONE': {
            'TYPE': 'vit',
            'depth': 11,
            'switch_layer_1': 8,
            'switch_layer_2': 10,
            'MAMBA_VARIANT': 'spiral',
            'GCN_VARIANT': 'guided'
        },
        'SMPL_HEAD': {
            'TYPE': 'transformer_decoder',
            'IN_CHANNELS': 1280,
            'TRANSFORMER_DECODER': {'depth': 3, 'heads': 4, 'mlp_dim': 1024}
        }
    },
    'TRAIN': {
        'LR': 1e-5,
        'HEAD_LR': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'ACCUMULATE_GRAD_BATCHES': 1,
        'GRAD_CLIP_VAL': 0.5
    },
    'LOSS_WEIGHTS': {
        'ADVERSARIAL': 0.0,
        'KEYPOINTS_3D': 5.0,
        'KEYPOINTS_2D': 10.0,
        'GLOBAL_ORIENT': 5.0,
        'BODY_POSE': 5.0,
        'BETAS': 0.001,
        'HEATMAP': 1.0
    },
    'GENERAL': {
        'LOG_STEPS': 10
    }
})

def verify_freezing():
    print("\n--- Parameter Freezing Verification ---")
    model = GuidedHMR2Module(cfg)
    
    # Check Stage 0 (first 8 layers)
    stage0_frozen = True
    sl1 = cfg.MODEL.BACKBONE.switch_layer_1
    for i in range(sl1):
        layer = model.nvit_backbone.layers[i]
        for name, param in layer.named_parameters():
            if param.requires_grad:
                print(f"❌ Layer {i} ({layer.type}) param {name} is NOT frozen!")
                stage0_frozen = False
    if stage0_frozen:
        print(f"✅ Stage 0 (Layers 0-{sl1-1}) is FULLY FROZEN.")
        
    # Check Stage 1 (last layers)
    stage1_trainable = True
    sl2 = cfg.MODEL.BACKBONE.switch_layer_2
    for i in range(sl1, sl2 + 1):
        layer = model.nvit_backbone.layers[i]
        for name, param in layer.named_parameters():
            if not param.requires_grad:
                print(f"⚠️ Layer {i} ({layer.type}) param {name} is unexpectedly frozen.")
                stage1_trainable = False
    if stage1_trainable:
        print(f"✅ Stage 1 (Mamba/GCN) is TRAINABLE.")
        
    # Check SMPL Head
    head_trainable = True
    for name, param in model.smpl_head.named_parameters():
        if not param.requires_grad:
            # Note: Bias might be trainable even if weights are frozen if we follow some policies,
            # but here our freeze_stages freezes everything in the layer.
            head_trainable = False
    if head_trainable:
        print("✅ SMPL Head is FULLY TRAINABLE.")

if __name__ == "__main__":
    verify_freezing()
