import os
import torch
import numpy as np
from pathlib import Path
from nvit2_models.guided_hmr2 import GuidedHMR2Module
from hmr2.models.hmr2 import HMR2

# Environment-aware asset roots
FOURD_HUMANS_ROOT = "/home/yangz/4D-Humans"
CACHE_DATA_ROOT = "/home/yangz/.cache/4DHumans"

def patch_hmr2_config(cfg):
    """
    Code-level fix to resolve relative 'data/' paths into absolute paths
    based on the current environment, avoiding reliance on symbolic links.
    """
    def resolve_path(p):
        if not p or not isinstance(p, str) or os.path.isabs(p):
            return p
        
        # Candidate locations for relative paths (e.g., 'data/smpl_mean_params.npz')
        candidates = [
            os.path.join(FOURD_HUMANS_ROOT, p),
            os.path.join(CACHE_DATA_ROOT, p),
            os.path.join(os.getcwd(), p)
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return p

    # We use a try-except block to handle yacs CfgNode mechanics (defrost/freeze)
    try:
        is_frozen = getattr(cfg, 'is_frozen', lambda: False)()
        if is_frozen:
            cfg.defrost()
            
        # 1. Patch SMPL specific assets
        if hasattr(cfg, 'SMPL'):
            smpl_attrs = ['MEAN_PARAMS', 'MODEL_PATH', 'JOINT_REGRESSOR_EXTRA', 'JOINT_REGRESSOR_H36M']
            for attr in smpl_attrs:
                if hasattr(cfg.SMPL, attr):
                    val = getattr(cfg.SMPL, attr)
                    setattr(cfg.SMPL, attr, resolve_path(val))
        
        # 2. Patch Extra rendering/model assets
        if hasattr(cfg, 'EXTRA') and hasattr(cfg.EXTRA, 'CUBE_PARTS_FILE'):
            cfg.EXTRA.CUBE_PARTS_FILE = resolve_path(cfg.EXTRA.CUBE_PARTS_FILE)
            
        if is_frozen:
            cfg.freeze()
    except Exception as e:
        print(f"⚠️ Path patching warning: {e}")
        
    return cfg

def load_model_from_ckpt(ckpt_path: str, device="cuda"):
    """
    Enhanced loader that automatically patches checkpoint configs for portability.
    Supports robust device mapping.
    """
    # Fix device out of range issues
    if "cuda" in str(device):
        try:
            requested_idx = int(str(device).split(":")[-1]) if ":" in str(device) else 0
            if requested_idx >= torch.cuda.device_count():
                print(f"⚠️ Requested GPU {requested_idx} but only {torch.cuda.device_count()} available. Mapping to cpu.")
                device = "cpu"
        except:
             pass
             
    ckpt_lower = ckpt_path.lower()
    
    # 1. Pre-load checkpoint to extract and patch config
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        hparams = checkpoint.get('hyper_parameters', {})
        cfg = hparams.get('cfg', None)
        if cfg is not None:
            cfg = patch_hmr2_config(cfg)
    except Exception as e:
        print(f"⚠️ Could not pre-patch config: {e}")
        cfg = None

    # 2. Instantiate with patched config
    load_kwargs = {'strict': False, 'map_location': device}
    if cfg is not None:
        load_kwargs['cfg'] = cfg

    if ("guided" in ckpt_lower) or ("mamba" in ckpt_lower):
        model = GuidedHMR2Module.load_from_checkpoint(ckpt_path, **load_kwargs)
        return model

    try:
        model = HMR2.load_from_checkpoint(ckpt_path, **load_kwargs)
        return model
    except Exception:
        # Final fallback
        model = GuidedHMR2Module.load_from_checkpoint(ckpt_path, **load_kwargs)
        return model
