import os
import torch

from nvit2_models.guided_hmr2 import GuidedHMR2Module
from hmr2.models.hmr2 import HMR2

def load_model_from_ckpt(ckpt_path: str, device="cuda"):
    ckpt_lower = ckpt_path.lower()

    if ("guided" in ckpt_lower) or ("mamba" in ckpt_lower):
        model = GuidedHMR2Module.load_from_checkpoint(
            ckpt_path, strict=False, map_location=device
        )
        return model

    try:
        model = HMR2.load_from_checkpoint(
            ckpt_path, strict=False, map_location=device
        )
        return model
    except Exception:
        model = GuidedHMR2Module.load_from_checkpoint(
            ckpt_path, strict=False, map_location=device
        )
        return model
