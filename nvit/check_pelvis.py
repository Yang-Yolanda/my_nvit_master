#!/home/yangz/.conda/envs/4D-humans/bin/python
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
print(f"PELVIS_IND: {model_cfg.EXTRA.PELVIS_IND}")
