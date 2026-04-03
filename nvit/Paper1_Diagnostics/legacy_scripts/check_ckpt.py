import torch
import sys

# Path to PromptHMR checkpoint used in config
ckpt_path = '/home/yangz/NViT-master/external_models/PromptHMR/data/released_models/PromptHMR_v1.pth'
print(f"Loading {ckpt_path}")

try:
    ckpt = torch.load(ckpt_path)
    print(f"Checkpoint type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"Keys: {ckpt.keys()}")
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
            print(f"State Dict Keys Sample (first 5): {list(sd.keys())[:5]}")
        else:
             print(f"Direct State Dict Sample: {list(ckpt.keys())[:5]}")
except Exception as e:
    print(e)
