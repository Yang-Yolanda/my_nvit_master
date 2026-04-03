import torch
import os
import sys

# Minimal check
ckpt_path = 'checkpoints/ft_Control.ckpt'
print(f"Checking {ckpt_path}...")
if os.path.exists(ckpt_path):
    print("File exists.")
    try:
        data = torch.load(ckpt_path, map_location='cpu')
        print(f"Success! Keys: {list(data.keys())[:5]}")
    except Exception as e:
        print(f"Error loading: {e}")
else:
    print("File does not exist!")
