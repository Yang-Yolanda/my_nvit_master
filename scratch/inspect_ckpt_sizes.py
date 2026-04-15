import torch
import os

ckpt_path = '/mnt/yangz/nvit_output/smoke_test/train/runs/2026-04-14_19-15-22/checkpoints/step_step=000010.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')

import sys
import pickle

def get_size(obj):
    return len(pickle.dumps(obj))

print(f"{'Key':<30} | {'Size (MB)':<10}")
print("-" * 45)
for k, v in ckpt.items():
    size_mb = get_size(v) / (1024**2)
    print(f"{k:<30} | {size_mb:>10.2f}")

if 'state_dict' in ckpt:
    sd = ckpt['state_dict']
    print("\nTop 5 largest params in state_dict:")
    params = sorted(sd.items(), key=lambda x: x[1].numel(), reverse=True)
    for k, v in params[:5]:
        print(f"{k:<50} | {v.numel() * 4 / (1024**2):.2f} MB")
