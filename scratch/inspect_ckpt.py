import torch
import os

ckpt_path = '/mnt/yangz/nvit_output/smoke_test/train/runs/2026-04-14_19-15-22/checkpoints/step_step=000010.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"Checkpoint keys: {ckpt.keys()}")
if 'state_dict' in ckpt:
    sd = ckpt['state_dict']
    total_params = 0
    for k, v in sd.items():
        total_params += v.numel()
    print(f"Total params in state_dict: {total_params:,}")
    print(f"Approx size of state_dict (FP32): {total_params * 4 / (1024**3):.2f} GB")

if 'optimizer_states' in ckpt:
    opt_states = ckpt['optimizer_states']
    print(f"Number of optimizer states: {len(opt_states)}")
    # Check size of optimizer states
    # This is harder without a full environment, but we can try to guess
    print("Optimizer states found.")
else:
    print("No optimizer states found.")

print(f"File size: {os.path.getsize(ckpt_path) / (1024**3):.2f} GB")
