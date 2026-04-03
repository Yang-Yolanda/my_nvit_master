import os
import sys
import torch

print("TEST EVAL START")
humans_dir = "/home/yangz/4D-Humans"
if humans_dir not in sys.path: sys.path.append(humans_dir)
nvit_dir = "/home/yangz/NViT-master/nvit"
if nvit_dir not in sys.path: sys.path.append(nvit_dir)
imp_dir = os.path.join(nvit_dir, "Code_Paper2_Implementation")
if imp_dir not in sys.path: sys.path.append(imp_dir)

print("Imports starting...")
from hmr2.models import load_hmr2
print("hmr2 loaded.")
from nvit2_models.nvit_hybrid import AdaptiveNViT
print("NViT hybrid loaded.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = "/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"

print(f"Loading checkpoint from {ckpt}...")
model, _ = load_hmr2(ckpt)
model.to(device)
print("Model loaded and on device.")
print("SUCCESS")
