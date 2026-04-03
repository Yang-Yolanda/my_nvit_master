#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import torch.nn as nn
import numpy as np
import json
from model_manager import ModelManager

def analyze():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {'device': device, 'enable_detector': False}
    mm = ModelManager(config)
    mm.load_model()

    if mm.model is None:
        print("Error: Model not loaded")
        return

    backbone = mm.model.backbone if hasattr(mm.model, 'backbone') else None
    if backbone is None:
        print("Error: Backbone not found")
        return

    print(f"✅ Found backbone: {type(backbone)}")
    backbone.to(device)
    backbone.eval()

    if not hasattr(backbone, 'blocks'):
        print("Error: Backbone has no 'blocks'")
        return

    # 1. Cosine Similarity
    print("📈 Calculating Cosine Similarity...")
    input_size = (1, 3, 256, 192)
    x = torch.randn(*input_size).to(device)

    block_outputs = {}
    def get_hook(idx):
        def hook(module, input, output):
            block_outputs[idx] = output.detach().cpu()
        return hook

    hooks = []
    for i, block in enumerate(backbone.blocks):
        hooks.append(block.register_forward_hook(get_hook(i)))

    with torch.no_grad():
        _ = backbone.forward_features(x)

    for h in hooks: h.remove()

    similarities = []
    n_blocks = len(backbone.blocks)
    for i in range(n_blocks - 1):
        f1 = block_outputs[i].flatten(1)
        f2 = block_outputs[i+1].flatten(1)
        sim = torch.nn.functional.cosine_similarity(f1, f2).mean().item()
        similarities.append(float(sim))

    # 2. L1 Stats
    l1_stats = []
    for i, block in enumerate(backbone.blocks):
        fc1 = block.mlp.fc1
        norms = fc1.weight.abs().sum(dim=1).detach().cpu().numpy()
        l1_stats.append({
            'mean': float(np.mean(norms)),
            'near_zero': float(np.mean(norms < 0.1))
        })

    # Summary Table
    print("\n" + "="*70)
    print(f"{'Block':<5} | {'Cosine Sim':<12} | {'L1 Mean':<10} | {'Zero %'}")
    print("-" * 55)
    for i in range(n_blocks):
        sim = similarities[i] if i < n_blocks - 1 else 1.0
        st = l1_stats[i]
        tag = "<<< WATER" if 8 <= i <= 23 else ""
        print(f"{i:2d} | {sim:10.4f} | {st['mean']:8.3f} | {st['near_zero']*100:5.1f}% {tag}")
    print("="*70)

if __name__ == "__main__":
    analyze()
