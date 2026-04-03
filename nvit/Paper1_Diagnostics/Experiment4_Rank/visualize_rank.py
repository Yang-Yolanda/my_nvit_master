#!/home/yangz/.conda/envs/4D-humans/bin/python
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from pathlib import Path

def plot_rank_curves():
    results_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/Experiment4_Rank/results")
    
    # 1. Gather all valid data
    model_data = {}
    all_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "Random" not in d.name and "Scratch" not in d.name]
    
    for model_dir in all_dirs:
        json_path = model_dir / "layer_metrics_Control.json"
        if not json_path.exists(): continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        layers = sorted([int(k) for k in data.keys()])
        mean_ranks = []
        for l in layers:
            vals = data[str(l)].get('rank', [])
            if vals:
                mean_ranks.append(np.mean(vals))
            else:
                mean_ranks.append(0)
        
        model_data[model_dir.name] = (layers, mean_ranks)

    if not model_data:
        print("No valid rank data found in", results_dir)
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    for i, (name, (layers, vals)) in enumerate(model_data.items()):
        ax.plot(layers, vals, marker='o', label=name, linewidth=2, color=colors[i % len(colors)])
    
    ax.set_title("Layer-wise Effective Rank (Redundancy Analysis)", fontsize=14)
    ax.set_xlabel("Transformer Layer Depth", fontsize=12)
    ax.set_ylabel("Effective Rank (Feature Diversity)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    out_path = results_dir / "effective_rank_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Rank Plot: {out_path}")

if __name__ == "__main__":
    plot_rank_curves()
