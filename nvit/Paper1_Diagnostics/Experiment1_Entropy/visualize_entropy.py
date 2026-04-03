#!/home/yangz/.conda/envs/4D-humans/bin/python
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from pathlib import Path
import glob

def plot_entropy_curves():
    results_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/Experiment1_Entropy/results")
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Find all model directories
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    plotted_any = False
    
    for i, model_dir in enumerate(model_dirs):
        model_name = model_dir.name
        json_path = model_dir / "layer_metrics_Control.json"
        
        if not json_path.exists():
            print(f"Skipping {model_name}, no Control metrics found.")
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Extract Entropy Curve
        # Data format: "0": {"entropy": [val, val...], ...}
        # We want Mean Entropy per layer
        
        layers = sorted([int(k) for k in data.keys()])
        mean_entropies = []
        std_entropies = []
        
        for l in layers:
            vals = data[str(l)]['entropy']
            # Remove high dummy values if any
            vals = [v for v in vals if v < 90.0]
            if vals:
                mean_entropies.append(np.mean(vals))
                std_entropies.append(np.std(vals))
            else:
                mean_entropies.append(0)
                std_entropies.append(0)
        
        # Plot Comparison
        ax.plot(layers, mean_entropies, marker='o', label=model_name, linewidth=2, color=colors[i % len(colors)])
        
        # Save Per-Model Plot (User Request: Combined)
        fig_single, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:red' # Entropy = Chaos = Red
        ax1.set_xlabel('Layer Depth', fontsize=12)
        ax1.set_ylabel('Shannon Entropy', color=color, fontsize=12)
        l1, = ax1.plot(layers, mean_entropies, marker='o', color=color, linewidth=2, label='Entropy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()  
        color = 'tab:green' # Distance = Structure = Green
        
        mean_dists = []
        for l in layers:
            vals = data[str(l)].get('dist', [])
            if vals: mean_dists.append(np.mean(vals))
            else: mean_dists.append(0)
            
        ax2.set_ylabel('Mean Attn Distance (px)', color=color, fontsize=12)  
        l2, = ax2.plot(layers, mean_dists, marker='^', color=color, linewidth=2, linestyle='--', label='Distance')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Combined Legend
        lines = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title(f"Entropy vs Distance: {model_name}", fontsize=14)
        fig_single.tight_layout()
        fig_single.savefig(model_dir / "entropy_distance_combined.png", dpi=300)
        plt.close(fig_single) # Close to save memory
        
        plotted_any = True
        
    if not plotted_any:
        print("No data found to plot.")
        return

    ax.set_title("Layer-wise Attention Entropy (Experiment 1)", fontsize=14)
    ax.set_xlabel("Transformer Layer Depth", fontsize=12)
    ax.set_ylabel("Shannon Entropy (Mean)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    out_path = results_dir / "entropy_curve_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Entropy Plot saved to {out_path}")

if __name__ == "__main__":
    plot_entropy_curves()
