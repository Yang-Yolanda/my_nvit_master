#!/home/yangz/.conda/envs/4D-humans/bin/python
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_layer_kti(results_dir, output_path="kti_comparison.png"):
    """
    Plots Layer-wise KTI for different ablation settings.
    """
    files = [f for f in os.listdir(results_dir) if f.startswith("layer_metrics_") and f.endswith(".json")]
    
    plt.figure(figsize=(12, 6))
    
    for f in sorted(files):
        label = f.replace("layer_metrics_", "").replace(".json", "")
        with open(os.path.join(results_dir, f), 'r') as jf:
            data = json.load(jf)
        
        layers = sorted([int(k) for k in data.keys()])
        # Handle kti/kti key renaming
        kti_means = []
        for l in layers:
            vals = data[str(l)].get('kti', data[str(l)].get('kti', []))
            kti_means.append(np.mean(vals))
        
        plt.plot(layers, kti_means, marker='o', label=label)
    
    plt.xlabel("Layer Index")
    plt.ylabel("KTI Score (Soft Geodesic)")
    plt.title("Evolution of Kinematic Topology Interaction (KTI) across Layers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    res_dir = "/home/yangz/NViT-master/nvit/results/4D-Humans"
    if os.path.exists(res_dir):
        plot_layer_kti(res_dir, "nvit/Code_Paper2_Hybrid/kti_layer_plot.png")
    else:
        print(f"Directory {res_dir} not found. Skip plotting.")
