#!/home/yangz/.conda/envs/4D-humans/bin/python

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def plot_vitpose_kti(json_path, output_path):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    layers = sorted([int(k) for k in data.keys()])
    kti_means = [np.mean(data[str(l)]['kti']) for l in layers]
    kti_stds = [np.std(data[str(l)]['kti']) for l in layers]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, kti_means, marker='o', linestyle='-', color='tab:green', label='ViTPose Animal (AP10K)')
    plt.fill_between(layers, np.array(kti_means) - np.array(kti_stds), np.array(kti_means) + np.array(kti_stds), alpha=0.2, color='tab:green')
    
    plt.title('Layer-wise KTI: ViTPose Animal (AP10K)')
    plt.xlabel('Layer Index')
    plt.ylabel('KTI Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add a horizontal line for the Random Baseline (found to be ~0.006 in HaMeR with 16x16 grid)
    # For ViTPose (192 tokens/13x13 grid), the baseline is similarly low.
    plt.axhline(y=0.006, color='r', linestyle='--', label='Noise Floor (Random)')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    json_file = "nvit/Paper1_Diagnostics/Experiment2_KTI/results/ViTPose_Animal_AP10K/layer_metrics_Control.json"
    plot_file = "nvit/Paper1_Diagnostics/Experiment2_KTI/results/ViTPose_Animal_AP10K/kti_plot.png"
    plot_vitpose_kti(json_file, plot_file)
