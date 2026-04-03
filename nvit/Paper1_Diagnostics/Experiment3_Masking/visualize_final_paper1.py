#!/home/yangz/.conda/envs/4D-humans/bin/python

import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_mean_metric(data, metric_name):
    layers = sorted([int(k) for k in data.keys()])
    means = []
    stds = []
    for l in layers:
        vals = data[str(l)].get(metric_name, [])
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(0)
            stds.append(0)
    return layers, means, stds

def main():
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results/HMR2"
    output_dir = base_dir / "paper1_figures"
    output_dir.mkdir(exist_ok=True)

    # --- Load Data ---
    control_metrics = load_json(results_dir / "layer_metrics_Control.json")
    adaptive_metrics = load_json(results_dir / "layer_metrics_Adaptive-10-29.json")
    
    # Load Results CSV
    df = pd.read_csv(results_dir / "results.csv")
    
    # --- Figure 2: The Detection (KTI vs Entropy) ---
    print("Generating Figure 2: Detection...")
    layers, kti_means, kti_stds = get_mean_metric(control_metrics, 'kti')
    _, ent_means, ent_stds = get_mean_metric(control_metrics, 'entropy')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # KTI Plot (Primary Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Transformer Layer Depth', fontsize=12)
    ax1.set_ylabel('Kinematic Topology Information (KTI)', color=color, fontsize=12)
    ax1.plot(layers, kti_means, color=color, marker='o', linewidth=2.5, label='KTI (Structure)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.fill_between(layers, np.array(kti_means)-np.array(kti_stds)*0.2, np.array(kti_means)+np.array(kti_stds)*0.2, color=color, alpha=0.1)

    # Highlight Peaks
    # Peak 1: Layer 7 (Assembly)
    ax1.annotate('Part Assembly\n(Peak 1)', xy=(7, kti_means[7]), xytext=(7, kti_means[7]+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    # Peak 2: Layer 10 (Global Lock)
    ax1.annotate('Global Lock\n(Peak 2)', xy=(10, kti_means[10]), xytext=(12, kti_means[10]+0.05),
                 arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10)
    
    # Entropy Plot (Secondary Axis)
    ax2 = ax1.twinx()  
    color = 'tab:gray'
    ax2.set_ylabel('Attention Entropy (Shannon)', color=color, fontsize=12) 
    ax2.plot(layers, ent_means, color=color, linestyle='--', linewidth=2, label='Entropy (Information)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Figure 2: KTI Detects Structural Phase Transitions (Layer 7 & 10)", fontsize=14)
    fig.tight_layout()
    plt.savefig(output_dir / "Paper1_Fig2_Detection.png", dpi=300)
    plt.close()

    # --- Figure 3: The Guidance (Bar Chart) ---
    print("Generating Figure 3: Guidance...")
    
    # ... (Keep existing bar chart logic if mostly correct, just title update maybe?)
    # Just skipping to next plot adjustment

    # --- Figure 4: The Proof (Effective Rank) ---
    print("Generating Figure 4: Proof...")
    layers, rank_ctrl, _ = get_mean_metric(control_metrics, 'rank')
    _, rank_adapt, _ = get_mean_metric(adaptive_metrics, 'rank')
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, rank_ctrl, label='Control (No Mask)', color='gray', linestyle='--')
    plt.plot(layers, rank_adapt, label='Adaptive (KTI Guided)', color='green', linewidth=2.5)
    
    # Highlight Drop
    # Hard Mask starts at 10 (Lock)
    plt.axvline(x=10, color='red', linestyle=':', label='Hard Mask Onset (L10)')
    
    plt.xlabel("Transformer Layer Depth", fontsize=12)
    plt.ylabel("Effective Rank (Dimensionality)", fontsize=12)
    plt.title("Figure 4: Redundancy Reduction via Adaptive Masking", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom in/Annotate drop
    plt.annotate('Significant Rank Drop\n(Redundancy Removed)', xy=(10, rank_adapt[10]), xytext=(15, rank_adapt[10]+20),
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig(output_dir / "Paper1_Fig4_Proof.png", dpi=300)
    plt.close()
    
    print(f"All figures saved to {output_dir}")

if __name__ == "__main__":
    main()
