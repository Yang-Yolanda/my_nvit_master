#!/home/yangz/.conda/envs/4D-humans/bin/python
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from pathlib import Path
import seaborn as sns

def load_data(results_dir):
    data_map = {}
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir(): continue
        json_path = model_dir / "layer_metrics_Control.json"
        if not json_path.exists(): continue
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        layers = sorted([int(k) for k in data.keys()])
        kti_means = []
        kti_stds = []
        
        for l in layers:
            vals = data[str(l)].get('kti', data[str(l)].get('kti', []))
            if vals:
                kti_means.append(np.mean(vals))
                kti_stds.append(np.std(vals))
            else:
                kti_means.append(0)
                kti_stds.append(0)
        
        data_map[model_dir.name] = {
            'layers': layers,
            'means': np.array(kti_means),
            'stds': np.array(kti_stds)
        }
    return data_map

def plot_robustness():
    base_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics")
    kti_results = base_dir / "Experiment2_KTI" / "results"
    output_dir = base_dir / "paper1_figures_refined"
    output_dir.mkdir(exist_ok=True)
    
    data = load_data(kti_results)
    
    # Selection of "Representative" models for cross-task proof
    # Human: HMR2, HSMR
    # Hand: HaMeR_Random_FreiHAND (assuming this is the tuned core)
    # Animal: ViTPose_Animal_AP10K (assuming)
    # Robot: Toy_Robot_Pretrained
    
    categories = {
        'Human (HMR2)': 'HMR2',
        'Human (HSMR)': 'HSMR',
        'Hand (HaMeR)': 'HaMeR_Random_FreiHAND',
        'Animal (ViTPose)': 'ViTPose_Animal_AP10K',
        'Robot (Pretrained)': 'Toy_Robot_Pretrained'
    }
    
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    colors = sns.color_palette("husl", len(categories))
    
    for i, (label, key) in enumerate(categories.items()):
        if key not in data: 
            print(f"Skipping {key} - not found")
            continue
            
        d = data[key]
        plt.plot(d['layers'], d['means'], label=label, color=colors[i], linewidth=2.5)
        plt.fill_between(d['layers'], d['means'] - d['stds']*0.5, d['means'] + d['stds']*0.5, 
                         color=colors[i], alpha=0.15)
        
        # Mark Peak 1 and 2 for Human as primary evidence
        if 'HMR2' in key:
            p1_idx = 7
            p2_idx = 10
            plt.scatter([p1_idx, p2_idx], [d['means'][p1_idx], d['means'][p2_idx]], 
                        color='red', zorder=5, s=60, edgecolors='black')

    plt.title("Figure 1: Robustness of KTI Double-Peak Across Domains", fontsize=16, fontweight='bold')
    plt.xlabel("Transformer Layer Depth", fontsize=14)
    plt.ylabel("KTI Score (Structural Awareness)", fontsize=14)
    plt.legend(frameon=True, fontsize=12)
    plt.ylim(0, 0.25)
    plt.tight_layout()
    
    save_path = output_dir / "Paper1_Fig1_Robustness.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_robustness()
