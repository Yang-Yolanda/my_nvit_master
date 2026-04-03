#!/home/yangz/.conda/envs/4D-humans/bin/python
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from pathlib import Path

def plot_kti_curves():
    results_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/Experiment2_KTI/results")
    
    # 1. Gather all valid data
    model_data = {}
    all_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "Random" not in d.name and "Scratch" not in d.name]
    
    for model_dir in all_dirs:
        json_path = model_dir / "layer_metrics_Control.json"
        if not json_path.exists(): continue
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        layers = sorted([int(k) for k in data.keys()])
        mean_ktis = [np.mean(data[str(l)].get('kti', data[str(l)].get('kti', []))) for l in layers]
        model_data[model_dir.name] = (layers, mean_ktis)

    # 2. Define Groups
    groups = {
        "robot": [name for name in model_data if "Robot" in name],
        "hmr_hsmr": [name for name in model_data if "HMR" in name or "HSMR" in name],
        "others": [name for name in model_data if "Robot" not in name and "HMR" not in name and "HSMR" not in name and "CLIP" not in name]
    }

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    for group_name, model_names in groups.items():
        if not model_names: continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, name in enumerate(model_names):
            layers, vals = model_data[name]
            ax.plot(layers, vals, marker='s', label=name, linewidth=2, color=colors[i % len(colors)])
        
        ax.set_title(f"KTI Depth Analysis: {group_name.replace('_', ' & ').upper()}", fontsize=14)
        ax.set_xlabel("Transformer Layer Depth", fontsize=12)
        ax.set_ylabel("KTI Score (Structure Similarity)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        
        out_path = results_dir / f"kti_curve_{group_name}.png"
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved Group Plot: {out_path}")

def plot_model_comparison_bar():
    """
    Generate a bar chart comparing Peak KTI across models (No Random/No CLIP).
    """
    results_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/Experiment2_KTI/results")
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "Random" not in d.name and "Scratch" not in d.name and "CLIP" not in d.name]
    
    model_names = []
    peak_ktis = []
    
    for model_dir in model_dirs:
        json_path = model_dir / "layer_metrics_Control.json"
        if not json_path.exists(): continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        max_kti = 0
        for l in data:
            vals = data[l].get('kti', data[l].get('kti', []))
            if vals:
                m = np.mean(vals)
                if m > max_kti: max_kti = m
        
        model_names.append(model_dir.name)
        peak_ktis.append(max_kti)
        
    if not model_names: return

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(model_names, peak_ktis, color='skyblue', edgecolor='black')
    
    ax.set_title("Peak KTI Score Comparison (Official Models)", fontsize=14)
    ax.set_ylabel("Peak KTI", fontsize=12)
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig(results_dir / "kti_bar_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Saved Bar Chart: {results_dir / 'kti_bar_comparison.png'}")

if __name__ == "__main__":
    plot_kti_curves()
    plot_model_comparison_bar()
