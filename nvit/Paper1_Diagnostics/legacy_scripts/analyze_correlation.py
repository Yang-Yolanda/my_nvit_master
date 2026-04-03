#!/home/yangz/.conda/envs/4D-humans/bin/python
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path
import seaborn as sns
from scipy.stats import pearsonr

def load_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    layers = sorted([int(k) for k in data.keys()])
    metrics = {
        'kti': [],
        'entropy': [],
        'rank': []
    }
    for l in layers:
        metrics['kti'].append(np.mean(data[str(l)].get('kti', data[str(l)].get('kti', [0]))))
        metrics['entropy'].append(np.mean(data[str(l)].get('entropy', [0])))
        metrics['rank'].append(np.mean(data[str(l)].get('rank', [0])))
    return layers, metrics

def plot_correlation():
    base_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics")
    hmr2_res = base_dir / "Experiment3_Masking" / "results" / "HMR2" / "layer_metrics_Control.json"
    output_dir = base_dir / "paper1_figures_refined"
    output_dir.mkdir(exist_ok=True)
    
    layers, metrics = load_metrics(hmr2_res)
    
    # Normalize for triple-curve alignment
    def norm(v): return (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-6)
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, norm(metrics['kti']), label='KTI (Structure)', color='blue', linewidth=3)
    plt.plot(layers, 1 - norm(metrics['entropy']), label='1 - Entropy (Focus)', color='gray', linestyle='--')
    plt.plot(layers, norm(metrics['rank']), label='Eff. Rank (Diversity)', color='green', alpha=0.5)
    
    plt.title("Figure 2a: Alignment of Structure, Focus, and Diversity", fontsize=14)
    plt.xlabel("Layer Depth")
    plt.ylabel("Normalized Magnitude")
    plt.legend()
    plt.savefig(output_dir / "Paper1_Fig2a_Alignment.png", dpi=300)
    plt.close()

    # Scatter Plot
    plt.figure(figsize=(8, 7))
    r_val, p_val = pearsonr(metrics['kti'], metrics['entropy'])
    
    sns.regplot(x=metrics['kti'], y=metrics['entropy'], scatter_kws={'s': 50, 'alpha': 0.6}, line_kws={'color': 'red'})
    plt.title(f"Figure 2b: KTI vs. Entropy Correlation (r = {r_val:.3f})", fontsize=14)
    plt.xlabel("KTI Score")
    plt.ylabel("Attention Entropy")
    plt.savefig(output_dir / "Paper1_Fig2b_Scatter.png", dpi=300)
    plt.close()
    
    print(f"Correlation Analysis Complete. r = {r_val:.3f}")

if __name__ == "__main__":
    plot_correlation()
