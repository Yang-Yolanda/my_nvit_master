import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def aggregate_seed_data(model_name, seeds=[0, 1, 2]):
    metrics_path = Path('outputs/diagnostics') / model_name
    all_seed_metrics = []
    
    for seed in seeds:
        seed_file = metrics_path / f'seed_{seed}' / 'layer_metrics.json'
        if seed_file.exists():
            with open(seed_file, 'r') as f:
                all_seed_metrics.append(json.load(f))
    
    if not all_seed_metrics:
        return None
        
    # Standardize layer indices
    layers = sorted(all_seed_metrics[0].keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else int(x))
    
    aggregated = {}
    for layer in layers:
        layer_data = {}
        # Each layer has: entropy, kti, rank, rank_feat, dist
        m_keys = all_seed_metrics[0][layer].keys()
        for k in m_keys:
            vals = []
            for s_m in all_seed_metrics:
                if layer in s_m and k in s_m[layer]:
                    vals.extend(s_m[layer][k])
            if vals:
                layer_data[k] = {'mean': np.mean(vals), 'std': np.std(vals)}
        aggregated[layer] = layer_data
    
    return aggregated, layers

def plot_single_metric(model_name, agg_data, layers, metric_key, ylabel, title, out_path):
    plt.figure(figsize=(10, 6))
    x = [int(l.split('_')[1]) if '_' in l else int(l) for l in layers]
    y = [agg_data[l][metric_key]['mean'] for l in layers]
    err = [agg_data[l][metric_key]['std'] for l in layers]
    
    plt.plot(x, y, marker='o', label=model_name)
    plt.fill_between(x, np.array(y)-np.array(err), np.array(y)+np.array(err), alpha=0.2)
    
    plt.xlabel('Layer Index')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_path)
    plt.close()

def plot_overlay_kti(models_data, out_path):
    plt.figure(figsize=(12, 7))
    for model_name, (agg_data, layers) in models_data.items():
        x = [int(l.split('_')[1]) if '_' in l else int(l) for l in layers]
        y = [agg_data[l]['kti']['mean'] if 'kti' in agg_data[l] else 0 for l in layers]
        plt.plot(x, y, marker='s', label=model_name, linewidth=2)
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('KTI Score', fontsize=12)
    plt.title('Kinematic Topology Interaction (KTI) Over Models', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(out_path)
    plt.close()

def generate_summary(models_data):
    summary_rows = []
    for model_name, (agg_data, layers) in models_data.items():
        kti_vals = [agg_data[l]['kti']['mean'] for l in layers]
        peak_layer = layers[np.argmax(kti_vals)]
        peak_val = np.max(kti_vals)
        
        # Simple bimodal detection: is there a peak followed by a drop then another peak?
        # For small layer counts, we look for two local maxima
        local_maxima = []
        for i in range(1, len(kti_vals)-1):
            if kti_vals[i] > kti_vals[i-1] and kti_vals[i] > kti_vals[i+1]:
                local_maxima.append(i)
        bimodal = "Yes" if len(local_maxima) >= 2 else "No"
        
        # Rear drop detection
        rear_drop = "Significant" if kti_vals[-1] < 0.7 * peak_val else "Gradual"
        
        summary_rows.append({
            'Model': model_name,
            'Peak_Layer': peak_layer,
            'Max_KTI': f"{peak_val:.4f}",
            'Bimodal': bimodal,
            'Rear_Drop': rear_drop,
            'Avg_ERank': f"{np.mean([agg_data[l]['rank_feat']['mean'] for l in layers]):.2f}"
        })
    
    return pd.DataFrame(summary_rows)

def main():
    models = ['HMR2', 'PromptHMR', 'HSMR', 'CameraHMR']
    models_data = {}
    
    out_root = Path('outputs/diagnostics/summary_plots')
    out_root.mkdir(parents=True, exist_ok=True)
    
    for m in models:
        res = aggregate_seed_data(m)
        if res:
            models_data[m] = res
            agg, layers = res
            plot_single_metric(m, agg, layers, 'kti', 'KTI Score', f'KTI Curve: {m}', out_root / f'{m.lower()}_kti_curve.pdf')
            plot_single_metric(m, agg, layers, 'rank_feat', 'Effective Rank (Features)', f'ERank Curve: {m}', out_root / f'{m.lower()}_erank_curve.pdf')
            plot_single_metric(m, agg, layers, 'dist', 'Attn Distance', f'Attention Distance: {m}', out_root / f'{m.lower()}_attn_dist.pdf')
            plot_single_metric(m, agg, layers, 'entropy', 'Attention Entropy', f'Attention Entropy: {m}', out_root / f'{m.lower()}_attn_entropy.pdf')

    if models_data:
        plot_overlay_kti(models_data, out_root / 'models_kti_overlay.pdf')
        df = generate_summary(models_data)
        df.to_csv(out_root / 'summary.csv', index=False)
        
        # LaTeX Table
        with open(out_root / 'summary_table.tex', 'w') as f:
            f.write(df.to_latex(index=False, caption='Summary of Topology Diagnostics across HMR models', label='tab:diag_summary'))

if __name__ == "__main__":
    main()
