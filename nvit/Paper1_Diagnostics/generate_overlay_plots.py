import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Specific Requirements
MODEL_ORDER = ['HMR2', 'HSMR', 'PromptHMR', 'CameraHMR']
MODEL_LABELS = {'HMR2': 'HMR2', 'HSMR': 'HSMR', 'PromptHMR': 'ProMoHMR', 'CameraHMR': 'CameraHMR'}
MODEL_COLORS = {
    'HMR2': 'tab:blue',
    'HSMR': 'tab:orange',
    'PromptHMR': 'tab:green',
    'CameraHMR': 'tab:red'
}

def load_all_metrics(root_dir, models, seeds=[0, 1, 2]):
    all_data = {}
    for model in models:
        model_seeds_data = []
        for seed in seeds:
            seed_file = Path(root_dir) / model / f'seed_{seed}' / 'layer_metrics.json'
            if seed_file.exists():
                with open(seed_file, 'r') as f:
                    model_seeds_data.append(json.load(f))
        if model_seeds_data:
            all_data[model] = model_seeds_data
    return all_data

def process_metric_stats(model_seeds_data, metric_key):
    """Aggregates multiple seeds into mean/std arrays."""
    # Assume all seeds have same layers
    layers = sorted(model_seeds_data[0].keys(), key=lambda x: int(x))
    num_layers = len(layers)
    
    layer_means = []
    layer_stds = []
    
    for layer in layers:
        vals = []
        for seed_data in model_seeds_data:
            if layer in seed_data and metric_key in seed_data[layer]:
                vals.extend(seed_data[layer][metric_key])
        
        if vals:
            layer_means.append(np.mean(vals))
            layer_stds.append(np.std(vals))
        else:
            layer_means.append(0.0); layer_stds.append(0.0)
            
    return np.array(layer_means), np.array(layer_stds), layers

def plot_overlay(all_data, metric_key, ylabel, title_prefix, out_dir, normalized=False):
    plt.figure(figsize=(10, 6))
    
    y_max = 0
    all_peaks = []

    for model in MODEL_ORDER:
        if model not in all_data: continue
        
        means, stds, layers = process_metric_stats(all_data[model], metric_key)
        L = len(layers)
        
        if normalized:
            x = np.linspace(0, 1, L)
            xlabel = 'Normalized Depth (layer/L)'
        else:
            x = np.arange(L)
            xlabel = 'Layer Index'
            
        color = MODEL_COLORS[model]
        label = MODEL_LABELS[model]
        
        plt.plot(x, means, label=label, color=color, linewidth=2)
        plt.fill_between(x, means - stds, means + stds, color=color, alpha=0.2)
        
        y_max = max(y_max, np.max(means + stds))
        
        # Mark KTI peak if metric is kti
        if metric_key == 'kti':
            peak_idx = np.argmax(means)
            peak_x = x[peak_idx]
            plt.axvline(x=peak_x, color=color, linestyle='--', alpha=0.5)
            plt.scatter(peak_x, means[peak_idx], color=color, s=40, zorder=5)
            all_peaks.append((model, peak_idx, means[peak_idx]))

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    title = f"{title_prefix} ({'Normalized' if normalized else 'Layer Index'})"
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, loc='upper right')
    
    # Set y-axis margin
    plt.ylim(-0.02 * y_max, 1.15 * y_max)
    
    suffix = "_norm" if normalized else ""
    fn = metric_key.replace('_feat', '') if 'feat' in metric_key else metric_key
    if fn == 'rank': fn = 'erank'
    if fn == 'dist': fn = 'attn_distance'
    plt.savefig(out_dir / f"{fn}_overlay{suffix}.pdf")
    plt.savefig(out_dir / f"{fn}_overlay{suffix}.png")
    plt.close()

def main():
    root_dir = 'outputs/diagnostics'
    out_dir = Path(root_dir) / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = load_all_metrics(root_dir, MODEL_ORDER)
    if not all_data:
        print("No data found to plot!")
        return

    metrics = [
        ('kti', 'KTI Score', 'Kinematic Topology Interaction (KTI)'),
        ('rank_feat', 'Effective Rank (Features)', 'Layer-wise Effective Rank'),
        ('dist', 'Mean Attention Distance', 'Attention Distance Spectrum'),
        ('entropy', 'Attention Entropy', 'Attention Entropy / Dispersion')
    ]

    summary_rows = []

    for m_key, ylabel, title in metrics:
        # Plot both versions
        plot_overlay(all_data, m_key, ylabel, title, out_dir, normalized=False)
        plot_overlay(all_data, m_key, ylabel, title, out_dir, normalized=True)

    # Generate Summary CSV
    for model in MODEL_ORDER:
        if model not in all_data: continue
        kti_means, _, layers = process_metric_stats(all_data[model], 'kti')
        erank_means, _, _ = process_metric_stats(all_data[model], 'rank_feat')
        
        peak_layer = np.argmax(kti_means)
        peak_val = kti_means[peak_layer]
        
        summary_rows.append({
            'model_name': MODEL_LABELS[model],
            'num_layers': len(layers),
            'kti_peak_layer': peak_layer,
            'kti_peak_value': f"{peak_val:.4f}",
            'avg_erank': f"{np.mean(erank_means):.2f}"
        })
    
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / 'overlay_summary.csv', index=False)
    print(f"Overlay plots and summary saved to {out_dir}")

if __name__ == "__main__":
    main()
