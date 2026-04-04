import os
import json
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LayerPlotter")

def generate_comparative_plots(chapter, output_base="/home/yangz/NViT-master/outputs/eval_global"):
    """
    Scans the chapter directory for run subdirectories, extracting layer_metrics_Control.json
    and overlays them on a 2x2 layer-wise diagnostic grid.
    """
    chapter_dir = Path(output_base) / chapter
    if not chapter_dir.exists():
        logger.warning(f"Chapter directory {chapter_dir} does not exist.")
        return

    # Find all layer_metrics_Control.json
    run_metrics = {}
    
    # We walk chapter_dir. Run subdirs are 1 depth deep.
    for run_dir in chapter_dir.iterdir():
        if run_dir.is_dir():
            json_file = run_dir / "diagnostics" / "layer_metrics_Control.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    try:
                        data = json.load(f)
                        run_metrics[run_dir.name] = data
                    except Exception as e:
                        logger.error(f"Failed to parse JSON for {run_dir.name}: {e}")

    if not run_metrics:
        logger.warning(f"No layer metric diagnostics found in {chapter_dir}")
        return

    # Initialize Plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'NViT {chapter} Multi-Model Layer-wise Scientific Diagnostics', fontsize=16)

    metrics_map = {
        'rank': (axs[0, 0], 'Effective Rank (Feature Collapse)'),
        'entropy': (axs[0, 1], 'Shannon Entropy (Routing Uniformity)'),
        'kmi': (axs[1, 0], 'KTI (Topological Geodesic Grounding)'),
        'dist': (axs[1, 1], 'MAD (Mean Attention/Affinity Distance)'),
    }

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (run_name, data) in enumerate(run_metrics.items()):
        color = colors[i % len(colors)]
        
        # Sort layers to ensure correct progression (0, 1, 2... 11)
        try:
            sorted_layers = sorted([int(k) for k in data.keys()])
        except ValueError:
            sorted_layers = sorted(data.keys())
            
        x_layers = [str(k) for k in sorted_layers]
        
        # Extract the mean scalar for each metric per layer
        run_plot_data = {'rank': [], 'entropy': [], 'kmi': [], 'dist': []}
        
        for k_layer in sorted_layers:
            layer_data = data[str(k_layer)]
            for m_key in run_plot_data.keys():
                vals = layer_data.get(m_key, [])
                if vals and len(vals) > 0:
                    run_plot_data[m_key].append(np.mean(vals))
                else:
                    run_plot_data[m_key].append(np.nan) # Handle missing
        
        # Plot each metric
        for m_key, (ax, title) in metrics_map.items():
            y_vals = run_plot_data[m_key]
            
            # Check if there is valid data
            if not all(np.isnan(y) for y in y_vals):
                ax.plot(x_layers, y_vals, marker='o', linewidth=2, color=color, label=run_name)
    
    # Format axes
    for ax, title in metrics_map.values():
        ax.set_title(title)
        ax.set_xlabel("Layer Depth")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Only show legend if lines exist
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save Image
    save_path = chapter_dir / "layer_metrics_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"✅ Successfully generated comparative layer plots: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Layer Metrics for a Chapter.")
    parser.add_argument("--chapter", type=str, required=True, help="Chapter name (e.g., Ch6A)")
    args = parser.parse_args()
    generate_comparative_plots(args.chapter)
