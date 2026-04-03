import json
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

REQUIRED_GROUPS = [
    "Control",
    "T2-A-H-Baseline",
    "T2-A-S-Baseline",
    "T2-Static-Mid",
    "T2-Static-Late",
    "T2-KTI-Adaptive"
]

def process_single_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    layers = sorted([int(k) for k in data.keys()])
    
    out_metrics = {}
    
    # 2.3 指标字段一致性检查
    expected_keys = ['kti', 'entropy', 'erank', 'dist']
    sample_layer = data[str(layers[0])]
    
    # Handle legacy 'rank' vs 'erank'
    actual_keys = list(sample_layer.keys())
    if 'rank' in actual_keys and 'erank' not in actual_keys:
        # User's old files use 'rank'
        expected_keys[2] = 'rank'
        
    missing_keys = [k for k in expected_keys if k not in sample_layer]
    if missing_keys:
        top_keys = list(sample_layer.keys())
        raise RuntimeError(f"JSON {json_path} missing expected metric keys: {missing_keys}. Top-level keys found: {top_keys}")

    for m in expected_keys:
        mapped_m = 'erank' if m == 'rank' else m
        out_metrics[mapped_m] = []
        for l in layers:
            val = data[str(l)].get(m, 0.0)
            if isinstance(val, dict):
                out_metrics[mapped_m].append(val.get('mean', 0.0))
            elif isinstance(val, list):
                out_metrics[mapped_m].append(np.mean(val) if len(val)>0 else 0.0)
            else:
                out_metrics[mapped_m].append(float(val))
    return layers, out_metrics

def plot_thesis_curves(json_path, output_dir, model_name="HMR2"):
    """Legacy single-file plot (unchanged logic)"""
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not json_path.exists():
        print(f"Metrics file missing: {json_path}")
        return
        
    layers, metrics = process_single_json(json_path)
    entropy_means = metrics['entropy']
    kti_means = metrics['kti']
    
    sns.set_style("whitegrid")
    
    # 1. Plot KTI
    plt.figure(figsize=(10, 6))
    plt.plot(layers, kti_means, marker='o', color='purple', linewidth=2.5, markersize=8)
    plt.title(f'Kinematic Topology Indicator (KTI) across Layers - {model_name}', fontsize=16)
    plt.xlabel('ViT Layer Index', fontsize=14)
    plt.ylabel('Soft KTI Score', fontsize=14)
    
    plt.axvspan(5, 10, color='purple', alpha=0.1, label='Structural Bottleneck')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_KTI_Curve.png', dpi=300)
    plt.close()
    
    # 2. Phase Transition
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('ViT Layer Index', fontsize=14)
    ax1.set_ylabel('Attention Entropy', color=color, fontsize=14)
    ax1.plot(layers, entropy_means, marker='s', color=color, linewidth=2, label='Entropy')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:purple'
    ax2.set_ylabel('Soft KTI', color=color, fontsize=14)  
    ax2.plot(layers, kti_means, marker='o', color=color, linewidth=2, label='KTI')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Attention Entropy vs KTI Phase Transition - {model_name}', fontsize=16)
    fig.tight_layout()  
    plt.savefig(output_dir / f'{model_name}_Phase_Transition.png', dpi=300)
    plt.close()
    
    print(f"Legacy Plots successfully generated in {output_dir}")

def check_overlap_sanity(experiments, prefix_n, eps):
    print(f"\n--- Overlap Sanity Check (First {prefix_n} layers) ---")
    warnings = []
    
    for l in range(prefix_n):
        kti_vals = []
        for g, data in experiments.items():
            # some experiments might have shorter length unexpectedly, though we check it later
            if l < len(data['metrics']['kti']):
                kti_vals.append(data['metrics']['kti'][l])
        
        if kti_vals:
            diff = max(kti_vals) - min(kti_vals)
            if diff > eps:
                warn_msg = f"! WARNING: Layer {l} KTI divergence {diff:.6f} > {eps}. Data may not be from the same base weights."
                print(warn_msg)
                warnings.append(warn_msg)
            else:
                print(f"Layer {l} KTI divergence {diff:.6f} (OK)")
    
    if warnings:
        print(">> Sanity check completed with MINOR WARNINGS. Continuing plot...")
    else:
        print(">> Sanity check PASSED perfectly. All groups overlap as expected in prefix layers.")
    return warnings

def plot_clearer_style_batch(exp_dir, output_dir, args):
    exp_dir = Path(exp_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = {}
    manifest = {}
    
    # 1. 扫描文件
    for json_file in exp_dir.rglob('layer_metrics*.json'):
        group_name = json_file.parent.name
        
        # 针对 6 组滴定过滤
        if args.only_6_titration:
            # check if group_name matches any required group
            matched = [rg for rg in REQUIRED_GROUPS if rg in group_name or group_name in rg]
            if not matched:
                continue
            group_name = matched[0] # normalize name
            
        try:
            layers, metrics = process_single_json(json_file)
        except Exception as e:
            raise RuntimeError(f"Error reading JSON {json_file}: {e}")
            
        experiments[group_name] = {'layers': layers, 'metrics': metrics, 'path': str(json_file)}
        manifest[group_name] = {
            'json_path': str(json_file),
            'num_layers': len(layers)
        }
        
    if not experiments:
        print(f"No experimennts found in {exp_dir}!")
        return
        
    # 1.5 确保 6 组完整
    if args.only_6_titration:
        missing = [g for g in REQUIRED_GROUPS if g not in experiments]
        if missing:
            raise RuntimeError(f"Missing required titration groups: {missing}. Found: {list(experiments.keys())}")
        
    # 2.2 层数一致性检查
    num_layers_set = set(data['num_layers'] for data in manifest.values())
    if len(num_layers_set) > 1:
        err_msg = "Mismatch in number of layers across experiments:\n"
        for g, data in manifest.items():
            err_msg += f"- {g}: {data['num_layers']} layers\n"
        raise RuntimeError(err_msg)
        
    # 2.1 打印前5层值确认
    print("\n" + "="*50)
    for g, data in experiments.items():
        print(f"Exp: {g}\nPath: {data['path']}")
        for m in ['kti', 'erank', 'entropy', 'dist']:
            vals = data['metrics'][m][:5]
            print(f"  {m}[0..4]: {[round(v, 5) for v in vals]}")
        print("-" * 30)

    # 2.4 早期层重叠 Sanity Check
    if args.expect_overlap_prefix_layers > 0:
        warnings = check_overlap_sanity(experiments, args.expect_overlap_prefix_layers, args.overlap_sanity_eps)
        manifest['_sanity_warnings'] = warnings

    # Save manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    # 3. 画图
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    metric_keys = ['kti', 'erank', 'entropy', 'dist']
    metric_titles = ['KTI Score', 'Effective Rank', 'Attention Entropy', 'Effective Distance (ED)']
    
    if args.only_6_titration:
        plot_order = REQUIRED_GROUPS
    else:
        plot_order = sorted(list(experiments.keys()))
        
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(plot_order))))
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for ax_idx, (m_key, m_title) in enumerate(zip(metric_keys, metric_titles)):
        ax = axes[ax_idx]
        
        for i, group_name in enumerate(plot_order):
            if group_name not in experiments: continue
            data = experiments[group_name]
            layers = np.array(data['layers'])
            y = np.array(data['metrics'][m_key])
            if len(y) == 0: continue
            
            # B: 描边 Path Effects
            path_effects = [pe.Stroke(linewidth=args.lw + 2.5, foreground='white'), pe.Normal()]
            marker = markers[i % len(markers)]
            
            ax.plot(layers, y, label=group_name, color=colors[i], 
                    lw=args.lw, alpha=args.alpha, marker=marker, 
                    markevery=args.marker_every, markersize=8,
                    path_effects=path_effects)
            
        ax.set_title(m_title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel(m_title, fontsize=12)
        
        # 网格淡化
        ax.grid(True, which='both', alpha=0.25, ls='--')
        
    # Legend 右侧
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=12, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 0.98, 1])
    
    if args.only_6_titration:
        out_prefix = output_dir / "layerwise_compare__titration6"
    else:
        out_prefix = output_dir / "layerwise_compare__all_metrics"
        
    plt.savefig(f"{out_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nBatch enhanced plots generated at: {out_prefix}.pdf")
    print(f"Saved manifest to: {output_dir}/manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 0. Legacy
    parser.add_argument('--json', type=str, default="", help="Path to layer_metrics_Control.json")
    parser.add_argument('--out', type=str, default="outputs/plots", help="Output directory")
    parser.add_argument('--name', type=str, default="HMR2", help="Model Name")
    
    # 1. New Visual Styles & Overlap processing
    parser.add_argument('--style', type=str, choices=['legacy', 'clearer'], default='legacy')
    parser.add_argument('--exp_dir', type=str, default="", help="Directory containing multiple experiment runs")
    parser.add_argument('--marker_every', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--lw', type=float, default=2.0)
    parser.add_argument('--overlap_eps', type=float, default=1e-3, help="Overlap testing parameter (internal)")
    
    # 2. Sanity Checks & 6-Titration Filter
    parser.add_argument('--only_6_titration', action='store_true', help="Only plot the 6 specified titration groups")
    parser.add_argument('--expect_overlap_prefix_layers', type=int, default=0, help="Number of layers to check for KTI overlap (Sanity Check)")
    parser.add_argument('--overlap_sanity_eps', type=float, default=1e-3, help="Tolerance for early layer overlap")
    
    args = parser.parse_args()
    
    if args.exp_dir and args.style == 'clearer':
        plot_clearer_style_batch(args.exp_dir, args.out, args)
    elif args.json:
        # Default legacy behavior
        plot_thesis_curves(args.json, args.out, model_name=args.name)
    else:
        print("Please provide either --json (legacy) or --exp_dir (batch) with --style clearer.")
