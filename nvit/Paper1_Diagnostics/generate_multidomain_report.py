import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def aggregate_results(raw_root):
    raw_root = Path(raw_root)
    all_data = []
    
    # Structure: raw/seed_{seed}/{domain}/Control/layer_metrics_Control.json
    for seed_dir in raw_root.glob("seed_*"):
        seed = int(seed_dir.name.split("_")[1])
        for domain_dir in seed_dir.iterdir():
            if not domain_dir.is_dir(): continue
            domain = domain_dir.name
            if domain in ['Robot_Arm', 'Mech_Arm']: continue
            
            metrics_path = domain_dir / "Control" / "layer_metrics_Control.json"
            if not metrics_path.exists(): continue
            
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            for layer, data in metrics.items():
                row = {
                    'Domain': domain,
                    'Seed': seed,
                    'Layer': int(layer),
                    'KTI': np.mean(data.get('kti', data.get('kmi', [0]))),
                    'ERank': np.mean(data.get('rank', data.get('erank', [0]))),
                    'Dist': np.mean(data.get('dist', [0])),
                    'Entropy': np.mean(data.get('entropy', [0]))
                }
                all_data.append(row)
                
    return pd.DataFrame(all_data)

def plot_metrics(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = ['KTI', 'ERank', 'Dist', 'Entropy']
    domains = df['Domain'].unique()
    
    # Also save to user requested path
    user_requested_path = Path("/home/yangz/NViT-master/nvit/external_models/figures")
    user_requested_path.mkdir(parents=True, exist_ok=True)
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for domain in domains:
            domain_df = df[df['Domain'] == domain]
            # Group by Layer and calculate mean/std across seeds
            agg = domain_df.groupby('Layer')[metric].agg(['mean', 'std']).reset_index()
            
            plt.plot(agg['Layer'], agg['mean'], label=domain, marker='o', markersize=4)
            plt.fill_between(agg['Layer'], agg['mean'] - agg['std'], agg['mean'] + agg['std'], alpha=0.2)
            
        plt.title(f'Layer-wise {metric} across Domains')
        plt.xlabel('Layer Index')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(output_dir / f'{metric.lower()}_curves.pdf')
        plt.savefig(output_dir / f'{metric.lower()}_curves.png')
        plt.savefig(user_requested_path / f'{metric.lower()}_curves.png')
        plt.close()

def generate_latex_table(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate global averages per domain for the table
    summary = df.groupby('Domain')[['KTI', 'ERank', 'Dist', 'Entropy']].mean().reset_index()
    # Also get max/min info for bimodal detection if possible?
    # Simple summary for now.
    latex = summary.to_latex(index=False, float_format="%.4f")
    with open(output_path, 'w') as f:
        f.write(latex)

if __name__ == "__main__":
    raw_root = 'outputs/multidomain_diag/raw'
    df = aggregate_results(raw_root)
    if not df.empty:
        plot_metrics(df, 'outputs/multidomain_diag/figures')
        generate_latex_table(df, 'outputs/multidomain_diag/tables/summary.tex')
        df.to_csv('outputs/multidomain_diag/combined_results.csv', index=False)
        print("Results generated successfully.")
    else:
        print("No data found to aggregate.")
