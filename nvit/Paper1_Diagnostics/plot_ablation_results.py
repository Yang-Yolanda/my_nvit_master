import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def plot_ablation_results():
    base_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/Experiment1_Entropy/results/HMR2")
    
    # The groups we expect
    groups = [
        "Control",
        "T2-KTI-Adaptive",
        "T2-A-H-Baseline",
        "T2-A-S-Baseline",
        "T2-Static-Late",
        "T2-Static-Mid"
    ]
    
    results = []
    
    for g in groups:
        csv_path = base_dir / g / "results.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Take the last row (we append to the csv)
                latest_mpjpe = df.iloc[-1]['MPJPE']
                results.append({'Group': g, 'MPJPE': latest_mpjpe})
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        else:
            print(f"Warning: {csv_path} not found.")

    if not results:
        print("No results found to plot.")
        return
        
    df_res = pd.DataFrame(results)
    
    # Sort groups logically
    group_order = [
        "Control",
        "T2-KTI-Adaptive",
        "T2-A-S-Baseline",
        "T2-A-H-Baseline",
        "T2-Static-Late",
        "T2-Static-Mid"
    ]
    
    # Filter to only existing groups
    group_order = [g for g in group_order if g in df_res['Group'].values]
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Highlight T2-KTI-Adaptive
    colors = ['#1f77b4' if g != 'T2-KTI-Adaptive' else '#d62728' for g in group_order]
    
    ax = sns.barplot(data=df_res, x='Group', y='MPJPE', order=group_order, palette=colors)
    plt.title('Ablation Study: Impact of Masking Strategies on MPJPE (Zero-Shot 3DPW)', fontsize=14)
    plt.ylabel('MPJPE (mm, lower is better)', fontsize=12)
    plt.xlabel('Intervention Group', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for i, v in enumerate(df_res.set_index('Group').reindex(group_order)['MPJPE']):
        ax.text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to refined figures directory
    out_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/paper1_figures_refined")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Ablation_MPJPE_Comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    plot_ablation_results()
