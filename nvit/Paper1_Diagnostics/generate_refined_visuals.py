import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for publication
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 14, 
    'font.family': 'sans-serif',
    'axes.linewidth': 1.2,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

def plot_robustness():
    print("Generating Robustness Plot (Vector)...")
    csv_path = "/home/yangz/NViT-master/nvit/Paper1_Diagnostics/robustness_results.csv"
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # Filter groups
    groups = ['Control', 'T2-KTI-Adaptive']
    df_plot = df[df['Group'].isin(groups)].copy()
    
    # Also add T2-A-H-Baseline if available
    if 'T2-A-H-Baseline' in df['Group'].values:
        df_plot = pd.concat([df_plot, df[df['Group'] == 'T2-A-H-Baseline']])
        
    plt.figure(figsize=(9, 6))
    
    marker_map = {'Control': 'o', 'T2-KTI-Adaptive': 's', 'T2-A-H-Baseline': '^'}
    color_map = {'Control': '#7f8c8d', 'T2-KTI-Adaptive': '#e74c3c', 'T2-A-H-Baseline': '#2980b9'}
    label_map = {
        'Control': 'Control (Baseline ViT)',
        'T2-KTI-Adaptive': 'Ours (KTI-Adaptive)',
        'T2-A-H-Baseline': 'Hard-Mask Baseline'
    }
    
    for group in df_plot['Group'].unique():
        data = df_plot[df_plot['Group'] == group].sort_values('Occlusion')
        plt.plot(data['Occlusion'], data['MPJPE'], marker=marker_map.get(group, 'd'), 
                 linestyle='-', linewidth=3, markersize=10, 
                 label=label_map.get(group, group), color=color_map.get(group))
        
        # Add values with offset
        for x, y in zip(data['Occlusion'], data['MPJPE']):
            plt.text(x, y + 0.4, f"{y:.1f}", ha='center', va='bottom', fontsize=12, fontweight='bold', color=color_map.get(group))

    plt.xlabel('Occlusion Ratio', fontsize=16)
    plt.ylabel('MPJPE (mm) ↓', fontsize=16)
    plt.xticks([0.1, 0.2, 0.3])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, shadow=False, title='Method', fontsize=12)
    
    out_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/paper1_figures_refined")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save multiple formats
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(out_dir / f"fig_occlusion_robustness.{ext}")
    
    # Export raw data for this figure
    df_plot.to_csv(out_dir / "data_fig_robustness.csv", index=False)
    print(f"Saved to {out_dir}/fig_occlusion_robustness.[png/pdf/svg]")

def plot_ablation():
    print("Generating Ablation Plot (Vector)...")
    csv_path = "/home/yangz/NViT-master/nvit/Paper1_Diagnostics/finetune_eval_results.csv"
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    group_order = [
        "Control",
        "T2-Static-Late",
        "T2-Static-Mid",
        "T2-A-S-Baseline",
        "T2-KTI-Adaptive",
        "T2-A-H-Baseline"
    ]
    
    df = df[df['Group'].isin(group_order)]
    df['Group'] = pd.Categorical(df['Group'], categories=group_order, ordered=True)
    df = df.sort_values('Group')
    
    plt.figure(figsize=(11, 7))
    
    colors = []
    for g in df['Group']:
        if g == 'Control': colors.append('#95a5a6')
        elif g == 'T2-KTI-Adaptive': colors.append('#c0392b')
        elif 'H-Baseline' in g: colors.append('#2980b9')
        else: colors.append('#3498db')
        
    ax = sns.barplot(data=df, x='Group', y='MPJPE', palette=colors, edgecolor='0.2', linewidth=1.5)
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 10), 
                   textcoords = 'offset points',
                   fontsize=12, fontweight='bold')

    plt.ylabel('MPJPE (mm) ↓', fontsize=16)
    plt.xlabel('Masking Interventions', fontsize=16)
    plt.xticks(rotation=15, ha='right')
    plt.ylim(82, 89)
    # Remove top/right spines
    sns.despine()
    
    out_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/paper1_figures_refined")
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(out_dir / f"fig_ablation_mpjpe.{ext}")
    
    # Export raw data
    df.to_csv(out_dir / "data_fig_ablation.csv", index=False)
    print(f"Saved to {out_dir}/fig_ablation_mpjpe.[png/pdf/svg]")

if __name__ == "__main__":
    plot_robustness()
    plot_ablation()
