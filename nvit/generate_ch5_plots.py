import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.font_manager as fm

# Setup Chinese support
font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
zh_font = fm.FontProperties(fname=font_path)

# Set style
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['axes.unicode_minus'] = False 

def main():
    results_dir = Path("outputs/ch5_eval")
    csv_path = results_dir / "ch5_supplemental_results.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return
        
    df = pd.read_csv(csv_path)
    # Ensure Group is sorted correctly
    df['Group'] = pd.Categorical(df['Group'], categories=['Exp0', 'Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5'], ordered=True)
    df = df.sort_values('Group')
    
    # helper for chinese text
    def set_zh(ax, title, xlabel, ylabel):
        ax.set_title(title, fontproperties=zh_font, fontsize=16)
        ax.set_xlabel(xlabel, fontproperties=zh_font, fontsize=14)
        ax.set_ylabel(ylabel, fontproperties=zh_font, fontsize=14)

    # 1. Mean with Error Bars
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette('viridis', n_colors=len(df))
    ax = sns.barplot(x='Group', y='Mean MPJPE', data=df, hue='Group', palette=colors, legend=False)
    plt.errorbar(x=range(len(df)), y=df['Mean MPJPE'], yerr=df['Std MPJPE'], fmt='none', c='black', capsize=5)
    set_zh(ax, '各实验组平均误差 (MPJPE) 对比', '实验组 (Exp0-Exp5)', '平均误差 (mm)')
    plt.tight_layout()
    plt.savefig(results_dir / 'ch5_plot1_mean_error.png', dpi=300)
    plt.close()
    
    # 2. Error Distribution (Violin)
    all_data = []
    groups = df['Group'].tolist()
    for g in groups:
        path = results_dir / f"errors_{g}.npy"
        if path.exists():
            data = np.load(path)
            all_data.append(pd.DataFrame({'Group': [g]*len(data), 'Error': data}))
            
    if all_data:
        df_long = pd.concat(all_data)
        df_long['Group'] = pd.Categorical(df_long['Group'], categories=['Exp0', 'Exp1', 'Exp2', 'Exp3', 'Exp4', 'Exp5'], ordered=True)
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Group', y='Error', data=df_long, hue='Group', palette='muted', inner='quartile', legend=False)
        set_zh(ax, '误差分布稳定性对比 (小提琴图)', '实验组', 'MPJPE (mm)')
        plt.ylim(0, 250)
        plt.tight_layout()
        plt.savefig(results_dir / 'ch5_plot2_error_stability.png', dpi=300)
        plt.close()

    # 3. Extreme Cases
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Group', y='Extreme Mean (Top 5%)', data=df, hue='Group', palette='Reds_d', legend=False)
    set_zh(ax, '极端情况 (Top 5% 困难样本) 表现对比', '实验组', '平均误差 (mm)')
    plt.tight_layout()
    plt.savefig(results_dir / 'ch5_plot3_extreme_cases.png', dpi=300)
    plt.close()
    
    print(f"Plots saved to {results_dir}")

if __name__ == "__main__":
    main()
