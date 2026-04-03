import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def set_zh():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Droid Sans Fallback', 'Source Han Sans CN']
    plt.rcParams['axes.unicode_minus'] = False

def generate_robustness_plots():
    set_zh()
    results_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/outputs/ch5_eval")
    groups = ['Control', 'T2-Static-Mid', 'T2-Static-Late', 'T2-A-S-Baseline', 'T2-A-H-Baseline', 'T2-KTI-Adaptive']
    occlusions = [0.0, 0.4, 0.5]
    
    # Label Mapping
    label_map = {
        'Control': 'Control (不掩码)',
        'T2-Static-Mid': 'Static-Mid (硬切分-中)',
        'T2-Static-Late': 'Static-Late (硬切分-晚)',
        'T2-A-S-Baseline': 'Soft-Split (基准)',
        'T2-A-H-Baseline': 'Hard-Split (基准)',
        'T2-KTI-Adaptive': 'KTI-Adaptive (本章方法)'
    }
    
    # 1. Error Distribution (Violin Plot) at Occ=0.4
    all_data = []
    for g in groups:
        path = results_dir / f"robust_errors_{g}_occ0.4.npy"
        if path.exists():
            data = np.load(path)
            for d in data:
                all_data.append({'Strategy': label_map[g], 'MPJPE': d})
    
    df_occ04 = pd.DataFrame(all_data)
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Strategy', y='MPJPE', data=df_occ04, inner="quartile", palette="muted")
    plt.title("40% 深度遮挡下的误差分布对比 (反映极端情况稳定性)")
    plt.ylabel("MPJPE (mm)")
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 400) # Clip for visibility
    plt.tight_layout()
    plt.savefig(results_dir / "ch5_robustness_violin.png", dpi=300)
    plt.savefig(results_dir / "ch5_robustness_violin.pdf")
    
    # 2. Sensitivity Bar Chart (Degradation)
    stats_data = []
    for occ in occlusions:
        for g in groups:
            path = results_dir / f"robust_errors_{g}_occ{occ}.npy"
            if path.exists():
                data = np.load(path)
                stats_data.append({
                    'Strategy': label_map[g],
                    'Occlusion': f"{int(occ*100)}%",
                    'Mean': np.mean(data),
                    'Worst5%': np.mean(data[data >= np.percentile(data, 95)])
                })
    
    df_stats = pd.DataFrame(stats_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Strategy', y='Worst5%', hue='Occlusion', data=df_stats, palette="flare")
    plt.title("各策略在‘极端困难样本’(Top 5% 误差) 下的鲁棒性对比")
    plt.ylabel("Top 5% 平均误差 (mm)")
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(results_dir / "ch5_tail_robustness.png", dpi=300)
    plt.savefig(results_dir / "ch5_tail_robustness.pdf")
    
    print(f"Plots saved to {results_dir}")

if __name__ == "__main__":
    generate_robustness_plots()
