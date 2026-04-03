import numpy as np
import pandas as pd
from pathlib import Path

def analyze_robustness_distributions():
    results_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/outputs/ch5_eval")
    groups = ['Control', 'T2-Static-Mid', 'T2-Static-Late', 'T2-A-S-Baseline', 'T2-A-H-Baseline', 'T2-KTI-Adaptive']
    occlusions = [0.0, 0.4, 0.5]
    
    stats = []
    for occ in occlusions:
        for g in groups:
            path = results_dir / f"robust_errors_{g}_occ{occ}.npy"
            if not path.exists():
                print(f"Warning: File not found: {path}")
                continue
            data = np.load(path)
            
            stats.append({
                'Group': g,
                'Occlusion': occ,
                'Mean': np.mean(data),
                'Top5%_Mean': np.mean(data[data >= np.percentile(data, 95)]),
                'Top1%_Mean': np.mean(data[data >= np.percentile(data, 99)]),
                'Outliers_>150mm': (data > 150).sum(),
                'Max': np.max(data)
            })
            
    if not stats:
        print("Error: No data found to analyze.")
        return
        
    df = pd.DataFrame(stats)
    print("\n--- Robustness Statistical Analysis (Occ=0.0 vs Occ=0.4) ---")
    print(df.to_markdown(index=False))
    
    # Calculate performance drop (Sensitivity)
    p_df = df.pivot(index='Group', columns='Occlusion', values='Mean')
    if 0.0 in p_df.columns and 0.4 in p_df.columns:
        p_df['Degradation (%)'] = (p_df[0.4] - p_df[0.0]) / p_df[0.0] * 100
        print("\n--- Error Degradation Comparison ---")
        print(p_df.sort_values('Degradation (%)').to_markdown())

if __name__ == "__main__":
    analyze_robustness_distributions()
