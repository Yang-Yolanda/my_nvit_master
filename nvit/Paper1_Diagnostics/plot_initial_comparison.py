import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def plot_comparison():
    base_dir = Path("/home/yangz/NViT-master/nvit/Paper1_Diagnostics/logs/baseline_diagnostics")
    models = ["HMR2", "HSMR"]
    
    all_data = []
    for model in models:
        csv_path = base_dir / model / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['Model'] = model
            # Take the last 'Control' row since we append
            all_data.append(df.iloc[-1:])
        else:
            print(f"Warning: {csv_path} not found")
            
    if not all_data:
        print("No data found to plot.")
        return
        
    df_combined = pd.concat(all_data)
    
    # Melting for plotting
    metrics = ["Avg_Entropy", "Avg_KTI", "Avg_Rank"]
    df_melted = df_combined.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Value")
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot 1: Bar chart of averages
    plt.subplot(1, 3, 1)
    sns.barplot(data=df_melted[df_melted['Metric'] == 'Avg_Entropy'], x='Model', y='Value')
    plt.title('Average Entropy')
    
    plt.subplot(1, 3, 2)
    sns.barplot(data=df_melted[df_melted['Metric'] == 'Avg_KTI'], x='Model', y='Value')
    plt.title('Average KTI')
    plt.ylim(0, 1.0)
    
    plt.subplot(1, 3, 3)
    sns.barplot(data=df_melted[df_melted['Metric'] == 'Avg_Rank'], x='Model', y='Value')
    plt.title('Average Rank')
    
    plt.tight_layout()
    output_path = "/home/yangz/NViT-master/nvit/Paper1_Diagnostics/initial_baseline_comparison.png"
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

    # Now let's plot layer-wise trends if available
    plt.figure(figsize=(15, 5))
    for i, metric_key in enumerate(['entropy', 'kti', 'rank']):
        plt.subplot(1, 3, i+1)
        for model in models:
            json_path = base_dir / model / "layer_metrics_Control.json"
            if json_path.exists():
                import json
                with open(json_path, 'r') as f:
                    metrics_data = json.load(f)
                    values = metrics_data.get(metric_key, [])
                    layers = range(len(values))
                    plt.plot(layers, values, label=model, marker='o', markersize=3)
        plt.title(f'Layer-wise {metric_key.capitalize()}')
        plt.xlabel('Layer Index')
        plt.legend()
        
    plt.tight_layout()
    layer_output_path = "/home/yangz/NViT-master/nvit/Paper1_Diagnostics/initial_baseline_layer_trends.png"
    plt.savefig(layer_output_path)
    print(f"Layer trends plot saved to {layer_output_path}")

if __name__ == "__main__":
    plot_comparison()
