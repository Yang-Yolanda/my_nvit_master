#!/home/yangz/.conda/envs/4D-humans/bin/python

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_three_stage_logic():
    base_dir = "nvit/Paper1_Diagnostics/Experiment2_KTI/results"
    
    paths = {
        "Human (HMR2)": f"{base_dir}/HMR2/layer_metrics_Control.json",
        "Robot (Panda)": f"{base_dir}/Toy_Robot_Pretrained/layer_metrics_Control.json",
        "Animal (ViTPose)": f"{base_dir}/ViTPose_Animal_AP10K/layer_metrics_Control.json"
    }
    
    colors = {"Human (HMR2)": "tab:blue", "Robot (Panda)": "tab:red", "Animal (ViTPose)": "tab:green"}
    
    plt.figure(figsize=(12, 7))
    
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"Skipping {name}, path not found: {path}")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        layers = sorted([int(l) for l in data.keys()])
        means = [np.mean(data[str(l)]['kti']) for l in layers]
        
        # Normalize for visualization of "Double Peak" trend (optional, but helps see the shape)
        # means = np.array(means) / np.max(means) 
        
        plt.plot(layers, means, marker='o', label=name, color=colors[name], linewidth=2)

    # Highlight Stages
    plt.axvspan(0, 1.5, color='gray', alpha=0.1, label='Stage 1: Part ID')
    plt.axvspan(4, 6.5, color='orange', alpha=0.1, label='Stage 2: Assembly')
    plt.axvspan(7, 11, color='blue', alpha=0.05, label='Stage 3: Global Refine')

    plt.title('KTI Double-Peak Phenomenon: The Three Stages of Reconstruction', fontsize=14)
    plt.xlabel('ViT Layer Index', fontsize=12)
    plt.ylabel('KTI Score (Structural Alignment)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    
    plt.annotate('Local Part Identification', xy=(0.5, 0.4), xytext=(1, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    plt.annotate('Topological Multi-Joint Assembly', xy=(6, 0.6), xytext=(7, 0.7),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    output_path = "/home/yangz/.gemini/antigravity/brain/813ffac6-886a-4a03-aa53-b0f91b9994a4/three_stage_kti_logic.png"
    plt.savefig(output_path, dpi=300)
    print(f"Three-Stage logic plot saved to {output_path}")

if __name__ == "__main__":
    plot_three_stage_logic()
