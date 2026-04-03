#!/home/yangz/.conda/envs/4D-humans/bin/python
import matplotlib.pyplot as plt
import numpy as np

# Data
configs = ['Baseline', 'Mid-Heavy (Pruned)']
latency = [22.40, 18.79]  # ms (Lower is better)
error = [52.69, 152.78]   # mm (Lower is better)

x = np.arange(len(configs))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Latency (Left Axis)
rects1 = ax1.bar(x - width/2, latency, width, label='Latency (ms)', color='#1f77b4', alpha=0.8)
ax1.set_ylabel('Inference Latency (ms)', color='#1f77b4', fontweight='bold')
ax1.set_ylim(0, 30)
ax1.tick_params(axis='y', labelcolor='#1f77b4')

# Plot Error (Right Axis)
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, error, width, label='MPJPE (mm)', color='#d62728', alpha=0.8)
ax2.set_ylabel('Error MPJPE (mm)', color='#d62728', fontweight='bold')
ax2.set_ylim(0, 180)
ax2.tick_params(axis='y', labelcolor='#d62728')

# Labels and Title
ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=12, fontweight='bold')
plt.title('Experiment 1: Mid-Heavy Pruning Impact\n(Direct Pruning without Finetuning)', fontsize=14)

# Add value labels
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1, ax1)
autolabel(rects2, ax2)

fig.tight_layout()
plt.savefig('/home/yangz/NViT-master/Project_Archives/Expt_01_Mid_Heavy_Validation/result_plot.png')
print("Plot saved to /home/yangz/NViT-master/Project_Archives/Expt_01_Mid_Heavy_Validation/result_plot.png")
