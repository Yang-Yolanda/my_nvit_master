#!/home/yangz/.conda/envs/4D-humans/bin/python
import matplotlib.pyplot as plt
import json
import numpy as np

with open('plot_data.json', 'r') as f:
    data = json.load(f)

sim = data['sim']
l1 = data['l1']
blocks = np.arange(len(l1))

# Set professional style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.axisbelow'] = True

fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)

# Plot Similarity
color_sim = '#2c3e50'
ax1.set_xlabel('ViT Block Index', fontsize=12, fontweight='bold')
ax1.set_ylabel('Inter-layer Cosine Similarity', color=color_sim, fontsize=12, fontweight='bold')
line1 = ax1.plot(blocks[:-1], sim, marker='o', markersize=6, linewidth=2.5, color=color_sim, label='Cosine Similarity', alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color_sim)
ax1.set_ylim(0.75, 1.01)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot L1 Norm
ax2 = ax1.twinx()
color_l1 = '#e74c3c'
ax2.set_ylabel('L1-Norm Importance (MLP)', color=color_l1, fontsize=12, fontweight='bold')
line2 = ax2.plot(blocks, l1, marker='s', markersize=6, linewidth=2.5, color=color_l1, linestyle='--', label='L1-Norm Importance', alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color_l1)
ax2.set_ylim(0, 50)

# Highlight Water Zone (Blocks 8-23)
ax1.axvspan(8, 23, color='#3498db', alpha=0.15, label='Redundancy "Water" Zone')
ax1.text(15.5, 0.85, 'WATER ZONE', horizontalalignment='center', fontsize=14, fontweight='bold', color='#2980b9')

# Annotate peaks
max_sim_idx = np.argmax(sim)
ax1.annotate(f'Peak Sim: {sim[max_sim_idx]:.4f}', xy=(max_sim_idx, sim[max_sim_idx]), xytext=(max_sim_idx+1, sim[max_sim_idx]-0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.title('HMR2 ViT-H Structural Redundancy Map: The Water Layer Effect', fontsize=16, fontweight='bold', pad=20)
fig.tight_layout()

# Save both PNG and SVG for reports
plt.savefig('hmr2_redundancy_map.png', bbox_inches='tight')
print("✅ Research plot generated: hmr2_redundancy_map.png")
