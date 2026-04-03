#!/bin/bash

# Paper 1 Fast-Track Training Monitor
# Monitors 6 ablation groups on GPUs 0-5

echo "📊 Paper 1 Fast-Track Training Monitor"
echo "========================================"
echo ""

# Training groups
GROUPS=("T2-KTI-Adaptive" "T2-Static-Mid" "T2-Static-Late" "T2-A-H-Baseline" "T2-A-S-Baseline" "Control")
LOGS=("hero_adaptive" "comp_mid" "comp_late" "abl_AH" "abl_AS" "control")
OUTPUT_DIR="/mnt/ssd_samsung_1/home/nkd/yangz_data/nvit_output/paper1_fast_track"

# GPU Status
echo "🖥️  GPU Status:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
  awk -F', ' '{printf "  GPU %s: %3s%% util, %5s/%5s MB\n", $1, $2, $3, $4}'
echo ""

# Training Progress
echo "🚀 Training Progress:"
for i in "${!GROUPS[@]}"; do
    group="${GROUPS[$i]}"
    log="${LOGS[$i]}"
    
    # Get latest epoch/iteration from log
    if [ -f "logs/${log}.log" ]; then
        latest=$(tail -n 20 "logs/${log}.log" | grep -oP 'Epoch: \[\d+\].*?\[\s*\d+/\d+\]' | tail -n 1)
        if [ -n "$latest" ]; then
            echo "  ✓ $group: $latest"
        else
            echo "  ⏳ $group: Initializing..."
        fi
    else
        echo "  ❌ $group: No log found"
    fi
done
echo ""

# Output Directory Status
echo "💾 Output Status:"
for group in "${GROUPS[@]}"; do
    dir="$OUTPUT_DIR/$group"
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        files=$(ls "$dir" 2>/dev/null | wc -l)
        echo "  $group: $size ($files files)"
    else
        echo "  $group: Not created yet"
    fi
done
echo ""

# Occlusion Dataset
echo "🎭 Occlusion Dataset:"
if [ -f "/home/yangz/4D-Humans/data/3DPW_OCC/occlusion_subset.pkl" ]; then
    echo "  ✓ 3DPW-OCC prepared (22 samples, 62.9% ratio)"
else
    echo "  ❌ Not prepared"
fi
echo ""

# Estimated Time
echo "⏱️  Estimated Completion:"
echo "  Per epoch: ~2 hours (135k subset)"
echo "  Total (15 epochs): ~30 hours"
echo "  Expected finish: $(date -d '+30 hours' '+%Y-%m-%d %H:%M')"
