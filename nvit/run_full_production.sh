#!/bin/bash

# NViT "Scientific Seed" Fast-Track Launcher
# Strategy: 135k Samples (15%) x 15 Epochs
# Goal: 3x speedup with high resolution learning curves.

echo "🚀 Launching FAST-TRACK Blitz: 6 Groups"

cd /home/yangz/NViT-master/nvit

# Kill existing to be safe
pkill -f "finetune_dense.py"
sleep 2

PYTHON=/home/yangz/.conda/envs/4D-humans/bin/python

# Optimized config for Fast Track
SUBSET=135000
EPOCHS=15
BS=12

# Clear logs
for log in hero_adaptive comp_mid comp_late abl_AH abl_AS control; do
    echo "=== FAST-TRACK RUN STARTED $(date) ===" > logs/${log}.log
done

# Launch loop
groups=(
    "T2-KTI-Adaptive:hero_adaptive:0"
    "T2-Static-Mid:comp_mid:1"
    "T2-Static-Late:comp_late:2"
    "T2-A-H-Baseline:abl_AH:3"
    "T2-A-S-Baseline:abl_AS:4"
    "Control:control:5"
)

for item in "${groups[@]}"; do
    IFS=":" read -r name log gpu <<< "$item"
    echo "GPU $gpu: $name -> logs/$log.log"
    
    nohup env PYTHONPATH=.. $PYTHON skills/train_ablation/orchestrate.py \
        --target_groups "$name" \
        --gpu $gpu \
        --epochs $EPOCHS \
        --batch-size $BS \
        --lr 1e-05 \
        --output_dir /home/yangz/NViT-master/nvit/Paper1_Diagnostics/Experiment3_Masking/results \
        --extra_args "--fast-track --subset-size $SUBSET" \
        > logs/$log.log 2>&1 &
done

echo "✅ All 6 groups launched in FAST-TRACK mode."
echo "⏱️  Estimated Trend Visible in: 2 hours."
echo "⏱️  Estimated Completion: 10-12 hours."
echo "📊 Monitor: ./status_dashboard.sh"
