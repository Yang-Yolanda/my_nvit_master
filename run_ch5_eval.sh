#!/bin/bash

# =========================================================================
# Automated Global Evaluation for Ch5 Mask/Prior Models (M0 - M6)
# =========================================================================

# Ensure we are in the correct root directory
cd /home/yangz/NViT-master

# Define the target GPU and Chapter
GPU_ID=0
CHAPTER="Ch5"

# List all the directories containing your 6 (actually 7) experimental models
# These are your 2.8G models properly soft-linked to the external drive
# declare -a MODEL_PATHS=(
#     "output/ch5_prior_compare/M0_NoMask/"
#     "output/ch5_prior_compare/M1_Ours-SoftMask/"
#     "output/ch5_prior_compare/M2_Ours-HardMask/"
#     "output/ch5_prior_compare/M3_Ours-Adaptive/"
#     "output/ch5_prior_compare/M4_Prior-as-Loss/"
#     "output/ch5_prior_compare/M5_Hard-Adjacency-Only/"
#     "output/ch5_prior_compare/M6_Soft-Distance-Bias-Only/"
# )

declare -a MODEL_PATHS=(
    # "logs/train/runs/2026-04-07_17-28-22/"
    "logs/train/runs/2026-04-03_08-09-28"
    # "logs/train/runs/2026-04-06_07-42-42"
)


export PY=/home/yangz/.conda/envs/4D-humans/bin/python
export PYTHONPATH=/home/yangz/NViT-master/nvit/Code_Paper2_Implementation:/home/yangz/4D-Humans:$PYTHONPATH

# Ensure OUTPUT_DIR exists (it will be passed in from run_eval_suite_final.sh or created here)
if [ -z "${OUTPUT_DIR:-}" ]; then
    export OUTPUT_DIR="/home/yangz/NViT-master/artifacts/nvit_eval_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"/{logs,results,configs}
fi

echo "🚀 起飞！开始自动连续测算 ${#MODEL_PATHS[@]} 个微调模型..."
echo "--------------------------------------------------------"

# Loop through each directory and run the evaluator
for RUN_PATH in "${MODEL_PATHS[@]}"; do
    echo "======================================================"
    echo "🔍 正在测算模型: $RUN_PATH"
    echo "======================================================"
    MODEL_NAME=$(basename $RUN_PATH)
    
    # Check if the directory exists and actually contains the checkpoint folder
    if [ -d "$RUN_PATH/checkpoints" ] || [ -f "$RUN_PATH/last.ckpt" ]; then
        $PY nvit/global_evaluator.py \
            --chapter "$CHAPTER" \
            --run_path "$RUN_PATH" \
            --gpu $GPU_ID | tee "$OUTPUT_DIR/logs/ch5_${MODEL_NAME}.log"
            
        echo "✅ $RUN_PATH 测算完毕！"
    else
        echo "⚠️ 警告: 未在 $RUN_PATH 下找到有效架构 / checkpoints，将跳过！"
    fi
    echo ""
done

cp /home/yangz/NViT-master/outputs/eval_global/Ch5/summary.csv "$OUTPUT_DIR/results/ch5_summary.csv" || echo "No summary found to copy"

echo "🎉 全部的 M0-M6 模型已经连轴测算结束了！快去看你的 summary.csv 结果吧！"
