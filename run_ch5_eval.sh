#!/bin/bash

# =========================================================================
# Automated Global Evaluation for Ch5 Mask/Prior Models (M0 - M6)
# =========================================================================

# Ensure we are in the correct root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/nvit_env.sh"
cd "$PROJECT_ROOT"

# Define the target GPU and Chapter
GPU_ID=0
CHAPTER="Ch5"

# ... (MODEL_PATHS logic)

export PY=${CONDA_PREFIX}/bin/python
if [ ! -f "$PY" ]; then export PY=python; fi

export PYTHONPATH=${PROJECT_ROOT}/nvit/Code_Paper2_Implementation:${HUMANS_ROOT}:$PYTHONPATH

# Ensure OUTPUT_DIR exists
if [ -z "${OUTPUT_DIR:-}" ]; then
    export OUTPUT_DIR="${PROJECT_ROOT}/artifacts/nvit_eval_$(date +%Y%m%d_%H%M%S)"
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
