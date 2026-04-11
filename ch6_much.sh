#!/bin/bash

# =========================================================================
# 自动化评估：针对特定 Run 的最新 10 个 Checkpoints
# =========================================================================

# 1. 环境配置
cd /home/yangz/NViT-master
export PY=/home/yangz/.conda/envs/4D-humans/bin/python
export PYTHONPATH=/home/yangz/NViT-master/nvit/Code_Paper2_Implementation:/home/yangz/4D-Humans:$PYTHONPATH

# 2. 定义目标 Checkpoints 文件夹
# 这是你刚才给出的路径
CKPT_DIR="/home/yangz/NViT-master/logs/train/runs/2026-04-07_17-28-22/checkpoints"
GPU_ID=0
CHAPTER="Ch6"

# 3. 创建本次评估的输出目录
export OUTPUT_DIR="/home/yangz/NViT-master/artifacts/eval_series_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"/{logs,results}

# 4. 【核心逻辑】按时间倒序排列，取出前 10 个 .ckpt 文件
echo "🔍 正在从目录中提取最新的 10 个模型检查点..."
# ls -t 按时间排序（最新在前），grep 过滤 ckpt，head 取前 10
LATEST_CKPTS=($(ls -t "$CKPT_DIR"/*.ckpt | head -n 10))

if [ ${#LATEST_CKPTS[@]} -eq 0 ]; then
    echo "❌ 错误: 在 $CKPT_DIR 中没有找到 .ckpt 文件！"
    exit 1
fi

echo "✅ 找到以下 10 个检查点（按从新到旧排列）:"
for i in "${!LATEST_CKPTS[@]}"; do
    echo "   [$i] $(basename "${LATEST_CKPTS[$i]}")"
done
echo "--------------------------------------------------------"

# 5. 开始循环评测
for CKPT_PATH in "${LATEST_CKPTS[@]}"; do
    CKPT_NAME=$(basename "$CKPT_PATH")
    
    echo "======================================================"
    echo "🔥 正在评测检查点: $CKPT_NAME"
    echo "📍 路径: $CKPT_PATH"
    echo "======================================================"
    
    # 注意：这里调用 global_evaluator.py
    # 如果你的 evaluator 支持 --checkpoint_path 参数，请按下面写：
    # 如果它只支持 --run_path，通常它会默认读 last.ckpt，
    # 那我们可能需要稍微改一下调用方式或者确保 evaluator 能接收具体的 ckpt 路径。
    
    $PY nvit/global_evaluator.py \
        --chapter "$CHAPTER" \
        --checkpoint_path "$CKPT_PATH" \
        --gpu $GPU_ID | tee "$OUTPUT_DIR/logs/eval_${CKPT_NAME}.log"
        
    echo "✅ $CKPT_NAME 评测完成！"
    echo ""
done

# 6. 汇总结果（假设 evaluator 会更新 summary.csv）
cp /home/yangz/NViT-master/outputs/eval_global/Ch6/summary.csv "$OUTPUT_DIR/results/latest_10_summary.csv" || echo "未找到汇总数据"

echo "🎉 所有的 10 个检查点都跑完啦！结果保存在: $OUTPUT_DIR"