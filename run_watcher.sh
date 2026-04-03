#!/bin/bash
# NViT Evaluation Watcher Script
# This script polls for a new `last.ckpt` in the latest training run directory and evaluates it.
# It runs strictly on GPU 0 to avoid DDP context collisions.

# Find the latest Hydra run directory automatically
LAST_MTIME=0

while true; do
  OUTPUT_DIR=$(ls -td /home/yangz/NViT-master/logs/train/runs/*/ | head -1 | tr -d '\n')
  CKPT="${OUTPUT_DIR}checkpoints/last.ckpt"
  echo "[Watcher] Polling for Checkpoints in: $OUTPUT_DIR"
  
  if [ -f "$CKPT" ]; then
    CURRENT_MTIME=$(stat -c %Y "$CKPT")
    if [ "$CURRENT_MTIME" -gt "$LAST_MTIME" ]; then
      echo "[Watcher] 检测到新 CKPT ($CURRENT_MTIME)，丢入后端评测..."
      
      # 限制单卡执行，绝对不碰 DDP 集群
      CUDA_VISIBLE_DEVICES=0 python /home/yangz/4D-Humans/eval.py \
        --dataset h36m-p2 \
        --checkpoint "$CKPT" \
        --results_file "${OUTPUT_DIR}eval_h36m.csv" \
        > "${OUTPUT_DIR}eval.log" 2>&1
        
      LAST_MTIME=$CURRENT_MTIME
      echo "[Watcher] 评测结束。结果已写入 ${OUTPUT_DIR}eval_h36m.csv"
    fi
  fi
  sleep 600
done
