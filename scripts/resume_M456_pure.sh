#!/bin/bash
# Dedicated resume script for M4, M5, M6 mapped to GPUs 5, 6, 7
# Explicitly uses Pure ViT configuration (no Mamba/GCN replacement)
cd /home/yangz/NViT-master || exit 1
set -euo pipefail

PIDS=()
cleanup() {
  echo "[cleanup] killing child jobs..."
  trap - SIGINT SIGTERM
  for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
  pkill -TERM -P $$ 2>/dev/null || true
  wait 2>/dev/null || true
  exit 1
}
trap cleanup SIGINT SIGTERM

CKPT_PATH="/home/yangz/NViT-master/logs/train/runs/2026-01-21_15-28-28/checkpoints/last.ckpt"
OUT_ROOT="output/ch5_prior_compare"
LOG_ROOT="logs/ch5_prior_compare"

train_method() {
    GPU_ID=$1
    METHOD_ID=$2
    NAME=$3
    OVERRIDES=$4

    echo "▶️  Starting Training for $METHOD_ID ($NAME) on GPU $GPU_ID"
    DIR_NAME="${METHOD_ID}_${NAME}"
    OUT_DIR="${OUT_ROOT}/${DIR_NAME}"
    LOG_DIR="${LOG_ROOT}/${DIR_NAME}"
    
    mkdir -p $OUT_DIR
    mkdir -p $LOG_DIR
    
    RESUME_CKPT="${OUT_DIR}/checkpoints/last.ckpt"
    CKPT_OVERRIDE="++ckpt_path=null"
    FINETUNE_OVERRIDE="++FINETUNE_FROM=${CKPT_PATH}"

    if [ -f "${RESUME_CKPT}" ]; then
        CKPT_OVERRIDE="++ckpt_path=${RESUME_CKPT}"
        FINETUNE_OVERRIDE="++FINETUNE_FROM=null"
    fi

    # Explicitly disabling mamba/gcn variant to force purely ViT
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python nvit/train_guided.py experiment=hmr_vit_transformer data=full_ext \
        ++DATASETS_CONFIG_FILE=datasets_full_ext.yaml \
        ++trainer.max_epochs=5 \
        ++TRAIN.BATCH_SIZE=8 \
        ++TRAIN.ACCUMULATE_GRAD_BATCHES=8 \
        ++GENERAL.NUM_WORKERS=8 ++GENERAL.PREFETCH_FACTOR=2 \
        ++trainer.devices=1 ++trainer.precision=bf16-mixed \
        ${FINETUNE_OVERRIDE} \
        ${CKPT_OVERRIDE} \
        ++paths.output_dir=${OUT_DIR} \
        ++GENERAL.task_name=ch5_${DIR_NAME} \
        ++MODEL.BACKBONE.switch_layer_1=32 \
        ++MODEL.BACKBONE.switch_layer_2=32 \
        ${OVERRIDES}"

    echo "$CMD" > "${LOG_DIR}/cmd_resume_pure.txt"
    echo "$CMD" > "${LOG_DIR}/cmd.txt"

    (
        set +e
        eval $CMD > "${LOG_DIR}/train_resume_pure.log" 2>&1
    ) &
    PIDS+=($!)
}

# M4 on GPU 5, M5 on GPU 6, M6 on GPU 7
train_method 5 "M4" "Prior-as-Loss" "++MODEL.BACKBONE.apply_logits_mask=False ++TRAIN.LOSS_WEIGHTS.prior_loss=1.0"
train_method 6 "M5" "Hard-Adjacency-Only" "++MODEL.BACKBONE.apply_logits_mask=True ++MODEL.BACKBONE.mask_type=hard_1hop_only"
train_method 7 "M6" "Soft-Distance-Bias-Only" "++MODEL.BACKBONE.apply_logits_mask=True ++MODEL.BACKBONE.mask_type=soft_distance_only"

echo "⏳ Waiting for pure ViT M4, M5, M6 to complete..."
wait
echo "🎉 Done!"
