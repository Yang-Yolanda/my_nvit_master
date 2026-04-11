#!/bin/bash
# Dedicated resume script for M4, M5, M6 for a few epochs.
# Project root detection
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
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

# CKPT_PATH="${PROJECT_ROOT}/logs/train/runs/2026-01-21_15-28-28/checkpoints/last.ckpt"
CKPT_PATH="${HOME}/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
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

    # Using max_epochs=2 to run a few epochs from Epoch 0
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python nvit/train_guided.py experiment=hmr_vit_transformer data=full_ext \
        ++DATASETS_CONFIG_FILE=datasets_full_ext.yaml \
        ++trainer.max_epochs=2 \
        ++TRAIN.BATCH_SIZE=8 \
        ++TRAIN.ACCUMULATE_GRAD_BATCHES=8 \
        ++trainer.devices=1 ++trainer.precision=bf16-mixed \
        ${FINETUNE_OVERRIDE} \
        ${CKPT_OVERRIDE} \
        ++paths.output_dir=${OUT_DIR} \
        ++GENERAL.task_name=ch5_${DIR_NAME} \
        ${OVERRIDES}"

    echo "$CMD" > "${LOG_DIR}/cmd_resume.txt"

    (
        set +e
        echo "start_utc=$(date -u +%F_%T) host=$(hostname) gpu=${GPU_ID} pid=$$" > "${LOG_DIR}/RUNNING_RESUME"
        eval $CMD > "${LOG_DIR}/train_resume.log" 2>&1
        rc=$?
        rm -f "${LOG_DIR}/RUNNING_RESUME"
        if [ $rc -eq 0 ]; then
            echo "Completed successfully at $(date -u +%F_%T)" > "${LOG_DIR}/DONE_RESUME"
        else
            echo "Failed (rc=$rc) at $(date -u +%F_%T)" > "${LOG_DIR}/FAILED_RESUME"
        fi
    ) &
    PIDS+=($!)
}

# Layers to mask (32 for VitPose-Huge)
ALL_LAYERS="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]"

# Run simultaneously on GPU 4, 5, 6
train_method 4 "M4" "Prior-as-Loss" "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++TRAIN.LOSS_WEIGHTS.HEATMAP=2.0"
train_method 5 "M5" "Hard-Adjacency-Only" "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=hard ++MASK_CONFIG.mask_layers=${ALL_LAYERS} ++MASK_CONFIG.domain=skeleton"
train_method 6 "M6" "Soft-Distance-Bias-Only" "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=${ALL_LAYERS} ++MASK_CONFIG.domain=skeleton"

echo "⏳ Waiting for M4, M5, M6 to complete max_epochs=2..."
wait
echo "🎉 Done!"
