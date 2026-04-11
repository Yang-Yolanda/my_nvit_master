#!/bin/bash
# scripts/ch5_prior_compare_train.sh
# Parallel Training Script for Chapter 5 External Paradigm Comparison (M0-M6)

# Project root detection
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
set -euo pipefail

PIDS=()

cleanup() {
  echo "[cleanup] killing child jobs..."
  trap - SIGINT SIGTERM
  for pid in "${PIDS[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  # 兜底：杀掉本脚本的直接子进程（避免漏网）
  pkill -TERM -P $$ 2>/dev/null || true
  wait 2>/dev/null || true
  exit 1
}
trap cleanup SIGINT SIGTERM

# CKPT_PATH="/home/yangz/NViT-master/logs/train/runs/2026-01-21_15-28-28/checkpoints/last.ckpt"
# Use tilde for home directory to avoid hardcoding the username
CKPT_PATH="${HOME}/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
OUT_ROOT="output/ch5_prior_compare"
LOG_ROOT="logs/ch5_prior_compare"

mkdir -p $OUT_ROOT
mkdir -p $LOG_ROOT

# Function to run training for a single method on a specific GPU
train_method() {
    GPU_ID=$1
    METHOD_ID=$2
    NAME=$3
    OVERRIDES=$4

    echo "================================================================="
    echo "▶️  Starting Training for $METHOD_ID ($NAME) on GPU $GPU_ID"
    
    DIR_NAME="${METHOD_ID}_${NAME}"
    OUT_DIR="${OUT_ROOT}/${DIR_NAME}"
    LOG_DIR="${LOG_ROOT}/${DIR_NAME}"
    
    mkdir -p $OUT_DIR
    mkdir -p $LOG_DIR
    
    # ----------------------------
    # Resume detection + RUNNING/HEARTBEAT sentinel
    # ----------------------------

    # Skip if already DONE (strict contract)
    if [ -f "${LOG_DIR}/DONE" ]; then
        echo "SKIP: ${DIR_NAME} already DONE."
        return 0
    fi

    RESUME_CKPT="${OUT_DIR}/checkpoints/last.ckpt"
    CKPT_OVERRIDE="++ckpt_path=null"
    FINETUNE_OVERRIDE="++FINETUNE_FROM=${CKPT_PATH}"
    RESUME_MODE="cold_start_no_ckpt"

    # Only attempt resume if last.ckpt exists and is readable
    if [ -f "${RESUME_CKPT}" ]; then
        if python - <<PY "${RESUME_CKPT}" >/dev/null 2>&1
import sys, torch
p=sys.argv[1]
torch.load(p, map_location="cpu")
print("ok")
PY
        then
            CKPT_OVERRIDE="++ckpt_path=${RESUME_CKPT}"
            FINETUNE_OVERRIDE="++FINETUNE_FROM=null"
            RESUME_MODE="resume_from_last"
        else
            # Corrupt ckpt: do not pass ckpt_path; cold start
            CKPT_OVERRIDE="++ckpt_path=null"
            FINETUNE_OVERRIDE="++FINETUNE_FROM=${CKPT_PATH}"
            RESUME_MODE="cold_start_corrupt_last"
        fi
    fi

    echo "${RESUME_MODE}" > "${LOG_DIR}/RESUME_MODE"

    # Save the exact command to cmd.txt
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python nvit/train_guided.py experiment=hmr_vit_transformer data=full_ext \
        ++DATASETS_CONFIG_FILE=datasets_full_ext.yaml \
        ++trainer.max_epochs=15 \
        ++TRAIN.BATCH_SIZE=256 \
        ++TRAIN.ACCUMULATE_GRAD_BATCHES=8 \
        ++trainer.devices=1 ++trainer.precision=bf16-mixed \
        ${FINETUNE_OVERRIDE} \
        ${CKPT_OVERRIDE} \
        ++paths.output_dir=${OUT_DIR} \
        ++GENERAL.task_name=ch5_${DIR_NAME} \
        ${OVERRIDES}"

    echo "$CMD" > "${LOG_DIR}/cmd.txt"

    # Run in parallel, redirect output
    (
        set +e

        # RUNNING sentinel (objective evidence)
        echo "start_utc=$(date -u +%F_%T) host=$(hostname) gpu=${GPU_ID} pid=$$ out_dir=${OUT_DIR} resume_mode=${RESUME_MODE}" > "${LOG_DIR}/RUNNING"

        # HEARTBEAT (survives SIGKILL as a stale file pattern)
        ( while true; do date -u +%F_%T > "${LOG_DIR}/HEARTBEAT"; sleep 300; done ) &
        HB_PID=$!

        on_signal() {
            echo "Interrupted by signal at $(date -u +%F_%T)" > "${LOG_DIR}/FAILED"
            rm -f "${LOG_DIR}/RUNNING"
            kill "${HB_PID}" 2>/dev/null || true
            exit 130
        }
        trap on_signal INT TERM HUP

        eval $CMD > "${LOG_DIR}/train.log" 2>&1
        rc=$?

        kill "${HB_PID}" 2>/dev/null || true
        rm -f "${LOG_DIR}/RUNNING"

        if [ $rc -eq 0 ]; then
            echo "Completed successfully at $(date -u +%F_%T)" > "${LOG_DIR}/DONE"
            exit 0
        else
            echo "Failed (rc=$rc) at $(date -u +%F_%T)" > "${LOG_DIR}/FAILED"
            exit $rc
        fi
    ) &
    PIDS+=($!)
}

# TODO: The overrides below for M4, M5, M6 assume that patches have been made 
# to nvit/train_guided.py / nvit2_models to support logits masking without structural replacement.

# M0: NoMask (Exp0) -> GPU 0
train_method 0 "M0" "NoMask" "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False"

# Layers to mask (32 for VitPose-Huge)
ALL_LAYERS="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]"

# M1: Ours-SoftMask (Exp3 mapped purely to soft mask logic) -> GPU 1
train_method 1 "M1" "Ours-SoftMask" "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=${ALL_LAYERS}"

# M2: Ours-HardMask (Exp5 mapped purely to hard mask logic) -> GPU 2
train_method 2 "M2" "Ours-HardMask" "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=hard ++MASK_CONFIG.mask_layers=${ALL_LAYERS}"

echo "⏳ Waiting for Batch 1 (M0, M1, M2) to complete to avoid OOM..."
wait

# M3: Ours-Adaptive (Exp4 mapped to adaptive soft->hard) -> GPU 3
train_method 3 "M3" "Ours-Adaptive" "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=adaptive ++MASK_CONFIG.mask_layers=${ALL_LAYERS}"

# M4: Prior-as-Loss -> GPU 4
train_method 4 "M4" "Prior-as-Loss" "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++TRAIN.LOSS_WEIGHTS.HEATMAP=2.0"

# M5: Hard-Adjacency-Only -> GPU 5
train_method 5 "M5" "Hard-Adjacency-Only" "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=hard ++MASK_CONFIG.mask_layers=${ALL_LAYERS} ++MASK_CONFIG.domain=skeleton"

echo "⏳ Waiting for Batch 2 (M3, M4, M5) to complete to avoid OOM..."
wait

# M6: Soft-Distance-Bias-Only -> GPU 6
train_method 6 "M6" "Soft-Distance-Bias-Only" "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=${ALL_LAYERS} ++MASK_CONFIG.domain=skeleton"

echo "🚀 M6 dispatched to GPU 6."
echo "⏳ Waiting for Batch 3 (M6) to complete..."
wait

echo "🎉 All 7 groups have completed training!"
