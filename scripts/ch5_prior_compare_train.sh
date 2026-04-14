#!/bin/bash
# scripts/ch5_prior_compare_train.sh
# "The Omega" Extreme Optimization (M0-M5)
# Squeezing every bit of A100-80GB, 1TB SHM, 30 Epochs.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
set -eo pipefail

# --- OMEGA SYSTEM OPTIMIZATIONS ---
# 1. Reduce VRAM fragmentation and enable segment expansion
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
# 2. Optimize CPU-GPU Communication
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
# 3. Transparent Huge Pages (User requested)
export MALLOC_HUGEPAGES=1
# 4. Standard Environment
export HOME=/cpfs_infra/shared/yangz
source "nvit_env.sh"
# Ensure we use the correct Python env
PYTHON_EXE="/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python"

PIDS=()
cleanup() {
  echo "[cleanup] stopping child jobs..."
  trap - SIGINT SIGTERM
  for pid in "${PIDS[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  pkill -TERM -P $$ 2>/dev/null || true
  wait 2>/dev/null || true
  exit 1
}
trap cleanup SIGINT SIGTERM

CKPT_PATH="${PROJECT_ROOT}/ckpt_ch5_base.ckpt"
OUT_ROOT="/mnt/yangz/nvit_output/ch5_prior_compare"
LOG_ROOT="/mnt/yangz/nvit_output/ch5_prior_compare"

mkdir -p $OUT_ROOT
mkdir -p $LOG_ROOT

# Training Function
train_method() {
    GPU_ID=$1
    METHOD_ID=$2
    NAME=$3
    FREEZE=$4
    PHYSICAL_BS=$5
    ACC=$6
    OVERRIDES=$7

    echo "================================================================="
    echo "🚀 OMEGA DISPATCH: $METHOD_ID ($NAME) on GPU $GPU_ID"
    echo "🔹 Freeze: $FREEZE | Physical BS: $PHYSICAL_BS | Acc: $ACC"
    
    DIR_NAME="${METHOD_ID}_${NAME}"
    OUT_DIR="${OUT_ROOT}/${DIR_NAME}"
    LOG_DIR="${LOG_ROOT}/${DIR_NAME}"

    mkdir -p $OUT_DIR
    mkdir -p $LOG_DIR
    
    rm -f "${LOG_DIR}/RUNNING" "${LOG_DIR}/FAILED" "${LOG_DIR}/DONE"

    # --- 30-EPOCH SPRINT SETTINGS ---
    NUM_WORKERS=8        # Total 24 workers for 6 jobs (Maximum Stability)
    PREFETCH=4           
    MAX_EPOCHS=30
    SCALED_LR=$7         # Passed from call
    OVERRIDES=$8

    local cmd_args=(
        "experiment=hmr_vit_transformer"
        "data=full_ext"
        "++DATASETS_CONFIG_FILE=datasets_full_ext.yaml"
        "++trainer.max_epochs=${MAX_EPOCHS}"
        "++TRAIN.BATCH_SIZE=${PHYSICAL_BS}"
        "++TRAIN.ACCUMULATE_GRAD_BATCHES=${ACC}"
        "++TRAIN.LR=${SCALED_LR}"
        "++GENERAL.NUM_WORKERS=${NUM_WORKERS}"
        "++GENERAL.PREFETCH_FACTOR=${PREFETCH}"
        "++FREEZE_DEPTH=${FREEZE}"
        "++trainer.devices=1"
        "++trainer.precision=bf16-mixed"
        "++MODEL.BACKBONE.USE_CHECKPOINT=True"
        "++FINETUNE_FROM=${CKPT_PATH}"
        "++paths.output_dir=${OUT_DIR}"
        "++GENERAL.task_name=ch5_${DIR_NAME}"
    )

    eval "local extra_args=($OVERRIDES)"
    cmd_args+=("${extra_args[@]}")

    (
        set +e
        echo "start_utc=$(date -u +%F_%T) host=$(hostname) gpu=${GPU_ID}" > "${LOG_DIR}/RUNNING"
        
        # Standard redirection, no nohup inside subshell
        CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_EXE" -u nvit/train_guided.py "${cmd_args[@]}" > "${LOG_DIR}/train.log" 2>&1
        rc=$?

        rm -f "${LOG_DIR}/RUNNING"
        if [ $rc -eq 0 ]; then
            echo "DONE $(date -u +%F_%T)" > "${LOG_DIR}/DONE"
        else
            echo "FAILED (rc=$rc) $(date -u +%F_%T)" > "${LOG_DIR}/FAILED"
        fi
    ) &
    # Staggered startup (20s) to prevent system overload
    sleep 20
    PIDS+=($!)
}

# --- GROUP CONFIGURATION (30-Epoch Sprint) ---
# Goal: ACC=1, MAX_BS, SCALED_LR
# Base LR=1e-5 @ BS=48. Scaling rule: LR = 1e-5 * (BS/48)
L8_PL="[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]"
M4_MODES=""
for i in {8..10}; do M4_MODES+="++MASK_CONFIG.layer_modes.$i=soft "; done
for i in {11..31}; do M4_MODES+="++MASK_CONFIG.layer_modes.$i=hard "; done

# Dispatch M0-M5
# M0: Freeze 8, BS 384, LR 8e-5
train_method 0 "M0" "NoMask" 8 384 1 8.0e-5 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=none"
# M1: Freeze 16, BS 640, LR 1.33e-4
train_method 1 "M1" "Pos16" 16 640 1 1.33e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=[16]"
# M2: Freeze 24, BS 768, LR 1.6e-4
train_method 2 "M2" "Pos24" 24 768 1 1.6e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=[24]"
# M3-M5: Masking starting at 8, Freeze 8, BS 512, LR 1.06e-4
train_method 3 "M3" "8PlusSoft" 8 512 1 1.06e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=${L8_PL}"
train_method 4 "M4" "AdaptiveKTI" 8 512 1 1.06e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=adaptive ++MASK_CONFIG.mask_layers=${L8_PL} ${M4_MODES}"
train_method 5 "M5" "8PlusHard" 8 512 1 1.06e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=hard ++MASK_CONFIG.mask_layers=${L8_PL}"

echo "🏁 Omega Groups M0-M5 dispatched. Monitoring performance..."
wait
echo "✨ All Omega experiments completed!"
