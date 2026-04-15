#!/bin/bash
# scripts/ch5_prior_compare_train.sh - Rewritten for DLC "Best Practice"
# Reading from CPFS, Writing to OSS.

# --- 1. DYNAMIC ENVIRONMENT DETECTION (Sync with run_dlc_24gpu.sh) ---
PROJ_ROOT="/cpfs_infra/shared/yangz/NViT-master"
PARENT_DIR="$(dirname "$PROJ_ROOT")"
cd "$PROJ_ROOT" || exit 1

# [NEW] Automated Environment Setup: Ensure local cache symlink exists for 4D-Humans
SHARED_CACHE="/cpfs_infra/shared/yangz/.cache/4DHumans"
if [ ! -L "$HOME/.cache/4DHumans" ]; then
    echo "🔗 Creating symlink: $HOME/.cache/4DHumans -> $SHARED_CACHE"
    mkdir -p "$HOME/.cache"
    rm -rf "$HOME/.cache/4DHumans" # Remove if it's a dead link or directory
    ln -s "$SHARED_CACHE" "$HOME/.cache/4DHumans"
fi

export DATA_ROOT="$PROJ_ROOT/hmr2_training_data"
PYTHON_EXE="$PARENT_DIR/opt/Miniconda3/envs/4D-humans/bin/python"
CONFIG_DIR="$PARENT_DIR/4D-Humans/hmr2/configs_hydra"
OUT_ROOT="/cpfs_infra/shared/yangz/NViT-master/output/ch5_prior_compare"
PROD_DATASET_CONFIG="$PROJ_ROOT/scripts/datasets_tar_prod.yaml"
SMPL_DATA_DIR="$SHARED_CACHE/data/"
CKPT_PATH="$SHARED_CACHE/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"

echo "🌐 Omega Dispatch Analysis (Ch5):"
echo "🚀 Project: $PROJ_ROOT"
echo "🚀 Data:    $DATA_ROOT"
echo "🚀 Output:  $OUT_ROOT"

# --- 2. CLUSTER PERFORMANCE & MEMORY TUNING (Sync with run_dlc_24gpu.sh) ---
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
export TORCH_CUDNN_V8_API_ENABLED=1

# --- 3. DISPATCH FUNCTION ---
PIDS=()

dispatch_experiment() {
    local GPU_ID=$1
    local METHOD_ID=$2
    local NAME=$3
    local FREEZE=$4
    local PHYSICAL_BS=$5
    local ACC=$6
    local SCALED_LR=$7
    local OVERRIDES=$8

    DIR_NAME="${METHOD_ID}_${NAME}"
    OUT_DIR="${OUT_ROOT}/${DIR_NAME}"
    LOG_DIR="${OUT_ROOT}/logs/${DIR_NAME}"

    mkdir -p "$OUT_DIR"
    mkdir -p "$LOG_DIR"
    
    echo "================================================================="
    echo "🚀 DISPATCHING: $METHOD_ID ($NAME) on GPU $GPU_ID"
    echo "🔹 Out: $OUT_DIR"
    echo "================================================================="

    # --- EXECUTION ---
    # 使用 nohup 彻底脱离终端，防止 SIGHUP 和 stdout IO 错误
    # [关键修复] 必须指定 ++paths.log_dir，否则 Hydra 会让 6 个进程同时写根目录的 train_guided.log，导致 OS Errno 22 并发写入冲突！
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup "$PYTHON_EXE" -u nvit/train_guided.py \
        --config-dir "$CONFIG_DIR" \
        --config-name train \
        experiment=hmr_vit_transformer \
        data=full_ext \
        ++DATASETS_CONFIG_FILE="$PROD_DATASET_CONFIG" \
        ++trainer.max_epochs=30 \
        ++TRAIN.BATCH_SIZE=$PHYSICAL_BS \
        ++TRAIN.ACCUMULATE_GRAD_BATCHES=$ACC \
        ++TRAIN.LR=$SCALED_LR \
        ++GENERAL.NUM_WORKERS=8 \
        ++GENERAL.PREFETCH_FACTOR=4 \
        ++FREEZE_DEPTH=$FREEZE \
        ++trainer.devices=1 \
        ++trainer.precision="bf16-mixed" \
        ++MODEL.BACKBONE.USE_CHECKPOINT=False \
        ++SMPL.DATA_DIR="$SMPL_DATA_DIR" \
        ++FINETUNE_FROM="'$CKPT_PATH'" \
        ++paths.output_dir="$OUT_DIR" \
        ++paths.log_dir="$LOG_DIR" \
        ++GENERAL.task_name="ch5_${DIR_NAME}" \
        $OVERRIDES > "${LOG_DIR}/train.log" 2>&1 &
    
    PIDS+=($!)
    # Staggered startup to avoid CPFS/OSS contention
    sleep 30
}

# --- 4. EXPERIMENT GROUP M0-M5 ---
# Baseline LR=1e-5 @ BS=48. Scaling: LR = 1e-5 * (BS/48)

# M0: Freeze 8, BS 192 (Scaled LR 4.0e-5? No, user previously asked for 8.0e-5)
dispatch_experiment 0 "M0" "NoMask" 8 192 1 8.0e-5 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=none"

# M1: Freeze 16, BS 192
dispatch_experiment 1 "M1" "Pos16" 16 192 1 1.33e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=[16]"

# M2: Freeze 24, BS 192
dispatch_experiment 2 "M2" "Pos24" 24 192 1 1.6e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=[24]"

# M3-M5: Global Masks starting at 8
L8_PL="[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]"
dispatch_experiment 3 "M3" "8PlusSoft" 8 192 1 1.06e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=${L8_PL}"

# Generate M4 Adaptive modes
M4_MODES=""
for i in {8..10}; do M4_MODES+="++MASK_CONFIG.layer_modes.$i=soft "; done
for i in {11..31}; do M4_MODES+="++MASK_CONFIG.layer_modes.$i=hard "; done
dispatch_experiment 4 "M4" "AdaptiveKTI" 8 192 1 1.06e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=adaptive ++MASK_CONFIG.mask_layers=${L8_PL} ${M4_MODES}"

# M5: Freeze 8, Hard Mask
dispatch_experiment 5 "M5" "8PlusHard" 8 192 1 1.06e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=hard ++MASK_CONFIG.mask_layers=${L8_PL}"

echo "🏁 All Group Ch5 Experiments Dispatched. Monitoring..."
wait
echo "✨ All experiments finished."
