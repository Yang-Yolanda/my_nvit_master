#!/bin/bash
# scripts/ch5_prior_compare_train.sh - Rewritten for DLC "Best Practice"
# Reading from CPFS, Writing to OSS.

# --- 1. DYNAMIC ENVIRONMENT DETECTION (Sync with run_dlc_24gpu.sh) ---
PROJ_ROOT="/cpfs_infra/shared/yangz/NViT-master"
PARENT_DIR="$(dirname "$PROJ_ROOT")"
cd "$PROJ_ROOT" || exit 1
# Hydra paths/root_dir uses ${oc.env:PROJECT_ROOT}; without this, configs or downstream code can fail early.

# Temp files (DataLoader/mp/tempfile): default /tmp sits on small rootfs and fills under parallel jobs → Errno 28.
export TMPDIR="${TMPDIR:-/cpfs_infra/shared/yangz/tmp}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"

# [NEW] Automated Environment Setup: Ensure local cache symlink exists for 4D-Humans
SHARED_CACHE="/cpfs_infra/shared/yangz/.cache/4DHumans"
if [ ! -L "$HOME/.cache/4DHumans" ]; then
    echo "🔗 Creating symlink: $HOME/.cache/4DHumans -> $SHARED_CACHE"
    mkdir -p "$HOME/.cache"
    rm -rf "$HOME/.cache/4DHumans" # Remove if it's a dead link or directory
    ln -s "$SHARED_CACHE" "$HOME/.cache/4DHumans"
fi

export PROJECT_ROOT="$PROJ_ROOT"

PROJ_ROOT="/cpfs_infra/shared/yangz/NViT-master"
export DATA_ROOT="$PROJ_ROOT/hmr2_training_data"
PYTHON_EXE="$PARENT_DIR/opt/Miniconda3/envs/4D-humans/bin/python"
CONFIG_DIR="$PARENT_DIR/4D-Humans/hmr2/configs_hydra"
OUT_ROOT="/cpfs_infra/shared/yangz/NViT-master/output/ch5_prior_compare"
DATASET_CONFIG="$PROJ_ROOT/scripts/datasets_tar.yaml"
SMPL_DATA_DIR="$SHARED_CACHE/data/"
CKPT_PATH="$SHARED_CACHE/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"

# Auto-resume: each experiment uses newest train/runs/<timestamp>/checkpoints/last.ckpt under that experiment's OUT_DIR.
# Set CH5_AUTO_RESUME=0 to always start from FINETUNE_FROM (ignore last.ckpt).
# Optional manual path (single-job only): CH5_RESUME_CKPT=/path/to.ckpt overrides auto-detection.
CH5_AUTO_RESUME="${CH5_AUTO_RESUME:-1}"
CH5_RESUME_CKPT="${CH5_RESUME_CKPT:-}"

# Stop condition: total optimizer steps only (not max_epochs). Lightning: max_epochs=-1, max_steps=CH5_MAX_TRAIN_STEPS.
# Default 90000 ≈ old "30 epochs × 3000 batches/epoch". Raise CH5_MAX_TRAIN_STEPS when resuming past that total.
# Synthetic "epoch" length (logging / tqdm) = CH5_STEPS_PER_EPOCH; same value = checkpoint save cadence.
CH5_STEPS_PER_EPOCH="${CH5_STEPS_PER_EPOCH:-3000}"
CH5_MAX_TRAIN_STEPS="${CH5_MAX_TRAIN_STEPS:-90000}"

echo "🌐 Omega Dispatch Analysis (Ch5):"
echo "🚀 Project: $PROJ_ROOT"
echo "🚀 Data:    $DATA_ROOT"
echo "🚀 Output:  $OUT_ROOT"
echo "🚀 TMPDIR:  $TMPDIR"
echo "🚀 CH5 batches per synthetic epoch + checkpoint cadence: $CH5_STEPS_PER_EPOCH"
echo "🚀 CH5 stop after optimizer steps (trainer.max_steps): $CH5_MAX_TRAIN_STEPS"
echo "🚀 CH5 auto-resume from last.ckpt: CH5_AUTO_RESUME=${CH5_AUTO_RESUME} (set to 0 to disable)"

# --- 2. CLUSTER PERFORMANCE & MEMORY TUNING (Sync with run_dlc_24gpu.sh) ---
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
export TORCH_CUDNN_V8_API_ENABLED=1

# --- 3. DISPATCH FUNCTION ---
# Newest Hydra run directory under OUT_DIR/train/runs/*/checkpoints/last.ckpt (by directory mtime order).
ch5_latest_last_ckpt() {
    local out="$1"
    local rund="${out}/train/runs"
    [ -d "$rund" ] || return 1
    local d ck
    for d in $(ls -td "$rund"/*/ 2>/dev/null); do
        ck="${d}checkpoints/last.ckpt"
        if [ -f "$ck" ] && [ -s "$ck" ]; then
            printf '%s' "$ck"
            return 0
        fi
    done
    return 1
}

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
    # Same tree as Hydra run dir (train/runs/<ts>/): checkpoints, tensorboard, .hydra — no split "logs/" root.
    LOG_DIR="$OUT_DIR"

    mkdir -p "$OUT_DIR"

    local RESUME_ARGS=""
    if [ "$CH5_AUTO_RESUME" != "0" ]; then
        local RCKPT=""
        if [ -n "$CH5_RESUME_CKPT" ]; then
            if [ -f "$CH5_RESUME_CKPT" ] && [ -s "$CH5_RESUME_CKPT" ]; then
                RCKPT="$CH5_RESUME_CKPT"
                echo "🔁 Explicit CH5_RESUME_CKPT: $RCKPT"
            else
                echo "⚠️ CH5_RESUME_CKPT set but missing/empty; falling back to auto last.ckpt"
            fi
        fi
        if [ -z "$RCKPT" ] && RCKPT=$(ch5_latest_last_ckpt "$OUT_DIR"); then
            echo "🔁 Auto-resume: $RCKPT"
        elif [ -z "$RCKPT" ]; then
            echo "🆕 No last.ckpt under ${OUT_DIR}/train/runs — training from FINETUNE_FROM pretrained."
        fi
        if [ -n "$RCKPT" ]; then
            RESUME_ARGS=" ++FINETUNE_FROM=null ++ckpt_path='${RCKPT}'"
        fi
    else
        echo "🆕 CH5_AUTO_RESUME=0 — training from FINETUNE_FROM pretrained (no ckpt_path)."
    fi
    
    echo "================================================================="
    echo "🚀 DISPATCHING: $METHOD_ID ($NAME) on GPU $GPU_ID"
    echo "🔹 Out: $OUT_DIR"
    echo "================================================================="

    # --- EXECUTION ---
    # 前台运行，stdout/stderr 经 tee 写入 train.log；长时间任务请用 tmux/screen 会话挂起。
    # [关键修复] 必须指定 ++paths.log_dir，否则 Hydra 会让多进程同时写根目录的 train_guided.log，导致 OS Errno 22 并发写入冲突。
    #
    # TRAIN_BATCHES_PER_EPOCH = Lightning limit_train_batches；CHECKPOINT_EVERY_N_TRAIN_STEPS = 存盘间隔；此处用同一 CH5_STEPS_PER_EPOCH 对齐。
    #
    # cwd=OUT_DIR：Hydra job 日志落在各实验目录，而非 NViT-master/train_guided.log。
    echo "🖥️ Foreground + tee -> ${LOG_DIR}/train.log"
    ( cd "$OUT_DIR" && CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_EXE" -u "$PROJ_ROOT/nvit/train_guided.py" \
        --config-dir "$CONFIG_DIR" \
        --config-name train \
        experiment=hmr_vit_transformer \
        data=full_ext \
        ++DATASETS_CONFIG_FILE="$DATASET_CONFIG" \
        ++GENERAL.TOTAL_STEPS="${CH5_MAX_TRAIN_STEPS}" \
        ++trainer.max_epochs=-1 \
        ++trainer.max_steps="${CH5_MAX_TRAIN_STEPS}" \
        ++TRAIN.BATCH_SIZE=$PHYSICAL_BS \
        ++TRAIN.ACCUMULATE_GRAD_BATCHES=$ACC \
        ++TRAIN.LR=$SCALED_LR \
        ++GENERAL.NUM_WORKERS=8 \
        ++GENERAL.PREFETCH_FACTOR=4 \
        ++FREEZE_DEPTH=$FREEZE \
        ++trainer.devices=1 \
        ++trainer.precision="bf16-mixed" \
        ++trainer.enable_progress_bar=False \
        ++MODEL.BACKBONE.USE_CHECKPOINT=False \
        ++SMPL.DATA_DIR="$SMPL_DATA_DIR" \
        ++FINETUNE_FROM="'$CKPT_PATH'" \
        ++paths.log_dir="$LOG_DIR" \
        ++GENERAL.task_name="ch5_${DIR_NAME}" \
        ++GENERAL.TRAIN_BATCHES_PER_EPOCH="${CH5_STEPS_PER_EPOCH}" \
        ++GENERAL.CHECKPOINT_EVERY_N_TRAIN_STEPS="${CH5_STEPS_PER_EPOCH}" \
        $OVERRIDES \
        $RESUME_ARGS 2>&1 | tee "${LOG_DIR}/train.log" )
}

# --- 4. EXPERIMENT GROUP M0-M5 ---
# Baseline LR=1e-5 @ BS=48. Scaling: LR = 1e-5 * (BS/48)

L8_PL="[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]"
M4_MODES=""
for i in {8..10}; do M4_MODES+="++MASK_CONFIG.layer_modes.$i=soft "; done
for i in {11..31}; do M4_MODES+="++MASK_CONFIG.layer_modes.$i=hard "; done

run_method() {
    local METHOD=$1
    case "$METHOD" in
        M0)
            # M0: Freeze 8, NoMask (resume: dispatch_experiment finds .../train/runs/*/last.ckpt)
            dispatch_experiment 0 "M0" "NoMask" 8 192 1 8.0e-5 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=none"
            ;;
        M1)
            # M1: Freeze 16, Pos16
            dispatch_experiment 1 "M1" "Pos16" 16 192 1 1.33e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=[16]"
            ;;
        M2)
            # M2: Freeze 24, Pos24
            dispatch_experiment 2 "M2" "Pos24" 24 192 1 1.6e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=[24]"
            ;;
        M3)
            # M3: Global Soft Mask
            dispatch_experiment 3 "M3" "8PlusSoft" 8 192 1 1.06e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=soft ++MASK_CONFIG.mask_layers=${L8_PL}"
            ;;
        M4)
            # M4: Adaptive KTI
            dispatch_experiment 4 "M4" "AdaptiveKTI" 8 128 1 1.06e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=adaptive ++MASK_CONFIG.mask_layers=${L8_PL} ${M4_MODES}"
            ;;
        M5)
            # M5: Global Hard Mask
            dispatch_experiment 5 "M5" "8PlusHard" 8 128 1 1.06e-4 "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False ++MASK_CONFIG.mode=hard ++MASK_CONFIG.mask_layers=${L8_PL}"
            ;;
        *)
            echo "❌ Unknown method: $METHOD"
            echo "Usage: bash scripts/ch5_prior_compare_train.sh [all|M0|M1|M2|M3|M4|M5]"
            exit 1
            ;;
    esac
}

MODE="${1:-all}"
if [ "$MODE" = "all" ]; then
    run_method M0
    run_method M1
    run_method M2
    run_method M3
    run_method M4
    run_method M5
else
    run_method "$MODE"
fi

