#!/bin/bash
# scripts/verify_ch6_save.sh - Verification script for CPFS path saving

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT" || exit 1

# [NEW] Automated Environment Setup: Ensure local cache symlink exists for 4D-Humans
SHARED_CACHE="/cpfs_infra/shared/yangz/.cache/4DHumans"
if [ ! -L "$HOME/.cache/4DHumans" ]; then
    echo "🔗 Creating symlink: $HOME/.cache/4DHumans -> $SHARED_CACHE"
    mkdir -p "$HOME/.cache"
    rm -rf "$HOME/.cache/4DHumans" # Remove if it's a dead link or directory
    ln -s "$SHARED_CACHE" "$HOME/.cache/4DHumans"
fi

export DATA_ROOT="$PROJECT_ROOT/hmr2_training_data"
PYTHON_EXE="$PARENT_DIR/opt/Miniconda3/envs/4D-humans/bin/python"
CONFIG_DIR="$PARENT_DIR/4D-Humans/hmr2/configs_hydra"

# Use 1 GPU for verification test
export CUDA_VISIBLE_DEVICES=0

# Memory & VRAM Optimizations
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
export TORCH_CUDNN_V8_API_ENABLED=1

VERIFY_OUT="/cpfs_infra/shared/yangz/NViT-master/output/ch6/verify_test"
echo "🧪 Starting CPFS Path Verification Test..."
echo "🚀 Target Output Dir: $VERIFY_OUT"

"$PYTHON_EXE" -u nvit/train_guided.py \
    --config-dir "$CONFIG_DIR" \
    --config-name train \
    hydra.job.chdir=False \
    ++DATASETS_CONFIG_FILE="$PROJECT_ROOT/scripts/datasets_tar_smoke.yaml" \
    experiment=hmr_vit_transformer \
    data=mix_all \
    ++trainer.num_nodes=1 \
    ++trainer.devices=1 \
    ++trainer.max_steps=10 \
    ++trainer.precision="bf16-mixed" \
    ++TRAIN.BATCH_SIZE=16 \
    ++TRAIN.ACCUMULATE_GRAD_BATCHES=1 \
    ++GENERAL.NUM_WORKERS=4 \
    ++FREEZE_DEPTH=7 \
    ++GENERAL.CHECKPOINT_STEPS=5 \
    ++GENERAL.CHECKPOINT_SAVE_TOP_K=1 \
    ++SMPL.DATA_DIR="$SHARED_CACHE/data/" \
    ++MODEL.BACKBONE.PRETRAINED_WEIGHTS="$DATA_ROOT/vitpose_backbone.pth" \
    ++paths.log_dir="$VERIFY_OUT" \
    ++paths.output_dir="$VERIFY_OUT" \
    ++FINETUNE_FROM="'/cpfs_infra/shared/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'"
