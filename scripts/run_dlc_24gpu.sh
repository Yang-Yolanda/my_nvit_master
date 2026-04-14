#!/bin/bash
# scripts/run_dlc_24gpu.sh - Fully Portable Multi-Node Entry Point (3x8 GPUs)

# --- 1. DYNAMIC ENVIRONMENT DETECTION ---
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT" || exit 1

export DATA_ROOT="$PROJECT_ROOT/hmr2_training_data"
PYTHON_EXE="$PARENT_DIR/opt/Miniconda3/envs/4D-humans/bin/python"
CONFIG_DIR="$PARENT_DIR/4D-Humans/hmr2/configs_hydra"

echo "🌐 Dynamic Context:"
echo "🚀 Project: $PROJECT_ROOT"
echo "🚀 Config:  $CONFIG_DIR"
echo "🚀 Data:    $DATA_ROOT"

# --- 2. CLUSTER PERFORMANCE & NETWORKING TUNING (V23: DNS -> IP Fix) ---
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=22
export NCCL_IGNORE_DISABLED_P2P=1

# Fix DNS & IPv6 issues on Aliyun DLC
export NCCL_IPV6=0
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0

# Resolve DLC Rank Collision
export NODE_RANK=${RANK:-0}
export WORLD_SIZE_NODES=${WORLD_SIZE:-1}

# V23: Force resolve MASTER_ADDR to IP to bypass Python DNS/IPv6 retrieval issues
if [ ! -z "$MASTER_ADDR" ]; then
    echo "📡 Resolving Master Hostname: $MASTER_ADDR"
    MAX_RETRIES=60
    COUNT=0
    resolved_ip=""
    while [ $COUNT -lt $MAX_RETRIES ]; do
        resolved_ip=$(getent hosts "$MASTER_ADDR" | awk '{print $1}' | head -n 1)
        if [ ! -z "$resolved_ip" ]; then
            break
        fi
        sleep 2
        COUNT=$((COUNT+1))
    done

    if [ ! -z "$resolved_ip" ]; then
        echo "✅ Master Address Resolved to IP: $resolved_ip"
        export MASTER_ADDR="$resolved_ip"
    else
        echo "⚠️  DNS Resolution failed after 120s! Using hostname fallback: $MASTER_ADDR"
    fi
fi

# Memory & VRAM Optimizations
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
export TORCH_CUDNN_V8_API_ENABLED=1

# --- 3. EXECUTION ---
"$PYTHON_EXE" -u nvit/train_guided.py \
    --config-dir "$CONFIG_DIR" \
    --config-name train \
    experiment=hmr_vit_transformer \
    data=mix_all \
    ++trainer.num_nodes=3 \
    ++trainer.devices=8 \
    ++trainer.strategy=ddp \
    ++trainer.strategy.find_unused_parameters=True \
    ++trainer.precision="bf16-mixed" \
    ++TRAIN.BATCH_SIZE=512 \
    ++TRAIN.ACCUMULATE_GRAD_BATCHES=1 \
    ++GENERAL.NUM_WORKERS=8 \
    ++FREEZE_DEPTH=8 \
    ++GENERAL.PREFETCH_FACTOR=2 \
    ++TRAIN.LR=2.4e-4 \
    ++LOSS_WEIGHTS.KEYPOINTS_3D=5.0 \
    ++LOSS_WEIGHTS.GLOBAL_ORIENT=5.0 \
    ++LOSS_WEIGHTS.BODY_POSE=5.0 \
    ++MODEL.BACKBONE.USE_CHECKPOINT=True \
    ++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=False \
    ++MASK_CONFIG.mode=none \
    ++paths.log_dir=/mnt/yangz/nvit_output/ch6