#!/bin/bash
# scripts/run_dlc_24gpu.sh - Fully Portable Multi-Node Entry Point (3x8 GPUs)

# --- 1. DYNAMIC ENVIRONMENT DETECTION ---
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
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PYTHON_EXE="$PARENT_DIR/opt/Miniconda3/envs/4D-humans/bin/python"
TORCHRUN_EXE="$PARENT_DIR/opt/Miniconda3/envs/4D-humans/bin/torchrun"
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

# PAI-DLC: RANK=节点 Index，WORLD_SIZE=节点数。子进程 RANK/WORLD_SIZE 由 torchrun 覆盖（有 MASTER 时）。
NNODES="${WORLD_SIZE:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${RANK:-0}"
export NODE_RANK
export WORLD_SIZE_NODES="${NNODES}"

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
# Experiment root on shared storage. Each run writes under:
#   ${OUTPUT_DIR}/${task_name}/runs/<YYYY-MM-DD_HH-MM-SS>/
# (configs, checkpoints/, tensorboard/, exec_time.log, .hydra/)
# Override via: ++paths.log_dir=/other/root (do not set a separate flat paths.output_dir; training syncs to Hydra run dir)
OUTPUT_DIR="/cpfs_infra/shared/yangz/NViT-master/output/ch6"
mkdir -p "$OUTPUT_DIR"

if [ -n "${MASTER_ADDR:-}" ] && [ -n "${MASTER_PORT:-}" ] && [ -x "$TORCHRUN_EXE" ]; then
  echo "🔥 torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --node_rank=$NODE_RANK"
  LAUNCHER=("$TORCHRUN_EXE" --nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE" --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
else
  LAUNCHER=("$PYTHON_EXE" -u)
fi

"${LAUNCHER[@]}" nvit/train_guided.py \
    --config-dir "$CONFIG_DIR" \
    --config-name train \
    ++DATASETS_CONFIG_FILE="$PROJECT_ROOT/scripts/datasets_tar.yaml" \
    experiment=hmr_vit_transformer \
    data=mix_all \
    ++trainer.num_nodes=3 \
    ++trainer.devices=8 \
    ++trainer.strategy=ddp_find_unused_parameters_true \
    ++trainer.precision="bf16-mixed" \
    ++TRAIN.BATCH_SIZE=512 \
    ++TRAIN.ACCUMULATE_GRAD_BATCHES=1 \
    ++GENERAL.NUM_WORKERS=8 \
    ++FREEZE_DEPTH=7 \
    ++GENERAL.PREFETCH_FACTOR=2 \
    ++TRAIN.LR=2.4e-4 \
    ++LOSS_WEIGHTS.KEYPOINTS_3D=5.0 \
    ++LOSS_WEIGHTS.GLOBAL_ORIENT=5.0 \
    ++LOSS_WEIGHTS.BODY_POSE=5.0 \
    ++SMPL.DATA_DIR=$SHARED_CACHE/data/ \
    ++MODEL.BACKBONE.USE_CHECKPOINT=False \
    ++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=True \
    ++MASK_CONFIG.mode=none \
    ++paths.log_dir="$OUTPUT_DIR" \
    ++FINETUNE_FROM="'$SHARED_CACHE/logs/train/multiruns/hmr2/0/ch6_history_ckpts/epoch_epoch=05.ckpt'" \
    ++GENERAL.CHECKPOINT_EVERY_N_TRAIN_STEPS=10 \
    ++GENERAL.TRAIN_BATCHES_PER_EPOCH=10 \
    ++GENERAL.TOTAL_STEPS=50