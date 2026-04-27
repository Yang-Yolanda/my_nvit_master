#!/bin/bash
# scripts/run_dlc_24gpu.sh — NViT Guided Ch6 多机/单机入口（PAI-DLC 与本地）
#
# 多机: 各节点需有相同代码与数据；PAI 通常注入 MASTER_ADDR / MASTER_PORT / RANK(节点序) /
#       WORLD_SIZE(节点数)。本脚本用 torchrun 的 --nnodes / --node_rank 与
#       ++trainer.num_nodes=NNODES 对齐，Lightning DDP 即可跨机。
# 单机多卡: 不要设 MASTER_*，将 NNODES=1、NPROC_PER_NODE=本地 GPU 数（见下方“本地”段）。
#
# 从权重复训: ++FINETUNE_FROM 为「最好 step」的 weights-only ckpt；ckpt_path 置 null，
# 避免误用同目录 last.ckpt 的优化器/步数（改 FREEZE 后应与新优化器重训）。

set -euo pipefail

# --- 0) 可编辑参数（也可在提交作业前 export 覆盖）---------------------------------
# 微调入口权重：ch6 phase2 / unfreeze5 上当前表现最好的 step（如节点上无 /mnt/… 请改 CH6_BEST_CKPT 或做软链）
CH6_BEST_CKPT="${CH6_BEST_CKPT:-/mnt/yangz/nvit_output/ch6_phase2_unfreeze5/train/runs/2026-04-24_16-37-43/checkpoints/step_step=80000.ckpt}"

# 新实验输出根（unfreeze0 等配置与上阶段分目录；Hydra 会建 train/runs/<时间戳>/）
OUTPUT_DIR="${OUTPUT_DIR:-/cpfs_infra/shared/yangz/nvit_output/ch6_phase2_unfreeze0_from80k}"

# PAI: WORLD_SIZE = 节点数；单机: 用默认 1
NNODES="${WORLD_SIZE:-1}"
# 每机 GPU 数
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# 常用热修（不改 yaml 时在此改）；默认留空。
# 注意：展开必须用 "${EXTRA_HYDRA[@]}"；若写 "${EXTRA_HYDRA[@]:-}"，空数组时 bash 会多出一个 "" 参数，Hydra 报 mismatched input '<EOF>'。
EXTRA_HYDRA=()
# 例: EXTRA_HYDRA=( "++TRAIN.BATCH_SIZE=128" "++TRAIN.LR=8.0e-5" "++MODEL.BACKBONE.mamba_variant=bi" )

# ---------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT" || exit 1

# [NEW] Automated Environment Setup: Ensure local cache symlink exists for 4D-Humans
SHARED_CACHE="/cpfs_infra/shared/yangz/.cache/4DHumans"
if [ ! -L "$HOME/.cache/4DHumans" ]; then
    echo "🔗 Creating symlink: $HOME/.cache/4DHumans -> $SHARED_CACHE"
    mkdir -p "$HOME/.cache"
    rm -rf "$HOME/.cache/4DHumans" # Remove if it is a dead link or directory
    ln -s "$SHARED_CACHE" "$HOME/.cache/4DHumans"
fi

export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/hmr2_training_data}"
# 无缓冲输出：torchrun 不接受 `python -u` 的 -u，用环境变量等效
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PYTHON_EXE="${PYTHON_EXE:-$PARENT_DIR/opt/Miniconda3/envs/4D-humans/bin/python}"
TORCHRUN_EXE="${TORCHRUN_EXE:-$PARENT_DIR/opt/Miniconda3/envs/4D-humans/bin/torchrun}"
CONFIG_DIR="$PARENT_DIR/4D-Humans/hmr2/configs_hydra"

echo "🌐 Dynamic Context:"
echo "   Project:  $PROJECT_ROOT"
echo "   Config:   $CONFIG_DIR"
echo "   Data:     $DATA_ROOT"
echo "   Finetune: $CH6_BEST_CKPT"
echo "   Log root: $OUTPUT_DIR"
echo "   Nodes:    NNODES=$NNODES  nproc_per_node=$NPROC_PER_NODE"

# --- 2) CLUSTER PERFORMANCE & NETWORKING (DLC) ---
export NCCL_IB_DISABLE=0
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_GID_INDEX=3
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=22
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_IPV6=0
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-eth0}"

NODE_RANK="${RANK:-0}"
export NODE_RANK
export WORLD_SIZE_NODES="${NNODES}"

if [ -n "${MASTER_ADDR:-}" ]; then
    echo "📡 Resolving Master Hostname: $MASTER_ADDR"
    MAX_RETRIES=60
    COUNT=0
    resolved_ip=""
    while [ $COUNT -lt $MAX_RETRIES ]; do
        resolved_ip=$(getent hosts "$MASTER_ADDR" | awk '{print $1}' | head -n 1)
        if [ -n "$resolved_ip" ]; then
            break
        fi
        sleep 2
        COUNT=$((COUNT+1))
    done
    if [ -n "$resolved_ip" ]; then
        echo "✅ Master Address Resolved to IP: $resolved_ip"
        export MASTER_ADDR="$resolved_ip"
    else
        echo "⚠️  DNS Resolution failed after ${MAX_RETRIES}s! Using: $MASTER_ADDR"
    fi
fi

export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
export TORCH_CUDNN_V8_API_ENABLED=1

mkdir -p "$OUTPUT_DIR"

# --- 3) 启动器: 有 MASTER+torchrun → 多机/多进程；否则 python（调试用单进程）---
if [ -n "${MASTER_ADDR:-}" ] && [ -n "${MASTER_PORT:-}" ] && [ -x "$TORCHRUN_EXE" ]; then
  echo "🔥 Launch: torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --node_rank=$NODE_RANK"
  LAUNCHER=("$TORCHRUN_EXE" --nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE" --node_rank "$NODE_RANK"
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
else
  echo "ℹ️  未检测到 MASTER+torchrun → 使用 python 单进程（多卡请安装 torch 并设置 MASTER+torchrun，"
  echo "   或单机多卡: export CUDA_VISIBLE_DEVICES=0,1,... 后改用 torchrun 且 NNODES=1）"
  LAUNCHER=("$PYTHON_EXE" -u)
fi

# FINETUNE_FROM: ckpt 名常含 step_step=…，值内 '=' 必须用 Hydra 引号包住，否则报 mismatched input '='
FINETUNE_ARG="++FINETUNE_FROM='${CH6_BEST_CKPT}'"

# 3×8 GPU 时 total processes = 24, global batch = TRAIN.BATCH_SIZE（每步各卡各一份）
# 若你实际节点数/卡数不同，只改 NNODES / NPROC_PER_NODE / TRAIN.BATCH_SIZE 保持线性关系即可。

# CPFS + torchrun：每 rank 各跑一份 Hydra job，同时写同一 .hydra/config.yaml 会 OSError:22 / EBUSY；关闭 Hydra 的 .hydra 落盘（配置仍由 Lightning / 代码写 run 目录）
HYDRA_NO_DOT_DIR="${HYDRA_NO_DOT_DIR:-1}"
HYDRA_DOT_ARGS=()
if [ "$HYDRA_NO_DOT_DIR" = "1" ]; then
  HYDRA_DOT_ARGS=( "++hydra.output_subdir=null" )
fi

"${LAUNCHER[@]}" nvit/train_guided.py \
    --config-dir "$CONFIG_DIR" \
    --config-name train \
    "${HYDRA_DOT_ARGS[@]}" \
    "++DATASETS_CONFIG_FILE=$PROJECT_ROOT/scripts/datasets_tar.yaml" \
    experiment=ch6_phase2_finetune \
    data=mix_all \
    "++trainer.num_nodes=$NNODES" \
    "++trainer.devices=$NPROC_PER_NODE" \
    "++trainer.strategy=ddp_find_unused_parameters_true" \
    "++trainer.precision=bf16-mixed" \
    "++LOSS_WEIGHTS.KEYPOINTS_3D=5.0" \
    "++LOSS_WEIGHTS.GLOBAL_ORIENT=5.0" \
    "++LOSS_WEIGHTS.BODY_POSE=5.0" \
    "++SMPL.DATA_DIR=$SHARED_CACHE/data/" \
    "++MODEL.BACKBONE.USE_CHECKPOINT=False" \
    "++MODEL.BACKBONE.USE_ADAPTIVE_NVIT=True" \
    "++MASK_CONFIG.mode=none" \
    "++paths.log_dir=$OUTPUT_DIR" \
    "++ckpt_path=null" \
    "$FINETUNE_ARG" \
    "++GENERAL.CHECKPOINT_EVERY_N_TRAIN_STEPS=2000" \
    "${EXTRA_HYDRA[@]}"
