#!/usr/bin/env bash
# One-shot cluster eval for tmux test-mpjpe (3DPW smoke + limit batches). Remove after use.
set -euo pipefail
cd /cpfs_infra/shared/yangz/NViT-master
export PYTHON="${PYTHON:-/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python}"
export PYTHONPATH="${PWD}:/cpfs_infra/shared/yangz/4D-Humans"
export HUMANS_ROOT="${HUMANS_ROOT:-/cpfs_infra/shared/yangz/4D-Humans}"
export HMR2_CFG_REFERENCE_CKPT="${HMR2_CFG_REFERENCE_CKPT:-/cpfs_infra/shared/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt}"
export CLUSTER_DATASETS="${CLUSTER_DATASETS:-3DPW-TEST}"
export CLUSTER_LIMIT_BATCHES="${CLUSTER_LIMIT_BATCHES:-15}"
# Do not use `exec ... | tee` — bash parses it incorrectly; stdout never reaches tee reliably.
set -o pipefail
bash scripts/run_best_max_step_eval.sh cluster 2>&1 | tee /cpfs_infra/shared/yangz/NViT-master/artifacts/eval_unified/tmux_test_mpjpe_cluster.log
