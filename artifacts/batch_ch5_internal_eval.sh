#!/usr/bin/env bash
# Ch5 六组消融：人类学 + 内部指标（熵 / MAD / 有效秩 / KTI）→ outputs/eval_global/Ch5/summary.csv
#
# 多卡: 默认 CH5_INTERNAL_GPUS=0,1,2,3,4,5 六路并行（每任务一张物理卡 + 子进程 cuda:0）。
#   仅单卡: CH5_INTERNAL_GPUS=0
# 说明: 这是「6 个独立整段评测」的任务并行，不是训练 DDP。
#
# 用法:
#   cd /cpfs_infra/shared/yangz/NViT-master
#   conda activate 4D-humans   # 或设 PYTHON 指向 4D-humans
#   bash artifacts/batch_ch5_internal_eval.sh
#
# 环境变量:
#   CH5_INTERNAL_GPUS — 物理 GPU 列表，逗号分隔。默认 0,1,2,3,4,5
#   CH5_BASE, CH5_MANUAL_LIST_FILE, DIAG_BATCHES, PYTHON

set -euo pipefail
NVIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$NVIT_ROOT"
export PYTHONPATH="${NVIT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python}"
export CH5_BASE="${CH5_BASE:-$NVIT_ROOT/output/ch5_prior_compare}"
export DIAG_BATCHES="${DIAG_BATCHES:-50}"
# 默认 6 卡 0–5 与 6 组一一对应；只 1 张卡: CH5_INTERNAL_GPUS=0
export CH5_INTERNAL_GPUS="${CH5_INTERNAL_GPUS:-0,1,2,3,4,5}"
echo "CH5_INTERNAL_GPUS=$CH5_INTERNAL_GPUS" >&2

args=(
  --ch5-base "$CH5_BASE"
  --diag-batches "$DIAG_BATCHES"
  --python "$PYTHON"
  --datasets "${DATASETS:-ALL}"
)
# 空字符串表示只要单卡串行: Python 不读 --gpus
if [[ -n "$CH5_INTERNAL_GPUS" ]]; then
  args+=(--gpus "$CH5_INTERNAL_GPUS")
else
  args+=(--gpu "${GPU:-0}")
fi
if [[ -n "${CH5_MANUAL_LIST_FILE:-}" ]]; then
  args+=(--manual-list "$CH5_MANUAL_LIST_FILE")
fi
if [[ "${DRY:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

exec "$PYTHON" "$NVIT_ROOT/artifacts/ch5_ablation_internal_eval.py" "${args[@]}"
