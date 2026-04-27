#!/usr/bin/env bash
# Ch6：按 metrics_master 中 rank-sum 最小的前 10 个模型跑 global_evaluator（人体系 + 内部熵/距离/秩/KTI）
# 多卡: 本脚本通过 CH6_INTERNAL_GPUS 做「每进程一张物理卡 + CUDA_VISIBLE_DEVICES」任务级并行
#   （10 个独立完整评测，不是训练 DDP）。
#
# 用法:
#   cd /cpfs_infra/shared/yangz/NViT-master
#   # 10 张卡一轨跑完（有则改为你机器上的 id）:
#   CH6_INTERNAL_GPUS=0,1,2,3,4,5,6,7,8,9 bash artifacts/batch_ch6_top10_internal_eval.sh
#   # 仅 4 张卡: 会分 3 波 4+4+2
#   CH6_INTERNAL_GPUS=0,1,2,3 bash artifacts/batch_ch6_top10_internal_eval.sh
#
# 单卡串行: 去掉 CH6_INTERNAL_GPUS 或  CH6_INTERNAL_GPUS=""  ...
#
# 其它: TOP, CH6_INTERNAL_DIAG, RANK_3D, CHAPTER, DRY, PYTHON, CH6_INTERNAL_GPU(仅单卡)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

TOP="${TOP:-10}"
CHAPTER="${CHAPTER:-Ch6A}"
RANK_3D="${RANK_3D:-mode_re}"
DIAG="${CH6_INTERNAL_DIAG:-50}"
PY="${PYTHON:-/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python}"
# 默认 10 卡可一次跑满 10 个 job；机子不够请覆盖为例如 0,1,2,3
if [[ -z "${CH6_INTERNAL_GPUS:-}" ]]; then
  echo "未设置 CH6_INTERNAL_GPUS：将单卡串行。多卡请例如: CH6_INTERNAL_GPUS=0,1,2,3 $0" >&2
  export CH6_INTERNAL_GPUS=""
else
  echo "CH6_INTERNAL_GPUS=$CH6_INTERNAL_GPUS" >&2
fi

args=(--top "$TOP" --chapter "$CHAPTER" --rank-metric-3d "$RANK_3D" --diag-batches "$DIAG" --python "$PY")
# 有列表则传给 --gpus；无则让 Python 走单卡
if [[ -n "${CH6_INTERNAL_GPUS:-}" ]]; then
  args+=(--gpus "$CH6_INTERNAL_GPUS")
else
  args+=(--gpu "${CH6_INTERNAL_GPU:-0}")
fi

if [[ "${DRY:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

echo "== ch6 top-${TOP} internal eval ==" >&2
exec "$PY" "$ROOT/artifacts/ch6_topN_internal_eval.py" "${args[@]}"
