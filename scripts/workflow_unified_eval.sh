#!/usr/bin/env bash
# 统一评测工作流：预设环境变量并调用 run_best_max_step_eval.sh / 画图脚本。
# 用法: scripts/workflow_unified_eval.sh <子命令>
#
# 子命令:
#   full8              8 卡典型排布：1×SMPLer + 1×ch6 + 6×ch5（需 HMR2_CFG_REFERENCE_CKPT）
#   ch6-only-8gpu      仅 ch6，8 卡分片（CH5/SMPLer 关闭）
#   print-ch6-shard K N  打印「单卡手写 ch6 分片」命令（K=0..N-1）
#   plot-ch5           从 metrics_master 画 CH5 最大 step 柱状图（不跑评测）
#   env-print          打印常用环境变量说明
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${ROOT}/scripts/run_best_max_step_eval.sh"
PLOT_SCRIPT="${ROOT}/scripts/plot_ch5_maxstep_bars.py"

die() { echo "ERROR: $*" >&2; exit 1; }

require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

cmd_full8() {
  export CLUSTER_OUT_DIR="${CLUSTER_OUT_DIR:-${ROOT}/artifacts/eval_unified}"
  export CLUSTER_RUN_ID="${CLUSTER_RUN_ID:-full8gpu_$(date +%Y%m%d_%H%M%S)}"
  export CLUSTER_DATASETS="${CLUSTER_DATASETS:-ALL}"
  export CLUSTER_ENABLE_CH5="${CLUSTER_ENABLE_CH5:-1}"
  export CLUSTER_ENABLE_CH6="${CLUSTER_ENABLE_CH6:-1}"
  export CLUSTER_ENABLE_SMPLER="${CLUSTER_ENABLE_SMPLER:-1}"
  export CLUSTER_SMPLER_GPU="${CLUSTER_SMPLER_GPU:-0}"
  export CH6_GPU_LIST="${CH6_GPU_LIST:-1}"
  export CH5_GPU_LIST="${CH5_GPU_LIST:-2,3,4,5,6,7}"
  export PYTHON="${PYTHON:-/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python}"
  export HUMANS_ROOT="${HUMANS_ROOT:-/cpfs_infra/shared/yangz/4D-Humans}"
  if [[ -z "${HMR2_CFG_REFERENCE_CKPT:-}" ]]; then
    die "请设置 HMR2_CFG_REFERENCE_CKPT（SMPLer 需要）"
  fi
  require_file "$RUN_SCRIPT"
  echo "=== workflow full8 ==="
  echo "CLUSTER_OUT_DIR=$CLUSTER_OUT_DIR"
  echo "CLUSTER_RUN_ID=$CLUSTER_RUN_ID"
  echo "Logs: ${CLUSTER_OUT_DIR}/cluster_logs/${CLUSTER_RUN_ID}/"
  exec bash "$RUN_SCRIPT" cluster
}

cmd_ch6_only_8gpu() {
  export CLUSTER_OUT_DIR="${CLUSTER_OUT_DIR:-${ROOT}/artifacts/eval_unified}"
  export CLUSTER_RUN_ID="${CLUSTER_RUN_ID:-ch6_only_8gpu_$(date +%Y%m%d_%H%M%S)}"
  export CLUSTER_DATASETS="${CLUSTER_DATASETS:-ALL}"
  export CLUSTER_ENABLE_CH5=0
  export CLUSTER_ENABLE_SMPLER=0
  export CLUSTER_ENABLE_CH6=1
  export CH6_GPU_LIST="${CH6_GPU_LIST:-0,1,2,3,4,5,6,7}"
  export PYTHON="${PYTHON:-/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python}"
  export HUMANS_ROOT="${HUMANS_ROOT:-/cpfs_infra/shared/yangz/4D-Humans}"
  require_file "$RUN_SCRIPT"
  echo "=== workflow ch6-only-8gpu ==="
  echo "CLUSTER_OUT_DIR=$CLUSTER_OUT_DIR"
  echo "CLUSTER_RUN_ID=$CLUSTER_RUN_ID"
  echo "CH6_GPU_LIST=$CH6_GPU_LIST"
  echo "Logs: ${CLUSTER_OUT_DIR}/cluster_logs/${CLUSTER_RUN_ID}/"
  exec bash "$RUN_SCRIPT" cluster
}

cmd_print_ch6_shard() {
  local k="${1:-}"
  local n="${2:-}"
  [[ -n "$k" && -n "$n" ]] || die "用法: print-ch6-shard <K> <N>  （K 为 0..N-1 的分片下标）"
  export PYTHON="${PYTHON:-/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python}"
  export CLUSTER_OUT_DIR="${CLUSTER_OUT_DIR:-${ROOT}/artifacts/eval_unified}"
  cat <<EOF
# 单进程 ch6 分片（将 CUDA_VISIBLE_DEVICES 设为一张物理卡）:
export CUDA_VISIBLE_DEVICES=<物理GPU_ID>
export CLUSTER_OUT_DIR=${CLUSTER_OUT_DIR}
# 非默认 ch6 目录时追加一行，例: --ch6-experiment-dir ${ROOT}/output/ch6_phase2_unfreeze5 \\
"\$PYTHON" ${ROOT}/scripts/unified_eval_batch.py \\
  --python "\$PYTHON" \\
  --cuda-visible-devices 0 --gpu 0 \\
  --out-dir "\$CLUSTER_OUT_DIR" \\
  --datasets ALL --use-mean-alignment \\
  --ch6-all-steps \\
  --ch6-shard-index ${k} \\
  --ch6-shard-total ${n}
EOF
}

cmd_plot_ch5() {
  require_file "$PLOT_SCRIPT"
  exec "$PYTHON" "$PLOT_SCRIPT" "$@"
}

cmd_env_print() {
  cat <<'EOF'
统一评测相关环境变量（节选）:

  CLUSTER_OUT_DIR     结果根目录，默认 <NViT>/artifacts/eval_unified
  CLUSTER_RUN_ID      日志子目录名；日志在 $CLUSTER_OUT_DIR/cluster_logs/$CLUSTER_RUN_ID/
  CLUSTER_DATASETS    cluster 默认 ALL
  CLUSTER_LIMIT_BATCHES  可选，smoke
  CH5_GPU_LIST        6 个物理 GPU id，对应 M0–M5
  CH6_GPU_LIST        一个 id=单卡 ch6；多个逗号分隔=多分片（--ch6-shard-index/total）
  CH6_EXPERIMENT_DIR  可选，ch6 训练输出根（含 train/runs/.../checkpoints/）；run_best_max_step_eval.sh cluster 会传 --ch6-experiment-dir
  CLUSTER_CH6_GPU     兼容旧名；未设 CH6_GPU_LIST 时当作 CH6_GPU_LIST 的值
  CLUSTER_SMPLER_GPU    SMPLer 物理卡，默认 0
  CLUSTER_ENABLE_CH5 / CLUSTER_ENABLE_CH6 / CLUSTER_ENABLE_SMPLER  设为 0 跳过对应块

同一轮同时 8×ch6 + 6×ch5 + SMPLer 需要 15 张物理 GPU；8 卡请用 workflow full8 或 ch6-only-8gpu。
EOF
}

usage() {
  cat <<EOF
用法: $(basename "$0") <子命令> [参数]

  full8              8 卡：SMPLer(0) + ch6(1) + ch5(2–7)
  ch6-only-8gpu      仅 ch6，8 卡分片
  print-ch6-shard K N  打印单进程 ch6 分片命令
  plot-ch5 [args...]  调用 plot_ch5_maxstep_bars.py（可加其参数，如 --eval-mode full）
  env-print          打印环境变量说明

示例:
  export HMR2_CFG_REFERENCE_CKPT=/path/to/ref.ckpt
  $(basename "$0") full8

  $(basename "$0") ch6-only-8gpu

  $(basename "$0") plot-ch5 --eval-mode full --out-dir ${ROOT}/artifacts/eval_unified/plots
EOF
}

main() {
  local sub="${1:-}"
  [[ -n "$sub" ]] || { usage; exit 1; }
  shift || true
  export PYTHON="${PYTHON:-/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python}"
  case "$sub" in
    full8) cmd_full8 ;;
    ch6-only-8gpu) cmd_ch6_only_8gpu ;;
    print-ch6-shard) cmd_print_ch6_shard "${1:-}" "${2:-}" ;;
    plot-ch5) cmd_plot_ch5 "$@" ;;
    env-print) cmd_env_print ;;
    -h|--help|help) usage ;;
    *) die "未知子命令: $sub（见 --help）" ;;
  esac
}

main "$@"
