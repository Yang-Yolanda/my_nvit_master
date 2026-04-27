#!/usr/bin/env bash
# METRO / Mesh Graphormer 推理时间（224×224 随机输入）。
#
# 准备（一次）:
#   bash scripts/vendor_pytorch_transformers_baselines.sh
#   bash scripts/fetch_ms_baseline_aux_data.sh   # 可选: SMPL_PKL=.../basicModel_neutral...pkl
#   bash scripts/download_ms_azure_pretrained.sh   # HRNet + *_state_dict.bin
#
# 权重优先级:
#   METRO_PICKLED_CKPT / MESHG_PICKLED_CKPT — 整网 pickle（若设置且存在）
#   否则使用 release 的 metro_h36m_state_dict.bin / graphormer_h36m_state_dict.bin（若存在）
#
# Python: 默认依次尝试 MS_BASELINE_PYTHON、conda run -n BASELINE_LEGACY_ENV、python3。
set -euo pipefail
ROOT="/cpfs_infra/shared/yangz/NViT-master"
CONDA="${CONDA:-/cpfs_infra/shared/yangz/opt/Miniconda3/bin/conda}"
ENV="${BASELINE_LEGACY_ENV:-nvit_metro_cu101}"
GPU="${GPU:-0}"
OUT_CSV="${METRO_MESHG_SPEED_CSV:-${ROOT}/outputs/eval_global/Ch6A/metro_meshg_inference_speed.csv}"
METRO_ROOT="${ROOT}/nvit/external_baselines/MeshTransformer"
MESHG_ROOT="${ROOT}/nvit/external_baselines/MeshGraphormer"
BENCH_PY="${ROOT}/nvit/external_baselines/bench_metro_meshg_forward_time.py"

METRO_RELEASE_BIN="${METRO_ROOT}/models/metro_release/metro_h36m_state_dict.bin"
MESHG_RELEASE_BIN="${MESHG_ROOT}/models/graphormer_release/graphormer_h36m_state_dict.bin"

if [[ -n "${METRO_PICKLED_CKPT:-}" && -f "${METRO_PICKLED_CKPT}" ]]; then
  METRO_CKPT="${METRO_PICKLED_CKPT}"
elif [[ -f "$METRO_RELEASE_BIN" ]]; then
  METRO_CKPT="$METRO_RELEASE_BIN"
else
  METRO_CKPT=""
fi

if [[ -n "${MESHG_PICKLED_CKPT:-}" && -f "${MESHG_PICKLED_CKPT}" ]]; then
  MESHG_CKPT="${MESHG_PICKLED_CKPT}"
elif [[ -f "$MESHG_RELEASE_BIN" ]]; then
  MESHG_CKPT="$MESHG_RELEASE_BIN"
else
  MESHG_CKPT=""
fi

py_bench() {
  if [[ -n "${MS_BASELINE_PYTHON:-}" ]]; then
    "$MS_BASELINE_PYTHON" "$BENCH_PY" "$@"
    return
  fi
  if [[ -x "$CONDA" ]] && command -v conda >/dev/null 2>&1; then
    if "$CONDA" run -n "$ENV" --no-capture-output python "$BENCH_PY" "$@"; then
      return 0
    fi
  fi
  python3 "$BENCH_PY" "$@"
}

mkdir -p "$(dirname "$OUT_CSV")"
{
  echo "name,ckpt,batch,iters,ms_per_step,ms_per_image,images_per_s"
  if [[ -n "$METRO_CKPT" ]]; then
    py_bench --which metro --code-root "$METRO_ROOT" --ckpt "$METRO_CKPT" \
      --label METRO --warmup 10 --iters 50 --gpu "$GPU" || \
      echo "METRO,,1,0,,,SKIPPED_metro_bench_failed"
  else
    echo "METRO,,1,0,,,SKIPPED_no_metro_ckpt_run_download_ms_azure_pretrained"
  fi
  if [[ -n "$MESHG_CKPT" ]]; then
    py_bench --which meshg --code-root "$MESHG_ROOT" --ckpt "$MESHG_CKPT" \
      --label MeshGraphormer --warmup 10 --iters 50 --gpu "$GPU" || \
      echo "MeshGraphormer,,1,0,,,SKIPPED_meshg_bench_failed"
  else
    echo "MeshGraphormer,,1,0,,,SKIPPED_no_meshg_ckpt_run_download_ms_azure_pretrained"
  fi
} > "$OUT_CSV"

echo "[ok] $OUT_CSV"
