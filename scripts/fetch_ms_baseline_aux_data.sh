#!/usr/bin/env bash
# 下载 METRO / Mesh Graphormer 运行所需的小型辅助文件（GraphCMR 的 J_regressor *.npy）。
# SMPL pkl 需自行从官网放到 modeling/data（或设置 SMPL_PKL 源路径由本脚本拷贝）。
#
# Usage:
#   bash scripts/fetch_ms_baseline_aux_data.sh
#   SMPL_PKL=/path/to/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl bash scripts/fetch_ms_baseline_aux_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$ROOT/nvit/external_baselines"
RAW_J_EXTRA="${GRAPHCMR_J_EXTRA_URL:-https://raw.githubusercontent.com/nkolot/GraphCMR/master/data/J_regressor_extra.npy}"
RAW_J_H36M="${GRAPHCMR_J_H36M_URL:-https://raw.githubusercontent.com/nkolot/GraphCMR/master/data/J_regressor_h36m_correct.npy}"
RAW_MESH_DOWN="${GRAPHCMR_MESH_DOWN_URL:-https://raw.githubusercontent.com/nkolot/GraphCMR/master/data/mesh_downsampling.npz}"

fetch_one() {
  local url="$1" dst="$2"
  if [[ -f "$dst" ]]; then
    echo "[skip] exists $dst"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  if command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$dst" || return 1
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$dst" || return 1
  else
    echo "Need wget or curl to fetch $url" >&2
    return 1
  fi
  echo "[ok] $dst"
}

# MeshGraphormer: src/modeling/data/
GDATA="$BASE/MeshGraphormer/src/modeling/data"
fetch_one "$RAW_J_EXTRA" "$GDATA/J_regressor_extra.npy"
fetch_one "$RAW_J_H36M" "$GDATA/J_regressor_h36m_correct.npy"
fetch_one "$RAW_MESH_DOWN" "$GDATA/mesh_downsampling.npz"

# METRO: metro/modeling/data/
MDATA="$BASE/MeshTransformer/metro/modeling/data"
fetch_one "$RAW_J_EXTRA" "$MDATA/J_regressor_extra.npy"
fetch_one "$RAW_J_H36M" "$MDATA/J_regressor_h36m_correct.npy"
fetch_one "$RAW_MESH_DOWN" "$MDATA/mesh_downsampling.npz"

if [[ -n "${SMPL_PKL:-}" && -f "$SMPL_PKL" ]]; then
  cp -f "$SMPL_PKL" "$GDATA/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
  cp -f "$SMPL_PKL" "$MDATA/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
  echo "[ok] SMPL pkl copied to Graphormer + METRO data dirs"
else
  echo "[hint] 未设置 SMPL_PKL；请将 basicModel_neutral_lbs_10_207_0_v1.0.0.pkl 拷到:"
  echo "  $GDATA/"
  echo "  $MDATA/"
fi
