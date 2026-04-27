#!/usr/bin/env bash
# 从 Microsoft 公开 Azure Blob 拉取 HRNet 与 METRO/MeshGraphormer 的 state_dict（与官方 scripts/download_models.sh 同源）。
#
# Usage:
#   bash scripts/download_ms_azure_pretrained.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOB="${MS_METRO_BLOB:-https://datarelease.blob.core.windows.net/metro}"

dl() {
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" && "${FORCE_MS_DL:-0}" != "1" ]]; then
    echo "[skip] $out"
    return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$out"
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$out"
  else
    echo "Need wget or curl" >&2
    return 1
  fi
  echo "[ok] $out"
}

MG="$ROOT/nvit/external_baselines/MeshGraphormer"
MT="$ROOT/nvit/external_baselines/MeshTransformer"

mkdir -p "$MG/models/graphormer_release" "$MG/models/hrnet"
dl "$BLOB/models/graphormer_h36m_state_dict.bin" "$MG/models/graphormer_release/graphormer_h36m_state_dict.bin"
dl "$BLOB/models/hrnetv2_w64_imagenet_pretrained.pth" "$MG/models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
dl "$BLOB/models/hrnetv2_w40_imagenet_pretrained.pth" "$MG/models/hrnet/hrnetv2_w40_imagenet_pretrained.pth"

mkdir -p "$MT/models/metro_release" "$MT/models/hrnet"
dl "$BLOB/models/metro_h36m_state_dict.bin" "$MT/models/metro_release/metro_h36m_state_dict.bin"
dl "$BLOB/models/hrnetv2_w64_imagenet_pretrained.pth" "$MT/models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
dl "$BLOB/models/hrnetv2_w40_imagenet_pretrained.pth" "$MT/models/hrnet/hrnetv2_w40_imagenet_pretrained.pth"
