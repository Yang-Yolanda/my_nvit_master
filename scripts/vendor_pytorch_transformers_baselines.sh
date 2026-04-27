#!/usr/bin/env bash
# 用 PyPI 上的 pytorch-transformers 1.2.0 wheel 填充
# MeshGraphormer / MeshTransformer 所需的 transformers/pytorch_transformers/
# （不依赖 GitHub submodule；国内可用 pip 阿里云镜像）。
#
# Usage:
#   bash scripts/vendor_pytorch_transformers_baselines.sh
#   PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ bash scripts/vendor_pytorch_transformers_baselines.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
VER="${PYTORCH_TRANSFORMERS_VER:-1.2.0}"
TMP="$(mktemp -d)"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

"$PY" -m pip download "pytorch-transformers==${VER}" -d "$TMP" --no-deps \
  ${PIP_INDEX_URL:+-i "$PIP_INDEX_URL"}

WHL="$(echo "$TMP"/pytorch_transformers-*.whl)"
test -f $WHL

unzip -q -o "$WHL" -d "$TMP/extract"

for NAME in MeshGraphormer MeshTransformer; do
  DEST="$ROOT/nvit/external_baselines/$NAME/transformers/pytorch_transformers"
  mkdir -p "$ROOT/nvit/external_baselines/$NAME/transformers"
  rm -rf "$DEST"
  cp -a "$TMP/extract/pytorch_transformers" "$DEST"
  echo "[ok] $DEST ($(ls "$DEST" | wc -l) files)"
done
