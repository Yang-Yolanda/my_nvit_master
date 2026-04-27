#!/usr/bin/env bash
# Clone Microsoft METRO (MeshTransformer) + MeshGraphormer with submodules.
# 若无法访问 GitHub，可在已 vendored 的 nvit/external_baselines 副本上运行:
#   bash scripts/vendor_pytorch_transformers_baselines.sh
# 从 PyPI 填充 transformers/pytorch_transformers/（不依赖 submodule）。
#
# Usage:
#   bash scripts/sync_microsoft_hmr_baselines.sh
#   DEST=/path/to/parent  bash scripts/sync_microsoft_hmr_baselines.sh
#
# After clone, point diagnostics at the tree:
#   export MESHGRAPHORMER_ROOT=$DEST/MeshGraphormer
#   export MESHTRANSFORMER_ROOT=$DEST/MeshTransformer
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$ROOT/third_party}"
mkdir -p "$DEST"

clone_one() {
  local url="$1" name="$2"
  local target="$DEST/$name"
  if [[ -d "$target/.git" ]]; then
    echo "[sync] Updating submodules in $target"
    git -C "$target" submodule update --init --recursive || true
    git -C "$target" pull --ff-only || true
    return 0
  fi
  echo "[sync] Cloning $url -> $target"
  git clone --depth 1 --recurse-submodules "$url" "$target"
}

clone_one "https://github.com/microsoft/MeshGraphormer.git" "MeshGraphormer"
clone_one "https://github.com/microsoft/MeshTransformer.git" "MeshTransformer"

echo "[sync] Done. Trees under: $DEST"
echo "  export MESHGRAPHORMER_ROOT=$DEST/MeshGraphormer"
echo "  export MESHTRANSFORMER_ROOT=$DEST/MeshTransformer"
echo ""
echo "若仓库内 nvit/external_baselines/MeshGraphormer 的 modeling_bert.py 指向断开的 symlink，"
echo "可改用完整树（推荐）:"
echo "  export MESHGRAPHORMER_ROOT=$DEST/MeshGraphormer"
echo "或把子模块目录拷回 vendored 树:"
echo "  rsync -a --delete $DEST/MeshGraphormer/transformers/ $ROOT/nvit/external_baselines/MeshGraphormer/transformers/"
echo "  rsync -a --delete $DEST/MeshTransformer/transformers/ $ROOT/nvit/external_baselines/MeshTransformer/transformers/"
echo ""
echo "METRO/Mesh 推理测速（整网 pickle，见 artifacts/run_metro_meshg_speed.sh）:"
echo "  export METRO_PICKLED_CKPT=/path/to/metro_body_mesh.pth"
echo "  export MESHG_PICKLED_CKPT=/path/to/graphormer_body_mesh.pth"
echo "  bash artifacts/run_metro_meshg_speed.sh"
