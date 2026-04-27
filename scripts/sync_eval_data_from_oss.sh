#!/usr/bin/env bash
# Sync HMR2 eval images from Aliyun OSS into 4D-Humans data layout (matches hmr2/configs/datasets_eval.yaml).
#
# Source (your bucket):
#   oss://kai-ego/eval_shujuji/h36m/*.jpg          -> $HUMANS_ROOT/data/h36m/
#   oss://kai-ego/eval_shujuji/hr-lspet/image/*.png -> $HUMANS_ROOT/data/hr-lspet/image/
#   oss://kai-ego/eval_shujuji/coco/val2017/*.jpg   -> $HUMANS_ROOT/data/coco/val2017/
#
# Usage:
#   export HUMANS_ROOT=/cpfs_infra/shared/yangz/4D-Humans   # default below
#   bash scripts/sync_eval_data_from_oss.sh
#
# Requires: ossutil (configured for bucket kai-ego). Does not copy coco/*.tar shards (only val2017).
#
set -euo pipefail

OSS_PREFIX="${OSS_EVAL_PREFIX:-oss://kai-ego/eval_shujuji}"
HUMANS_ROOT="${HUMANS_ROOT:-/cpfs_infra/shared/yangz/4D-Humans}"
DEST_DATA="${HUMANS_ROOT}/data"

echo "OSS_PREFIX=$OSS_PREFIX"
echo "DEST_DATA=$DEST_DATA"

mkdir -p "${DEST_DATA}/h36m" "${DEST_DATA}/hr-lspet/image" "${DEST_DATA}/coco/val2017"

echo "=== [1/3] h36m (flat jpgs) ==="
ossutil cp -r "${OSS_PREFIX}/h36m/" "${DEST_DATA}/h36m/" -f

echo "=== [2/3] hr-lspet/image (png) ==="
ossutil cp -r "${OSS_PREFIX}/hr-lspet/image/" "${DEST_DATA}/hr-lspet/image/" -f

echo "=== [3/3] coco val2017 only (skip coco/*.tar) ==="
ossutil cp -r "${OSS_PREFIX}/coco/val2017/" "${DEST_DATA}/coco/val2017/" -f

echo "Done. Spot-check:"
ls "${DEST_DATA}/h36m" 2>/dev/null | head -3
ls "${DEST_DATA}/hr-lspet/image" 2>/dev/null | head -3
ls "${DEST_DATA}/coco/val2017" 2>/dev/null | head -3

echo "Verify H36M val-P2 coverage (expect 100%):"
echo "  python3 NViT-master/scripts/audit_h36m_eval_images.py"
echo "If still missing, top-up from list:"
echo "  python3 NViT-master/scripts/audit_h36m_eval_images.py --out-missing /tmp/h36m_missing.txt"
echo "  bash NViT-master/scripts/sync_h36m_missing_from_oss.sh /tmp/h36m_missing.txt"
