#!/usr/bin/env bash
# Pull only missing H36M val-P2 jpgs from OSS, using a list from audit_h36m_eval_images.py
#
# Prereq: ossutil configured; bucket contains flat files: oss://.../h36m/<basename>
#
# Usage:
#   export HUMANS_ROOT=/cpfs_infra/shared/yangz/4D-Humans
#   python3 NViT-master/scripts/audit_h36m_eval_images.py --out-missing /tmp/h36m_missing.txt
#   bash NViT-master/scripts/sync_h36m_missing_from_oss.sh /tmp/h36m_missing.txt
#
set -euo pipefail

MISSING_FILE="${1:-}"
if [[ -z "${MISSING_FILE}" || ! -f "${MISSING_FILE}" ]]; then
  echo "Usage: $0 /path/to/missing_list.txt" >&2
  echo "Generate list with: python3 scripts/audit_h36m_eval_images.py --out-missing /tmp/h36m_missing.txt" >&2
  exit 1
fi

OSS_PREFIX="${OSS_EVAL_PREFIX:-oss://kai-ego/eval_shujuji}"
HUMANS_ROOT="${HUMANS_ROOT:-/cpfs_infra/shared/yangz/4D-Humans}"
DEST="${HUMANS_ROOT}/data/h36m"
mkdir -p "${DEST}"

echo "OSS_PREFIX=${OSS_PREFIX}/h36m/"
echo "DEST=${DEST}/"
n=$(wc -l < "${MISSING_FILE}" | tr -d ' ')
echo "Files to try: ${n}"
ok=0
fail=0
set +e
while IFS= read -r line || [[ -n "${line}" ]]; do
  line="${line//$'\r'/}"
  [[ -z "${line}" ]] && continue
  # npz uses flat basename; strip any subdir
  base="${line##*/}"
  if [[ -f "${DEST}/${base}" ]]; then
    ok=$((ok + 1))
    continue
  fi
  if ossutil cp -f "${OSS_PREFIX}/h36m/${base}" "${DEST}/${base}" 2>/dev/null; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    if (( fail <= 10 )); then
      echo "FAIL: ${base}" >&2
    fi
  fi
done < "${MISSING_FILE}"

echo "Done. Present or fetched: ${ok}, failed (or not on OSS): ${fail}"
if (( fail > 0 )); then
  echo "Tip: re-run full tree: bash $(dirname "$0")/sync_eval_data_from_oss.sh" >&2
  exit 1
fi
exit 0
