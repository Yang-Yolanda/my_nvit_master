#!/usr/bin/env bash
# For each line in h36m_val_p2 missing list, find an OSS object that exists by trying
# hash suffix aliases (60457274 vs 54138969 vs 55011271), then copy to local using the
# **original npz name** (required for evaluation).
set -euo pipefail
MISSING="${1:?missing list path}"
: "${HUMANS_ROOT:=/cpfs_infra/shared/yangz/4D-Humans}"
OSSP="${OSS_EVAL_PREFIX:-oss://kai-ego/eval_shujuji}/h36m"
DEST="${HUMANS_ROOT}/data/h36m"
mkdir -p "$DEST"
ok=0; skip=0; fail=0
mapfile -t LINES < "$MISSING"
for needed in "${LINES[@]}"; do
  [[ -z "$needed" ]] && continue
  if [[ -f "$DEST/$needed" ]]; then skip=$((skip+1)); continue; fi
  c1="${needed/60457274/54138969}"
  c2="${needed/60457274/55011271}"
  c3="${needed/60457274/60457274}"  # noop
  src=""
  for cand in "$c1" "$c2"; do
    if ossutil stat "${OSSP}/${cand}" &>/dev/null; then src=$cand; break; fi
  done
  if [[ -z "$src" ]]; then
    echo "NOKEY $needed" >&2
    fail=$((fail+1))
    continue
  fi
  if ossutil cp -f "${OSSP}/${src}" "$DEST/${needed}" &>/dev/null; then
    ok=$((ok+1))
  else
    echo "CPFAIL $needed << $src" >&2
    fail=$((fail+1))
  fi
done
echo "DONE ok=$ok skip=$skip fail=$fail"
