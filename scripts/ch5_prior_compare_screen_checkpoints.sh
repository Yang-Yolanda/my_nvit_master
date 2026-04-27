#!/usr/bin/env bash
# Screen all step_*.ckpt under output/ch5_prior_compare using standard_eval (3DPW-TEST only).
#
# Prerequisites:
#   - 4D-Humans eval images on disk (e.g. 3DPW under $HUMANS_ROOT/data/3DPW/...). NPZ alone is NOT enough.
#   - conda env with torch (e.g. 4D-humans): /cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans
#
# Fast screening:
#   - Single dataset 3DPW-TEST, --limit_batches N (e.g. 50) for rough ranking; rerun full eval on top checkpoints.
#
# GPU 6 & 7 (other GPUs busy): run two terminals or use BACKGROUND_JOBS below.
#
# Usage:
#   export PYTHON=/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python
#   export OUT_CSV=/cpfs_infra/shared/yangz/NViT-master/output/ch5_prior_compare/screen_results.csv
#   bash scripts/ch5_prior_compare_screen_checkpoints.sh 6 50
#
# Args: [cuda_visible_device] [limit_batches]
set -euo pipefail

ROOT="/cpfs_infra/shared/yangz/NViT-master"
HUMANS="${HUMANS_ROOT:-/cpfs_infra/shared/yangz/4D-Humans}"
PYTHON="${PYTHON:-/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python}"
GPU_PHYS="${1:-6}"
LIMIT="${2:-50}"
OUT_CSV="${OUT_CSV:-${ROOT}/output/ch5_prior_compare/screen_ckpts_3dpw_limit${LIMIT}.csv}"

export PYTHONPATH="${ROOT}:${HUMANS}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_PHYS}"

DATA_DIR="${HUMANS}/hmr2_evaluation_data"
EVAL_PY="${ROOT}/nvit/skills/evaluate_model/standard_eval.py"

echo "Using PYTHON=${PYTHON} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} limit_batches=${LIMIT}"
echo "Writing CSV: ${OUT_CSV}"
echo "exp,ckpt_rel,mpjpe,pa_mpjpe" > "${OUT_CSV}"

mapfile -t CKPTS < <(find "${ROOT}/output/ch5_prior_compare" -path '*/checkpoints/step_step=*.ckpt' | sort)

for ck in "${CKPTS[@]}"; do
  rel="${ck#${ROOT}/output/ch5_prior_compare/}"
  exp="${rel%%/*}"
  tmp="$(mktemp)"
  "${PYTHON}" "${EVAL_PY}" \
    --ckpt "${ck}" \
    --dataset 3DPW-TEST \
    --gpu 0 \
    --batch_size 32 \
    --num_workers 4 \
    --limit_batches "${LIMIT}" \
    --data_dir "${DATA_DIR}" \
    --output "${tmp}" 2>/dev/null || true
  mpjpe="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); r=d.get('results',{}).get('3DPW-TEST',{}); print(r.get('mode_mpjpe',''))" "${tmp}" 2>/dev/null || echo "")"
  pampjpe="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); r=d.get('results',{}).get('3DPW-TEST',{}); print(r.get('mode_re',''))" "${tmp}" 2>/dev/null || echo "")"
  rm -f "${tmp}"
  echo "${exp},${rel},${mpjpe},${pampjpe}" >> "${OUT_CSV}"
  echo "OK ${exp} $(basename "${ck}") mpjpe=${mpjpe}"
done

echo "Done: ${OUT_CSV}"
