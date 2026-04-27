#!/usr/bin/env bash
# 后台跑：1) 本机 HMR2 multirun checkpoint 的 standard_eval (ALL) → json
#         2) 多数据集对比图 + 2D 四指标小表
#         3) 与 Ch6 best 的推理速度对比
# 日志：stdout/stderr 由调用方重定向，例如 nohup ... > logs/xxx.log 2>&1
set -euo pipefail
ROOT="/cpfs_infra/shared/yangz/NViT-master"
cd "$ROOT"
export PYTHONPATH="${ROOT}:/cpfs_infra/shared/yangz/4D-Humans${PYTHONPATH:+:$PYTHONPATH}"
export HUMANS_ROOT="${HUMANS_ROOT:-/cpfs_infra/shared/yangz/4D-Humans}"
PY="${PYTHON:-/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python}"
DATA_DIR="${HMR2_EVAL_DATA_DIR:-${HUMANS_ROOT}/hmr2_evaluation_data}"
GPU="${GPU:-0}"

HMR2_CKPT="${HMR2_CKPT:-/cpfs_infra/shared/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt}"
CH6_CKPT="${CH6_CKPT:-/mnt/yangz/nvit_output/ch6/train/runs/2026-04-17_13-28-24/checkpoints/step_step=492000.ckpt}"

HMR2_JSON_OUT="${HMR2_JSON_OUT:-${ROOT}/artifacts/eval_unified/json/hmr2_multirun_epoch35_step1M.json}"
BENCH_CSV="${BENCH_CSV_OUT:-${ROOT}/outputs/eval_global/Ch6A/hmr2_vs_nvit_bench.csv}"

echo "=== [1/4] standard_eval HMR2 (ALL) ===" 
echo "ckpt=$HMR2_CKPT"
echo "out=$HMR2_JSON_OUT"
if [[ -f "$HMR2_JSON_OUT" && "${FORCE_HMR2_EVAL:-0}" != "1" ]]; then
  echo "已存在 $HMR2_JSON_OUT ，跳过 standard_eval（设 FORCE_HMR2_EVAL=1 可强制重跑）"
else
  "$PY" -m nvit.skills.evaluate_model.standard_eval \
    --ckpt "$HMR2_CKPT" \
    --dataset ALL \
    --gpu "$GPU" \
    --use_mean_alignment \
    --data_dir "$DATA_DIR" \
    --output "$HMR2_JSON_OUT"
fi

echo "=== [2/4] plot: 文献 + HMR2 + Ch6 best ===" 
"$PY" "$ROOT/artifacts/plot_ch6_dataset_effects_with_baselines.py" \
  --hmr2-baseline-json "$HMR2_JSON_OUT" \
  --hmr2-baseline-label "HMR2 (4DH multirun e35 1M)"

echo "=== [3/4] 2D 表: 文献 + HMR2 + Ch6 best ===" 
"$PY" "$ROOT/artifacts/ch6_baseline_vs_best_compare.py" \
  --hmr2-json "$HMR2_JSON_OUT" \
  --hmr2-label "HMR2 (4DH multirun)" \
  --hmr2-ckpt "$HMR2_CKPT" \
  --hmr2-params-m 213

echo "=== [4/4] inference speed HMR2 vs Ch6 best ===" 
"$PY" "$ROOT/artifacts/bench_hmr2_vs_nvit_inference.py" \
  --ckpt-a "$HMR2_CKPT" \
  --ckpt-b "$CH6_CKPT" \
  --label-a "HMR2_multirun" \
  --label-b "NViT_ch6_best" \
  --gpu "$GPU" \
  --batch 1 \
  --iters 100 \
  --out-csv "$BENCH_CSV"

echo "=== [5/6] METRO / MeshG 推理（legacy env，需 METRO_PICKLED_CKPT / MESHG_PICKLED_CKPT）==="
bash "$ROOT/artifacts/run_metro_meshg_speed.sh"

echo "=== [6/6] 合并推理速度柱状图 ===" 
"$PY" "$ROOT/artifacts/plot_ch6_inference_speed_bars.py"

echo "全部完成。图/表见 outputs/eval_global/Ch6A/ ；HMR2 json: $HMR2_JSON_OUT"
echo "速度: $BENCH_CSV + metro_meshg_inference_speed.csv → ch6_inference_speed_all.png"
