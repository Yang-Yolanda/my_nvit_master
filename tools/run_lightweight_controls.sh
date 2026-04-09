#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

export PROJECT_ROOT="/home/yangz/NViT-master"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/nvit_env.sh"
export PYTHONPATH="$PROJECT_ROOT/nvit/Code_Paper2_Implementation:/home/yangz/4D-Humans:$PYTHONPATH"

export DATA_ROOT_3DPW="/home/yangz/4D-Humans/hmr2_evaluation_data"
export DATA_ROOT_H36M="/home/yangz/4D-Humans/hmr2_evaluation_data"
export CKPT_BASELINE="/home/yangz/NViT-master/nvit/Paper1_Diagnostics/checkpoints/ft_T2-A-H-Baseline.ckpt"

# 你的 provisional ours（用于最后合并到一张表）
export CKPT_OURS="/home/yangz/NViT-master/logs/train/runs/2026-04-07_17-28-22/checkpoints/last.ckpt"

OUT="$PROJECT_ROOT/artifacts/lightweight_controls_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"/{logs,results}

echo "OUT=$OUT" | tee "$OUT/logs/driver.log"
echo "python=$(which python)" | tee -a "$OUT/logs/driver.log"
python -V | tee -a "$OUT/logs/driver.log"

# 统一让 run_eval_suite_final.sh 读取我们传入的 ckpt/npz/output_dir
export NVIT_CKPT_BASELINE="$CKPT_BASELINE"
export NVIT_DATA_3DPW="$DATA_ROOT_3DPW/3dpw_test.npz"
export NVIT_DATA_H36M="$DATA_ROOT_H36M/h36m_val_p2.npz"

CSV="$OUT/results/lightweight_controls.csv"
echo "Method,Dataset,MPJPE_root_hipmean(mm),PA-MPJPE(mm),MPJPE_abs(mm)" > "$CSV"

run_one () {
  local TAG="$1"
  local CKPT="$2"
  local EXTRA_ENV="$3"

  local RUN_DIR="$OUT/$TAG"
  mkdir -p "$RUN_DIR/logs"

  echo "[RUN] $TAG" | tee -a "$OUT/logs/driver.log"

  # 清理互斥变量
  unset NVIT_TRUNCATE_BLOCKS NVIT_EARLY_EXIT_BLOCK
  # 应用本次设置
  eval "$EXTRA_ENV"

  export NVIT_OUTPUT_DIR="$RUN_DIR"

  bash "$PROJECT_ROOT/run_eval_suite_final.sh" \
    |& tee "$RUN_DIR/logs/run_eval_suite_final.log"

  # 解析四个日志（假设你们 run_eval_suite_final.sh 生成了这些文件名；若不同，改这里的文件名）
  python "$PROJECT_ROOT/tools/parse_metrics.py" \
    --log "$RUN_DIR/logs/full_3dpw_hmr2baseline.log" --method "$TAG" --dataset "3DPW(test)" \
    --out "$CSV" --append

  python "$PROJECT_ROOT/tools/parse_metrics.py" \
    --log "$RUN_DIR/logs/full_h36m_hmr2baseline.log" --method "$TAG" --dataset "H36M(val_p2)" \
    --out "$CSV" --append
}

# 0) baseline full（32 blocks）参考点
run_one "Baseline(32blocks)" "$CKPT_BASELINE" ""

# 1) Truncation：k=8,12,16,20,24
for k in 8 12 16 20 24; do
  run_one "Trunc(k=$k)" "$CKPT_BASELINE" "export NVIT_TRUNCATE_BLOCKS=$k"
done

# 2) Early-exit：exit=8,12,16,20,24
for k in 8 12 16 20 24; do
  run_one "EarlyExit(k=$k)" "$CKPT_BASELINE" "export NVIT_EARLY_EXIT_BLOCK=$k"
done

# 3) Ours (provisional) 参考点
echo "[RUN] Ours(provisional)" | tee -a "$OUT/logs/driver.log"
unset NVIT_TRUNCATE_BLOCKS NVIT_EARLY_EXIT_BLOCK
export NVIT_CKPT_BASELINE="$CKPT_OURS"
export NVIT_OUTPUT_DIR="$OUT/Ours_Provisional"
mkdir -p "$NVIT_OUTPUT_DIR/logs"
bash "$PROJECT_ROOT/run_eval_suite_final.sh" \
  |& tee "$NVIT_OUTPUT_DIR/logs/run_eval_suite_final.log"

python "$PROJECT_ROOT/tools/parse_metrics.py" \
  --log "$NVIT_OUTPUT_DIR/logs/full_3dpw_hmr2baseline.log" --method "AdaptiveNViT(Provisional)" --dataset "3DPW(test)" \
  --out "$CSV" --append

python "$PROJECT_ROOT/tools/parse_metrics.py" \
  --log "$NVIT_OUTPUT_DIR/logs/full_h36m_hmr2baseline.log" --method "AdaptiveNViT(Provisional)" --dataset "H36M(val_p2)" \
  --out "$CSV" --append

echo "[DONE] CSV=$CSV" | tee -a "$OUT/logs/driver.log"
cp "$CSV" /home/yangz/NViT-master/artifacts/lightweight_controls_latest.csv
head -n 30 "$CSV" | tee "$OUT/logs/csv_preview.txt"
