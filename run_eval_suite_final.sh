#!/bin/bash
set -euo pipefail

# ========================
# [0] 固定环境（禁止漂移）
# ========================
export PY=/home/yangz/.conda/envs/4D-humans/bin/python
export PYTHONPATH=/home/yangz/NViT-master/nvit/Code_Paper2_Implementation:/home/yangz/4D-Humans:$PYTHONPATH

export PROJECT_ROOT="/home/yangz/NViT-master"
export OUTPUT_DIR="${NVIT_OUTPUT_DIR:-$PROJECT_ROOT/artifacts/nvit_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"/{logs,results,configs}

export DATA_ROOT_3DPW="${NVIT_DATA_3DPW:-/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz}"
export DATA_ROOT_H36M="${NVIT_DATA_H36M:-/home/yangz/4D-Humans/hmr2_evaluation_data/h36m_val_p2.npz}"
export CKPT_HMR2_BASELINE="${NVIT_CKPT_BASELINE:-/home/yangz/NViT-master/nvit/Paper1_Diagnostics/checkpoints/ft_T2-A-H-Baseline.ckpt}"
# Ensure we use strictly the directory path for standard_eval.py since it expects dataset names, not direct npz files:
export DIR_DATA_3DPW=$(dirname "$DATA_ROOT_3DPW")
export DIR_DATA_H36M=$(dirname "$DATA_ROOT_H36M")
export CKPT_OURS="/home/yangz/NViT-master/nvit/Paper1_Diagnostics/checkpoints/ft_T2-KTI-Adaptive.ckpt"

ls -lh $DATA_ROOT_3DPW/3dpw_test.npz
ls -lh $DATA_ROOT_H36M/h36m_val_p2.npz

$PY -V | tee $OUTPUT_DIR/logs/python_version.txt
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available())" | tee $OUTPUT_DIR/logs/torch_info.txt

# ========================
# [4] 跑 full（baseline vs ours，3DPW + H36M）
# ========================
echo "Starting FULL evaluation..."

echo "1. 3DPW Baseline"
$PY nvit/skills/evaluate_model/standard_eval.py --ckpt $CKPT_HMR2_BASELINE --dataset 3DPW-TEST --batch_size 16 | tee $OUTPUT_DIR/logs/full_3dpw_hmr2baseline.log

echo "2. 3DPW Ours"
$PY nvit/skills/evaluate_model/standard_eval.py --ckpt $CKPT_OURS --dataset 3DPW-TEST --batch_size 16 | tee $OUTPUT_DIR/logs/full_3dpw_ours.log

echo "3. H36M Baseline"
$PY nvit/skills/evaluate_model/standard_eval.py --ckpt $CKPT_HMR2_BASELINE --dataset H36M-VAL-P2 --batch_size 16 | tee $OUTPUT_DIR/logs/full_h36m_hmr2baseline.log

echo "4. H36M Ours"
$PY nvit/skills/evaluate_model/standard_eval.py --ckpt $CKPT_OURS --dataset H36M-VAL-P2 --batch_size 16 | tee $OUTPUT_DIR/logs/full_h36m_ours.log

# ========================
# [5] 结果落盘：从日志抽取 MPJPE/PA-MPJPE，生成 CSV
# ========================
echo "Generating CSVs..."

$PY tools/parse_metrics.py --log $OUTPUT_DIR/logs/full_3dpw_hmr2baseline.log --method "HMR2(Baseline)" --dataset "3DPW(test)" --out $OUTPUT_DIR/results/main_table_3dpw.csv
$PY tools/parse_metrics.py --log $OUTPUT_DIR/logs/full_3dpw_ours.log --method "AdaptiveNViT(Ours)" --dataset "3DPW(test)" --out $OUTPUT_DIR/results/main_table_3dpw.csv --append
$PY tools/parse_metrics.py --log $OUTPUT_DIR/logs/full_h36m_hmr2baseline.log --method "HMR2(Baseline)" --dataset "H36M(val_p2)" --out $OUTPUT_DIR/results/main_table_h36m.csv
$PY tools/parse_metrics.py --log $OUTPUT_DIR/logs/full_h36m_ours.log --method "AdaptiveNViT(Ours)" --dataset "H36M(val_p2)" --out $OUTPUT_DIR/results/main_table_h36m.csv --append

cp $OUTPUT_DIR/results/main_table_3dpw.csv /home/yangz/NViT-master/artifacts/main_table_3dpw.csv
cp $OUTPUT_DIR/results/main_table_h36m.csv /home/yangz/NViT-master/artifacts/main_table_h36m.csv

echo "Pipeline complete! Artifacts saved."
