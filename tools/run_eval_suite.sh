#!/bin/bash
export PROJECT_ROOT="/home/yangz/NViT-master"
export OUTPUT_DIR="$PROJECT_ROOT/artifacts/nvit_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"/{logs,configs,results,adapters,baselines,figs}
echo "OUTPUT_DIR=$OUTPUT_DIR"

export DATA_ROOT_3DPW="/home/yangz/4D-Humans/hmr2_evaluation_data"
export DATA_ROOT_H36M="/home/yangz/4D-Humans/hmr2_evaluation_data"
export CKPT_HMR2_BASELINE="/home/yangz/NViT-master/nvit/Paper1_Diagnostics/checkpoints/ft_T2-A-H-Baseline.ckpt"
export CKPT_OURS="/home/yangz/NViT-master/nvit/Paper1_Diagnostics/checkpoints/ft_T2-KTI-Adaptive.ckpt"
export PYTHONPATH=$PROJECT_ROOT/nvit/Code_Paper2_Implementation:/home/yangz/4D-Humans:$PYTHONPATH
export PYTHON_CMD="/home/yangz/.conda/envs/4D-humans/bin/python"

cat > "$OUTPUT_DIR/env.sh" << EOF
export PROJECT_ROOT="$PROJECT_ROOT"
export OUTPUT_DIR="$OUTPUT_DIR"
export DATA_ROOT_3DPW="$DATA_ROOT_3DPW"
export DATA_ROOT_H36M="$DATA_ROOT_H36M"
export CKPT_HMR2_BASELINE="$CKPT_HMR2_BASELINE"
export CKPT_OURS="$CKPT_OURS"
EOF

# Make tools directory and parse_metrics if it doesn't exist
mkdir -p $PROJECT_ROOT/tools

echo "Starting evaluation suite..."
echo "1. Baseline on 3DPW"
$PYTHON_CMD nvit/skills/evaluate_model/standard_eval.py --ckpt "$CKPT_HMR2_BASELINE" --dataset 3DPW-TEST --batch_size 32 > "$OUTPUT_DIR/logs/full_3dpw_hmr2baseline.log" 2>&1

echo "2. Ours on 3DPW"
$PYTHON_CMD nvit/skills/evaluate_model/standard_eval.py --ckpt "$CKPT_OURS" --dataset 3DPW-TEST --batch_size 32 > "$OUTPUT_DIR/logs/full_3dpw_ours.log" 2>&1

echo "3. Baseline on H36M"
$PYTHON_CMD nvit/skills/evaluate_model/standard_eval.py --ckpt "$CKPT_HMR2_BASELINE" --dataset H36M-VAL-P2 --batch_size 32 > "$OUTPUT_DIR/logs/full_h36m_hmr2baseline.log" 2>&1

echo "4. Ours on H36M"
$PYTHON_CMD nvit/skills/evaluate_model/standard_eval.py --ckpt "$CKPT_OURS" --dataset H36M-VAL-P2 --batch_size 32 > "$OUTPUT_DIR/logs/full_h36m_ours.log" 2>&1

echo "Parsing metrics..."

$PYTHON_CMD $PROJECT_ROOT/tools/parse_metrics.py --log "$OUTPUT_DIR/logs/full_3dpw_hmr2baseline.log" --method "HMR2(Baseline)" --dataset "3DPW(test)" --out "$OUTPUT_DIR/results/main_table_3dpw.csv"
$PYTHON_CMD $PROJECT_ROOT/tools/parse_metrics.py --log "$OUTPUT_DIR/logs/full_3dpw_ours.log" --method "AdaptiveNViT(Ours)" --dataset "3DPW(test)" --out "$OUTPUT_DIR/results/main_table_3dpw.csv" --append

$PYTHON_CMD $PROJECT_ROOT/tools/parse_metrics.py --log "$OUTPUT_DIR/logs/full_h36m_hmr2baseline.log" --method "HMR2(Baseline)" --dataset "H36M(val_p2)" --out "$OUTPUT_DIR/results/main_table_h36m.csv"
$PYTHON_CMD $PROJECT_ROOT/tools/parse_metrics.py --log "$OUTPUT_DIR/logs/full_h36m_ours.log" --method "AdaptiveNViT(Ours)" --dataset "H36M(val_p2)" --out "$OUTPUT_DIR/results/main_table_h36m.csv" --append

cp "$OUTPUT_DIR/results/main_table_3dpw.csv" "$PROJECT_ROOT/artifacts/main_table_3dpw.csv"
cp "$OUTPUT_DIR/results/main_table_h36m.csv" "$PROJECT_ROOT/artifacts/main_table_h36m.csv"

echo "Evaluation suite complete."
