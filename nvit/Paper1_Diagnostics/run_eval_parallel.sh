#!/bin/bash
# Parallel evaluation on available GPUs (0, 1, 2, 3, 4, 5)

mkdir -p eval_logs

experiments=(
    "Control:0"
    "T2-KTI-Adaptive:1"
    "T2-A-H-Baseline:2"
    "T2-A-S-Baseline:3"
    "T2-Static-Late:4"
    "T2-Static-Mid:5"
)

echo "Starting parallel evaluation..."

for exp in "${experiments[@]}"; do
    group="${exp%%:*}"
    gpu="${exp##*:}"
    echo "Launching Eval for $group on GPU $gpu..."
    python eval_adaptive_masking.py --group "$group" --gpu $gpu > "eval_logs/eval_$group.log" 2>&1 &
done

echo "All evaluations launched in parallel."
echo "You can check progress with: tail -f eval_logs/*.log"
wait
echo "All parallel evaluations completed. Results saved to finetune_eval_results.csv."
