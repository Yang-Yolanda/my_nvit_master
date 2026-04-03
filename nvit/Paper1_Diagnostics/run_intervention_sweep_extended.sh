#!/bin/bash
# run_intervention_sweep_extended.sh
# Extended fine-tuning sweep for 20,000 steps across 6 GPUs in parallel.

mkdir -p ft_logs

experiments=(
    "Control:0"
    "T2-KTI-Adaptive:1"
    "T2-A-H-Baseline:2"
    "T2-A-S-Baseline:3"
    "T2-Static-Late:4"
    "T2-Static-Mid:5"
)

echo "Starting Extended (20k steps) Fine-tuning Sweep in parallel..."

for exp in "${experiments[@]}"; do
    group="${exp%%:*}"
    gpu="${exp##*:}"
    echo "Launching Extended FT for $group on GPU $gpu..."
    # Save the new checkpoints with a _20k suffix to distinguish from the 2k ones if needed
    python train_adaptive_masking.py --group "$group" --gpu $gpu --steps 20000 > "ft_logs/${group}_20k.log" 2>&1 &
done

echo "All 6 extended fine-tuning experiments launched successfully."
echo "You can monitor the progress with: tail -f ft_logs/*_20k.log"
