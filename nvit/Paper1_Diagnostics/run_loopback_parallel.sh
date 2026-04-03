#!/bin/bash
# run_loopback_parallel.sh
# Phase 5: Run diagnostic loopback for all groups + baseline on GPUs 4-7.

mkdir -p loopback_diagnostics

echo "Phase 5: Starting Diagnostic Loopback on GPUs 4-7..."

CUDA_VISIBLE_DEVICES=4 python run_loopback_diagnostics.py \
    --groups Pretrained_Baseline Control \
    --gpu 4 --n_batches 50 \
    > loopback_diagnostics/loopback_baseline_control.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python run_loopback_diagnostics.py \
    --groups T2-KTI-Adaptive \
    --gpu 5 --n_batches 50 \
    > loopback_diagnostics/loopback_kti_adaptive.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python run_loopback_diagnostics.py \
    --groups T2-A-H-Baseline T2-A-S-Baseline \
    --gpu 6 --n_batches 50 \
    > loopback_diagnostics/loopback_ablations.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python run_loopback_diagnostics.py \
    --groups T2-Static-Late T2-Static-Mid \
    --gpu 7 --n_batches 50 \
    > loopback_diagnostics/loopback_static.log 2>&1 &

echo "All loopback diagnostic jobs launched on GPUs 4-7."
echo "Monitor with: tail -f loopback_diagnostics/*.log"
echo "After completion, run: python plot_loopback_comparison.py"
