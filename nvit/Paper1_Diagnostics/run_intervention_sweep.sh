#!/bin/bash

# run_intervention_sweep.sh
# Automated sweep for Phase 3 Masking Experiments
# Queued to run on GPUs 0, 1, and 5 to avoid OOM

mkdir -p ft_logs

echo "Launching Control and T2-A-S-Baseline sequentially on GPU 0..."
nohup bash -c 'python train_adaptive_masking.py --group Control --gpu 0 --steps 2000 --batch_size 16 > ft_logs/Control.log 2>&1 && python train_adaptive_masking.py --group T2-A-S-Baseline --gpu 0 --steps 2000 --batch_size 16 > ft_logs/T2-A-S-Baseline.log 2>&1' &

echo "Launching T2-KTI-Adaptive and T2-Static-Late sequentially on GPU 1..."
nohup bash -c 'python train_adaptive_masking.py --group T2-KTI-Adaptive --gpu 1 --steps 2000 --batch_size 16 > ft_logs/T2-KTI-Adaptive.log 2>&1 && python train_adaptive_masking.py --group T2-Static-Late --gpu 1 --steps 2000 --batch_size 16 > ft_logs/T2-Static-Late.log 2>&1' &

echo "Launching T2-A-H-Baseline and T2-Static-Mid sequentially on GPU 5..."
nohup bash -c 'python train_adaptive_masking.py --group T2-A-H-Baseline --gpu 5 --steps 2000 --batch_size 16 > ft_logs/T2-A-H-Baseline.log 2>&1 && python train_adaptive_masking.py --group T2-Static-Mid --gpu 5 --steps 2000 --batch_size 16 > ft_logs/T2-Static-Mid.log 2>&1' &

echo "All 6 experiments queued on GPUs 0, 1, and 5. Monitor ft_logs/ for progress."
