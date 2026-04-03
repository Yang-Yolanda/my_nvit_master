#!/bin/bash
# run_robustness_sweep.sh
# Run occlusion robustness tests for Control and T2-KTI-Adaptive across multiple occlusion levels.

mkdir -p robust_logs

# Occlusion ratios to test: 0.1, 0.2, 0.3 (10%, 20%, 30% of side length)
# We focus on the two most important groups to prove the point.

# GPU 0: Control, Occ 0.1
# GPU 1: KTI-Adaptive, Occ 0.1
# GPU 2: Control, Occ 0.2
# GPU 3: KTI-Adaptive, Occ 0.2
# GPU 4: Control, Occ 0.3
# GPU 5: KTI-Adaptive, Occ 0.3

echo "Launching Robustness Occlusion Sweep..."

python eval_robustness_occlusion.py --group Control --occlusion 0.1 --gpu 0 > robust_logs/Control_occ01.log 2>&1 &
python eval_robustness_occlusion.py --group T2-KTI-Adaptive --occlusion 0.1 --gpu 1 > robust_logs/Adaptive_occ01.log 2>&1 &

python eval_robustness_occlusion.py --group Control --occlusion 0.2 --gpu 2 > robust_logs/Control_occ02.log 2>&1 &
python eval_robustness_occlusion.py --group T2-KTI-Adaptive --occlusion 0.2 --gpu 3 > robust_logs/Adaptive_occ02.log 2>&1 &

python eval_robustness_occlusion.py --group Control --occlusion 0.3 --gpu 4 > robust_logs/Control_occ03.log 2>&1 &
python eval_robustness_occlusion.py --group T2-KTI-Adaptive --occlusion 0.3 --gpu 5 > robust_logs/Adaptive_occ03.log 2>&1 &

echo "All 6 robustness jobs launched. Monitor with tail -f robust_logs/*.log"
