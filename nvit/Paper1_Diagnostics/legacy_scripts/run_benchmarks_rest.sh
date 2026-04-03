#!/bin/bash

# Activate environment (if needed, usually done by shell)
# source activate 4D-humans

echo "Starting ALL Benchmarks with FIXED RUNNER (Re-verification)..."

# 1. HMR2 (Baseline - Re-run to verify Hybrid Fix)
echo "1. Re-running HMR2..."
python Experiment3_Masking/runner_fixed.py --model 4D-Humans --num_batches 10 > hmr2_final.log 2>&1
echo "HMR2 Finished."

# 2. CameraHMR
echo "2. Running CameraHMR..."
python Experiment3_Masking/runner_fixed.py --model CameraHMR --num_batches 10 > camerahmr_final.log 2>&1
echo "CameraHMR Finished."

# 3. HSMR
echo "3. Running HSMR..."
python Experiment3_Masking/runner_fixed.py --model HSMR --num_batches 10 > hsmr_final.log 2>&1
echo "HSMR Finished."

# 4. PromptHMR (Running in background separately, but adding here just in case)
# echo "4. Running PromptHMR..."
# python Experiment3_Masking/runner_fixed.py --model PromptHMR --num_batches 10 > prompthmr_final.log 2>&1
# echo "PromptHMR Finished."

echo "All Experiments Completed."
