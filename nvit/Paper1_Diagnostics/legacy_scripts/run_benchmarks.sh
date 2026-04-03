#!/bin/bash
echo "Starting Sequential Benchmark Suite..."

echo "1. Running HMR 2.0 (Baseline)..."
/home/yangz/.conda/envs/4D-humans/bin/python Experiment3_Masking/runner.py --model hmr2 --num_batches 10 > hmr2_final.log 2>&1
echo "HMR 2.0 Finished."

echo "2. Running CameraHMR..."
/home/yangz/.conda/envs/4D-humans/bin/python Experiment3_Masking/runner.py --model camerahmr --num_batches 10 > camerahmr_final.log 2>&1
echo "CameraHMR Finished."

echo "3. Running HSMR..."
/home/yangz/.conda/envs/4D-humans/bin/python Experiment3_Masking/runner.py --model hsmr --num_batches 10 > hsmr_final.log 2>&1
echo "HSMR Finished."

echo "4. Running PromptHMR..."
/home/yangz/.conda/envs/4D-humans/bin/python Experiment3_Masking/runner.py --model prompthmr --num_batches 10 > prompthmr_final.log 2>&1
echo "PromptHMR Finished."

echo "All Experiments Completed."
