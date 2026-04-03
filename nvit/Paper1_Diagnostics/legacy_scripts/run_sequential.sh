#!/bin/bash
echo "Starting Full Benchmark Suite (Paper 1)..."
echo "Logical Flow: Diagnosis (Exp1: Entropy -> Exp2: KTI) -> Intervention (Exp3: Masking)"

# Exclude HMR2, CameraHMR (Already Done)
MODELS=("HSMR" "PromptHMR")

# --- Phase 1: Experiment 1 (Entropy Analysis - "Redundancy Verification") ---
echo "=== Phase 1: Experiment 1 (Entropy Analysis) ==="
for model in "${MODELS[@]}"; do
    echo "Running Exp1 (Entropy) for $model..."
    # 20 Batches is enough for stable statistical mean
    /home/yangz/.conda/envs/4D-humans/bin/python Experiment1_Entropy/runner_fixed.py --model $model --num_batches 20 > "logs/exp1_${model,,}.log" 2>&1
    echo "Exp1 $model Done."
done
echo ">> Generating Entropy Curves..."
/home/yangz/.conda/envs/4D-humans/bin/python Experiment1_Entropy/visualize_entropy.py

# --- Phase 2: Experiment 2 (KTI Analysis - "Structure Deficit Verification") ---
echo "=== Phase 2: Experiment 2 (KTI Analysis) ==="
for model in "${MODELS[@]}"; do
    echo "Running Exp2 (KTI) for $model..."
    /home/yangz/.conda/envs/4D-humans/bin/python Experiment2_KTI/runner_fixed.py --model $model --num_batches 20 > "logs/exp2_${model,,}.log" 2>&1
    echo "Exp2 $model Done."
done
echo ">> Generating KTI Curves..."
/home/yangz/.conda/envs/4D-humans/bin/python Experiment2_KTI/visualize_kti.py

# --- Phase 3: Experiment 3 (Masking Sweep - "Structure Injection Validation") ---
echo "=== Phase 3: Experiment 3 (Masking / Topology Injection) ==="
for model in "${MODELS[@]}"; do
    echo "Running Exp3 (Masking Sweep) for $model..."
    /home/yangz/.conda/envs/4D-humans/bin/python Experiment3_Masking/runner_fixed.py --model $model --num_batches 5 > "logs/exp3_${model,,}.log" 2>&1
    echo "Exp3 $model Done."
done

echo "All Experiments & Visualizations Finished."
