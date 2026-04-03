#!/bin/bash

# $1 = MODEL NAME
# $2 = NUM BATCHES

MODEL=$1
NUM_BATCHES=$2
BASE_DIR="/home/yangz/NViT-master/nvit/Paper1_Diagnostics"
CONDA_PY="/home/yangz/.conda/envs/4D-humans/bin/python"

cd $BASE_DIR

echo "Starting Unified Fast Diagnostics for $MODEL ($NUM_BATCHES batches)..."
$CONDA_PY run_baseline_diagnostics.py --model $MODEL --num_batches $NUM_BATCHES

SOURCE_JSON="$BASE_DIR/logs/baseline_diagnostics/$MODEL/layer_metrics_Control.json"

if [ -f "$SOURCE_JSON" ]; then
    echo "Routing extracted metrics to specific Experiment folders..."
    
    # --- Experiment 1 ---
    mkdir -p $BASE_DIR/Experiment1_Entropy/results/$MODEL
    cp $SOURCE_JSON $BASE_DIR/Experiment1_Entropy/results/$MODEL/
    echo "Running Visualization for Entropy/Distance..."
    $CONDA_PY $BASE_DIR/Experiment1_Entropy/visualize_entropy.py
    
    # --- Experiment 2 ---
    mkdir -p $BASE_DIR/Experiment2_KTI/results/$MODEL
    cp $SOURCE_JSON $BASE_DIR/Experiment2_KTI/results/$MODEL/
    echo "Running Visualization for KTI..."
    $CONDA_PY $BASE_DIR/Experiment2_KTI/scripts/visualize_kti.py
    
    # --- Experiment 4 ---
    mkdir -p $BASE_DIR/Experiment4_Rank/results/$MODEL
    cp $SOURCE_JSON $BASE_DIR/Experiment4_Rank/results/$MODEL/
    echo "Running Visualization for Rank..."
    $CONDA_PY $BASE_DIR/Experiment4_Rank/visualize_rank.py
    
    echo "All Done for $MODEL! Results saved cleanly into corresponding folders."
else
    echo "CRITICAL WARNING: $SOURCE_JSON was NOT generated! Pipeline crashed for $MODEL."
fi
