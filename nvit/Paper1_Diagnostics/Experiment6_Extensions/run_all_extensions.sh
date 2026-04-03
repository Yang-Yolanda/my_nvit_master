#!/bin/bash

# Configuration
CONDA_PY="/home/yangz/.conda/envs/4D-humans/bin/python"
EXTERNAL_ROOT="/home/yangz/NViT-master/nvit/external_models"
RESULTS_DIR="/home/yangz/NViT-master/nvit/Paper1_Diagnostics/Experiment6_Extensions/results"

# 1. Create centralized results directory
mkdir -p "$RESULTS_DIR"

# 1.5 Set PYTHONPATH to include the root and diagnostic core
export PYTHONPATH=$PYTHONPATH:/home/yangz/NViT-master:/home/yangz/NViT-master/nvit:/home/yangz/NViT-master/nvit/Paper1_Diagnostics:/home/yangz/NViT-master/nvit/Paper1_Diagnostics/diagnostic_core

echo "=========================================================="
echo " Starting Generalized Extension Diagnostics (Phase 3)"
echo " Output Destination: $RESULTS_DIR"
echo "=========================================================="

# 2. Run Animer, HaMeR, and YOLOPose (Animal, Hand, COCO)
echo -e "\n[1/2] Running Animal and Hand Diagnostics (AniMeR / HaMeR / YOLOPose)..."
cd "$EXTERNAL_ROOT"
$CONDA_PY run_external_diagnostics.py
if [ $? -ne 0 ]; then
    echo "❌ Error encountered in Animal/Hand diagnostics!"
else
    echo "✅ Animal and Hand diagnostics complete."
fi

# 3. Run Robotic Arm Simulation
echo -e "\n[2/2] Running Robotic Arm Diagnostics (Panda URDF Simulation)..."
cd "$EXTERNAL_ROOT/TaskB_Robot"
$CONDA_PY runner.py --epochs 15
if [ $? -ne 0 ]; then
    echo "❌ Error encountered in Robotic Arm diagnostics!"
else
    echo "✅ Robotic Arm diagnostics complete."
fi

echo -e "\n=========================================================="
echo " All Extension Experiments Finished!"
echo " Check $RESULTS_DIR for the aggregate CSV and JSON/PNG outputs."
echo "=========================================================="
