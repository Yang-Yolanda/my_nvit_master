#!/bin/bash
# Single GPU Launcher for Run 5 (Mamba=seq, GCN=random)
# User requested single card execution due to DDP issues.

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Run 5 Config
MAMBA="seq"
GCN="random"
EPOCHS=3
BATCH_SIZE=16  # Single GPU Batch Size

echo "🚀 Launching Run 5 (Single GPU): Mamba=${MAMBA}, GCN=${GCN} on GPU 0..."
echo "Log: nvit/Code_Paper2_Implementation/Expt2_Finetuning/logs/train_run5_single.log"

/home/yangz/.conda/envs/4D-humans/bin/python nvit/Code_Paper2_Implementation/Expt2_Finetuning/train.py \
    --mamba ${MAMBA} \
    --gcn ${GCN} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --devices 1 \
    > nvit/Code_Paper2_Implementation/Expt2_Finetuning/logs/train_run5_single.log 2>&1

echo "✅ Run 5 (Single) Completed."
