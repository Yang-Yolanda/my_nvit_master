#!/bin/bash

# Ensure OS tricks
export MALLOC_ARENA_MAX=2
export NCCL_SHM_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "🔍 Sweeping Batch Sizes for Single Card Maximum Performance on RTX 3090..."

for BS in 256 512 768 1024 1536 2048; do
    echo "======================================"
    echo ">> Testing BATCH_SIZE = $BS"
    
    CUDA_VISIBLE_DEVICES=0 python nvit/train_guided.py \
        experiment=hmr_vit_transformer data=mix_all \
        trainer.accelerator=gpu trainer.devices=1 trainer.num_nodes=1 \
        ++trainer.precision="bf16-mixed" \
        ++TRAIN.BATCH_SIZE=$BS \
        ++GENERAL.NUM_WORKERS=8 \
        ++GENERAL.PREFETCH_FACTOR=3 \
        ++TRAIN.SHUFFLE_BUFFER=1000 \
        ++TRAIN.ACCUMULATE_GRAD_BATCHES=1 \
        ++trainer.limit_train_batches=15 \
        ++trainer.max_epochs=1 \
        ++trainer.enable_checkpointing=false \
        ++trainer.logger=false > /tmp/test_limit.log 2>&1
        
    if grep -q "CUDA out of memory" /tmp/test_limit.log || grep -q "Killed" /tmp/test_limit.log; then
        echo "❌ FAILED at Batch Size $BS (Out of VRAM or memory)."
        break
    else
        # Extract throughput or time if possible
        echo "✅ PASSED Batch Size $BS!"
        grep "train/loss_step" /tmp/test_limit.log | tail -n 1
        MAX_PASS=$BS
    fi
done

echo "======================================"
echo "🎯 The Single-Card Maximum Batch Size for NViT is: $MAX_PASS"
