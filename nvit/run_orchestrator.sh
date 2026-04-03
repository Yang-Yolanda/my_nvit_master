#!/bin/bash

LOG_FILE="NViT-master/nvit/loop.log"
STATUS_FILE="NViT-master/nvit/status.txt"
FINETUNE_SCRIPT="NViT-master/nvit/finetune_dense.py"
BASE_OUTPUT_DIR="NViT-master/nvit/output/hmr2_ddp_v1"

echo "[Fri Jan  2 10:06:32 AM CST 2026] Orchestrator V3 (Batch Mode) started..." | tee -a $STATUS_FILE

while true; do
    if grep -q "SAVING_DONE_SAFE_TO_KILL" "$LOG_FILE"; then
        echo "[Fri Jan  2 10:06:32 AM CST 2026] ✅ Pruning Batch Complete detected!" | tee -a $STATUS_FILE
        
        # 1. Cleanup Pruning Process
        echo "[Fri Jan  2 10:06:32 AM CST 2026] Cleaning up pruning process..." | tee -a $STATUS_FILE
        pkill -f "4d_humans.py"
        sleep 5 

        # 2. Batch Finetuning Loop
        SPARSITIES=(40 60 80)
        
        for S in "${SPARSITIES[@]}"; do
            CHECKPOINT="$BASE_OUTPUT_DIR/hmr2_pruned_sparsity_${S}.pth"
            OUTPUT="NViT-master/nvit/output/hmr2_finetuned_${S}"
            LOG="NViT-master/nvit/finetune_${S}.log"
            
            if [ -f "$CHECKPOINT" ]; then
                echo "[Fri Jan  2 10:06:32 AM CST 2026] 🚀 Starting Finetuning for Sparsity ${S}%..." | tee -a $STATUS_FILE
                echo "Input: $CHECKPOINT" | tee -a $STATUS_FILE
                
                mkdir -p $OUTPUT
                
                # Run Synchronously (wait for each to finish)
                CUDA_VISIBLE_DEVICES=0 /home/yangz/.conda/envs/4D-humans/bin/python $FINETUNE_SCRIPT \
                    --checkpoint-path "$CHECKPOINT" \
                    --output_dir "$OUTPUT" \
                    > "$LOG" 2>&1
                
                if [ $? -eq 0 ]; then
                    echo "[Fri Jan  2 10:06:32 AM CST 2026] ✅ Finetuning ${S}% Completed." | tee -a $STATUS_FILE
                else
                    echo "[Fri Jan  2 10:06:32 AM CST 2026] ❌ Finetuning ${S}% Failed! Check $LOG." | tee -a $STATUS_FILE
                fi
            else
                echo "[Fri Jan  2 10:06:32 AM CST 2026] ⚠️ Checkpoint for ${S}% not found, skipping." | tee -a $STATUS_FILE
            fi
        done
        
        echo "[Fri Jan  2 10:06:32 AM CST 2026] 🎉 All Batch Tasks Completed." | tee -a $STATUS_FILE
        exit 0
    fi
    sleep 60
done
