#!/bin/bash
# NViT Phase 2 Self-Healing Watchdog (6-GPU DDP)
# GPUs: 0, 1, 4, 5, 6, 7 (2 & 3 RESERVED)

LOG_FILE="/home/yangz/NViT-master/outputs/watchdog.log"
BASE_DIR="/home/yangz/NViT-master"

echo "[$(date)] --- Watchdog Started (6-GPU Pool) ---" | tee -a "$LOG_FILE"

while true; do
    # Check if the training process is alive (search for the specific training script)
    if ! pgrep -f "train_guided.py" > /dev/null; then
        echo "[$(date)] ⚠️ Training stopped! Relaunching via standardized script..." | tee -a "$LOG_FILE"
        
        # Call the unified launcher with resume support
        cd "$BASE_DIR" || exit
        bash nvit/run_full_ddp.sh --resume >> "$BASE_DIR/outputs/phase2_training.log" 2>&1 &
        
        NEW_PID=$!
        echo "[$(date)] 🚀 Relaunched Phase 2. New PID (Launcher): $NEW_PID" | tee -a "$LOG_FILE"
        
        # Update project interaction log
        echo "- **Watchdog Action**: Restarted 6-GPU training at $(date) (PID: $NEW_PID)" >> "$BASE_DIR/nvit/INTERACTION_LOG.md"
    fi
    
    # Check every 2 minutes for faster response
    sleep 120
done
