#!/bin/bash

# Configuration
SESSION_NAME="paper1_diagnostics"
NUM_BATCHES=50
MODELS=("HMR2" "HSMR" "PromptHMR" "CameraHMR")

# Create a new tmux session or attach to existing one
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? != 0 ]; then
    echo "Creating new tmux session: $SESSION_NAME"
    tmux new-session -d -s $SESSION_NAME
fi

# Run diagnostics for each model in a separate window
for model in "${MODELS[@]}"; do
    WINDOW_NAME="${model,,}_diag"
    echo "Starting diagnostics for $model in window $WINDOW_NAME..."
    
    # Check if window exists, if not create it
    tmux list-windows -t $SESSION_NAME | grep -q $WINDOW_NAME
    if [ $? != 0 ]; then
        tmux new-window -t $SESSION_NAME -n $WINDOW_NAME
    fi
    
    # Send the command to the window
    # We use the absolute path to the python env and script
    tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" "cd /home/yangz/NViT-master/nvit/Paper1_Diagnostics" C-m
    tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" "chmod +x metrics_router.sh && ./metrics_router.sh $model $NUM_BATCHES" C-m
done

echo "Diagnostics started in tmux session: $SESSION_NAME"
echo "Use 'tmux attach -t $SESSION_NAME' to monitor progress."
