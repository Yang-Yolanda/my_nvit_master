# run_full_ddp.sh - Standardized Phase 2 Training Launcher
# Updated: Migrated to 4-GPU (4,5,6,7) with RAM optimization (B=128, A=3)

# 1. System & Resource Hygiene
ulimit -n 65535
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0 

# Auto-Cleanup: Kill zombie training processes before start
echo "🧹 Cleaning up existing training processes..."
pkill -9 -f "train_guided.py"
pkill -9 -f "torchrun"
sleep 2

# 2. Path Setup
BASE_DIR="/home/yangz/NViT-master"
export PYTHONPATH=$PYTHONPATH:$BASE_DIR:$BASE_DIR/../4D-Humans

# 3. Argument Handling (Resume Support)
# Usage: ./run_full_ddp.sh [--resume]
RESUME=false
for arg in "$@"; do
  if [ "$arg" == "--resume" ]; then
    RESUME=true
  fi
done

# 4. Determine Checkpoint for Resume
CKPT_ARG="++ckpt_path=null"
# Default output dir for full_data_run
OUTPUT_DIR="$HOME/NViT-master/outputs/phase2_full_run"
LAST_CKPT="$OUTPUT_DIR/checkpoints/last.ckpt"

if [ "$RESUME" = true ]; then
  if [ -f "$LAST_CKPT" ]; then
    echo "🔄 Resuming from $LAST_CKPT"
    CKPT_ARG="++ckpt_path=$LAST_CKPT"
  else
    echo "⚠️ Warning: --resume flag set but $LAST_CKPT not found. Starting fresh."
  fi
else
    # Auto-resume if last.ckpt exists to prevent accidental restarts
    if [ -f "$LAST_CKPT" ]; then
        echo "💡 Found $LAST_CKPT. Automatically enabling resume."
        CKPT_ARG="++ckpt_path=$LAST_CKPT"
    fi
fi

# 5. Launch Training via torchrun
# Global Batch = B * N * A = 128 * 4 * 3 = 1536 (Matches baseline 256*6*1)
echo "🚀 Launching 4-GPU DDP Training (4,5,6,7) | Global Batch: 1536"

# Default checkpoint path for FINETUNE_FROM (Run 9 Baseline)
BASELINE_CKPT="/home/yangz/NViT-master/logs/train/runs/2026-01-21_15-28-28/checkpoints/last.ckpt"
FINETUNE_ARG=""
if [ -f "$BASELINE_CKPT" ]; then
    FINETUNE_ARG="++FINETUNE_FROM=$BASELINE_CKPT"
fi

torchrun --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:29505 \
         nvit/train_guided.py experiment=hmr_vit_transformer \
         $CKPT_ARG \
         $FINETUNE_ARG \
         ++DATASETS_CONFIG_FILE=datasets_tar.yaml \
         ++TRAIN.BATCH_SIZE=128 \
         ++TRAIN.ACCUMULATE_GRAD_BATCHES=3 \
         ++TRAIN.GRAD_CLIP_VAL=0.5 \
         ++TRAIN.LR=1e-5 \
         ++GENERAL.NUM_WORKERS=2 \
         ++GENERAL.PREFETCH_FACTOR=1 \
         ++GENERAL.LOG_STEPS=100 \
         ++GENERAL.CHECKPOINT_STEPS=1000 \
         ++GENERAL.CHECKPOINT_SAVE_TOP_K=3 \
         ++trainer.num_sanity_val_steps=0 \
         ++MODEL.BACKBONE.depth=12 \
         ++MODEL.BACKBONE.switch_layer_1=8 \
         ++MODEL.BACKBONE.switch_layer_2=11 \
         ++MODEL.BACKBONE.mamba_variant=spiral \
         ++MODEL.BACKBONE.gcn_variant=guided \
         ++MODEL.SMPL_HEAD.TRANSFORMER_DECODER.depth=3 \
         ++MODEL.SMPL_HEAD.TRANSFORMER_DECODER.heads=4 \
         ++MODEL.SMPL_HEAD.TRANSFORMER_DECODER.mlp_dim=1024 \
         ++FREEZE_DEPTH=8 \
         ++trainer.max_epochs=150 \
         ++trainer.devices=4 \
         ++trainer.precision=bf16-mixed \
         ++trainer.log_every_n_steps=100 \
         ++paths.output_dir=$OUTPUT_DIR

echo "✨ Training execution ended."
