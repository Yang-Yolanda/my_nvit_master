---
name: manage_experiment
description: Standardized workflow for managing NViT experiments, including dataset setup via download/symlink and training execution.
---

# Manage Experiment Skill

This skill standardizes the lifecycle of an NViT experiment: **Data Setup -> Train -> Test**.

## 1. 📂 Dataset Setup (Action: `setup_data`)

### Locations
- **Training (Compound Mix)**: Centrally managed on `/mnt/hdd_toshiba_1/yangz_data/4D-Humans/data`
    - **H36M** (tar): `~/4D-Humans/data/finetune_ext/h36m`
    - **MPII** (tar): `~/4D-Humans/data/finetune_ext/mpii`
    - **COCO** (tar): `~/4D-Humans/data/finetune_ext/coco`
    - **MPI-INF** (tar): `~/4D-Humans/data/finetune_ext/mpi_inf`
    - **AIC/AVA/InstaVariety** (TBD): `~/4D-Humans/data/finetune_ext/{aic,ava,insta}`
    - **3DPW** (Map-style): `~/4D-Humans/data/3DPW`
- **Testing**:
    - `3DPW`: Protocol for SOTA verification
    - `H36M-Val-P2`: Centered protocol for pose accuracy

### Unification Strategy (Symlinks)
The project codebase ONLY sees `~/4D-Humans/data`. Physical storage is on HDD:
```bash
# Example Unification Flow
ln -sfn /mnt/hdd_toshiba_1/yangz_data/4D-Humans/data/finetune_ext_real ~/4D-Humans/data/finetune_ext
ln -sfn /mnt/hdd_toshiba_1/yangz_data/h36m/images ~/4D-Humans/data/h36m-val-p2
```

# Verify Tar Shards
ls -l ~/4D-Humans/data/finetune_ext/coco | head -n 5
```

## 2. 🧪 Training Protocols

### "Round 9" Gold Standard (Adaptive NViT)
- **Codebase**: `4D-Humans` / `NViT` (Main project root)
- **Script**: `nvit/train_guided.py` (via Hydra)
- **Architecture**: `GuidedHMR2Module` (Spatial Mamba + GCN)
- **Verified Command**:
```bash
python nvit/train_guided.py \
    experiment=hmr_vit_transformer \
    data=mix_all \
    trainer.devices=1 \
    +trainer.max_epochs=150 \
    +trainer.check_val_every_n_epoch=5 \
    trainer.precision=bf16-mixed \
    TRAIN.LR=1e-5 \
    TRAIN.BATCH_SIZE=32 \
    +TRAIN.GRAD_CLIP_VAL=0.5 \
    GENERAL.NUM_WORKERS=8 \
    GENERAL.VAL_STEPS=400 \
    MODEL.SMPL_HEAD.TRANSFORMER_DECODER.depth=3 \
    MODEL.SMPL_HEAD.TRANSFORMER_DECODER.heads=4 \
    +LOSS_WEIGHTS.HEATMAP=2.0 \
    +GENERAL.task_name=run9_pose_balance
```
*Note: Run 9 used a partial dataset (missing H36M). Run 10 aims to use the full Mix.*

## 3. 📊 Evaluation
- **Standard**: `nvit/skills/evaluate_model/standard_eval.py`
- **Video (PHALP)**: `nvit/skills/evaluate_video/standard_eval_video.py`
