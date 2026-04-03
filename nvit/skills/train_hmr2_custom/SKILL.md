---
name: Train HMR2 with Custom Datasets
description: Training orchestration for HMR2 using custom training data at /home/yangz/NViT-master/hmr2_training_data and validation at /home/yangz/4D-Humans/hmr2_evaluation_data
---

# HMR2 Custom Training Skill

This skill automates the training of HMR2 models using the custom datasets located in the specified directories. It ensures proper data loading from the `datasets_tar.yaml` configuration and validates the trained model on standard benchmarks.

## Standardized Research Workflow

Every experiment conducted via this skill follows the **Standard 6-Step Process**:

1.  **Step 1: Data Verification** - Automated verification of training and evaluation dataset availability (reusing `verify_datasets.py`).
2.  **Step 2: Dataset Config** - Confirmation of `datasets_tar.yaml` settings.
3.  **Step 3: Code Config** - Environment and output directory setup.
4.  **Step 4: Training** - Execution of the finetuning loop using `finetune_dense.py`.
5.  **Step 5: Testing** - Standard evaluation on validation datasets using `standard_eval.py`.
6.  **Step 6: Reporting** - Generation of a consolidated `training_summary.txt`.

## Dataset Configuration

### Training Data
- **Location**: `/home/yangz/NViT-master/hmr2_training_data`
- **Configuration**: `/home/yangz/4D-Humans/hmr2/configs/datasets_tar.yaml`
- **Datasets Included**:
  - MPI-INF-TRAIN-PRUNED
  - H36M-TRAIN-WMASK
  - MPII-TRAIN-WMASK
  - COCO-TRAIN-2014-WMASK-PRUNED
  - AVA-TRAIN-MIDFRAMES-1FPS-WMASK
  - AIC-TRAIN-WMASK
  - INSTA-TRAIN-WMASK
  - COCO-TRAIN-2014-VITPOSE-REPLICATE-PRUNED12

### Evaluation Data
- **Location**: `/home/yangz/4D-Humans/hmr2_evaluation_data`
- **Datasets**:
  - 3DPW-TEST
  - H36M-VAL-P2
  - COCO-VAL
  - POSETRACK-2018-VAL
  - HR-LSPET

## Usage

### 🚀 Standard Training

```bash
# Run full training pipeline
python nvit/skills/train_hmr2_custom/orchestrate.py --epochs 10 --gpu 0

# Dry run to verify commands
python nvit/skills/train_hmr2_custom/orchestrate.py --dry_run
```

### ⚙️ Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--gpu` | str | `0` | Target GPU ID. |
| `--epochs` | int | 10 | Number of training epochs. |
| `--batch-size` | int | 16 | Batch size per GPU. |
| `--lr` | float | 1e-5 | Learning rate. |
| `--output_dir` | str | `output/hmr2_custom_training` | Output directory for checkpoints and logs. |
| `--dry_run` | flag | False | Print commands without executing. |

## Key Features

- **Automatic Dataset Discovery**: Reads `datasets_tar.yaml` to locate all training tar files
- **Mixed Dataset Training**: Combines 3DPW with multiple WebDataset sources
- **Automatic Evaluation**: Runs validation immediately after training
- **Checkpoint Management**: Saves both best and latest models

## Dependencies

- `nvit/finetune_dense.py`: Core training engine
- `nvit/datasets_mixed.py`: Mixed dataset loader (now config-aware)
- `nvit/skills/evaluate_model/standard_eval.py`: Automated evaluation
- `nvit/skills/manage_experiment/skill_base.py`: Pipeline infrastructure

## Output Structure

```
output/hmr2_custom_training/
├── best_ft_model.pth          # Best checkpoint by validation MPJPE
├── latest_ft_model.pth        # Latest checkpoint
└── eval_results.json          # Evaluation metrics
```

# Architecture & Convergence (Failure-to-Success)

Lessons learned from Paper 2 ablation regressions (High MPJPE > 200mm):

- **NEVER Train from Scratch**: Initializing the backbone from scratch for 3D pose is extremely inefficient. **Rule**: Always use `FINETUNE_FROM` to load the 150-epoch baseline weights.
- **Heatmap Damping**: Avoid heavy logit damping (e.g., `* 0.1`) in the `HeatmapMapper`. It causes a signal blackout for soft-argmax.
- **Camera Bias**: Initialize scale `s` to `1.0` (bias index 0) in the `GuidedSMPLHead` to ensure valid projections.

# Troubleshooting & Monitoring

## 1. Silent Failures (DDP/Multiprocessing)
If logs are empty or processes disappear:
- **Check for Zombies**: `ps aux | grep train_guided`
- **Kill Lingering Processes**: `pkill -f train_guided`
- **Verify NCCL**: Ensure `MASTER_ADDR` and `MASTER_PORT` are set correctly if not using SLURM.

## 2. GPU Memory Mismatch
If `nvidia-smi` shows memory usage but 0% utilization:
- The process is likely dead or deadlocked (Zombie).
- **Action**: Kill process ID explicitly (`kill -9 <PID>`).

## 3. Debugging Tips
- Run with `--devices 1` (Single GPU) first to isolate code/config errors.
- Check `stderr` (terminal output) as Hydra sometimes prints errors there instead of the log file.
