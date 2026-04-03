---
name: Train Paper 1 Ablation Groups
description: Sequential orchestration of the 14 experimental masking groups (Hard/Soft/Hybrid) for Paper 1 soft-coding validation.
---

# Paper 1 Ablation Training Skill

This skill automates the finetuning of all experimental groups defined in `nvit/masking_utils.py`. It ensures each group is trained sequentially on a single GPU (avoiding DDP overhead) and aggregates the results.

## Standardized Research Workflow

Every experiment conducted via this skill follows the **"Screw" (Systematic) Process**:

1.  **Step 1: Data Loading** - Automated verification of `4D-Humans/data` availability.
2.  **Step 2: Dataset Setting** - Configuration of 3DPW, MPII, or COCO mix parameters.
3.  **Step 3: Code Setting** - Auto-identification of **Paper 1** vs **Paper 2** and injection of corresponding constraints (Masking vs Backbone).
4.  **Step 4: Training** - Execution of the finetuning loop on a single GPU.
5.  **Step 5: Testing** - Immediate evaluation on the test set using `standard_eval.py`.
6.  **Step 6: Result/Report** - Consolidation of metrics into `ablation_summary.csv`.

## Usage

### 🚀 Standard Tiered Titration
```bash
# Run Tier 1 (Depth Sweep) sequentially
python nvit/skills/train_ablation/orchestrate.py --tier 1 --gpu 0

# Run with Parallel Acceleration (Recommended for L40/High-VRAM)
python nvit/skills/train_ablation/orchestrate.py --tier 1 --gpu 0 --parallel 2 --batch-size 8
```

### ⚙️ Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--tier` | int | 1 | Titration stage (1: Depth, 2: Logic, 3: Guided). |
| `--parallel` | int | 1 | Number of concurrent training processes on the GPU. |
| `--gpu` | str | `0` | Targeting GPU ID. |
| `--batch_size`| int | 16 | Batch size per process (Reduce if using --parallel). |

## 🧪 Phase Logic (Titration)
- **Tier 1 (Depth Sweep)**: Locates the optimal partition point (25% to 75% depth) using Hard Masking.
- **Tier 2 (Logic Sweep)**: Compares Soft vs. Hybrid vs. Sandwich encoding at the optimal depth.
- **Tier 3 (Metric-Guided)**: Validates KTI-driven dynamic partitioning.

## Dependencies
- `nvit/finetune_dense.py`: Core training engine (Paper 1).
- `nvit/skills/manage_experiment/skill_base.py`: Pipeline infrastructure.
- `nvit/skills/evaluate_model/standard_eval.py`: Automated testing.
