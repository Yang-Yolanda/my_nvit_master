---
name: Evaluate Model Performance (High Fidelity)
description: Standardized workflow for evaluating NViT/HMR2 models on multiple benchmarks (3DPW, H36M, COCO) using HMR2's internal Evaluator and correction logic.
---

# Evaluate Model Skill (High Fidelity)

This skill provides a **standardized, repeatable pipeline** for model validation. It leverages the official HMR2 `Evaluator` to ensure metrics are comparable to the state-of-the-art.

## Core Script: `standard_eval.py`

Located at: `nvit/skills/evaluate_model/standard_eval.py`

### Features
1.  **Multi-Dataset Evaluation**: Support for `3DPW-TEST`, `H36M-VAL-P2`, `COCO-VAL`, `POSETRACK-VAL`, and `LSP-EXTENDED`.
2.  **Specialized Metrics**: Automatically switches between MPJPE (3D pose) and KPL2 (2D keypoints) based on dataset.
3.  **Dense GT Generation**: Automatically fixes 3DPW Zero-GT issues by regenerating joints from SMPL parameters.
4.  **Root Alignment**: Performs valid MPJPE calculation by aligning root joints.
5.  **Batch Execution**: New `run_all_evals.sh` script to evaluate a checkpoint on all datasets at once.

### Usage

```bash
# Evaluate on all datasets (ALL)
./run_all_evals.sh /path/to/checkpoint.ckpt [gpu_id]

# Manual run on specific datasets
python nvit/skills/evaluate_model/standard_eval.py \
    --ckpt /path/to/checkpoint.ckpt \
    --dataset ALL \
    --batch_size 64
```

### Arguments
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--ckpt` | str | Required | Path to model checkpoint (.ckpt). |
| `--dataset` | str | `3DPW-TEST` | Dataset config name (e.g., `H36M-VAL-P2`). |
| `--diagnostics` | flag | False | Enable KTI/Entropy calculation. |
| `--dense_gt` | flag | True | Force regeneration of dense GT (Critical for 3DPW). |
| `--batch_size` | int | 32 | Evaluation batch size. |
| `--limit_batches` | int | None | Limit number of batches (for debugging). |

### Prerequisites
*   Ensure `PYTHONPATH` includes `nvit/Code_Paper2_Implementation` if using `nvit2_models`.
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/nvit/Code_Paper2_Implementation
    ```
