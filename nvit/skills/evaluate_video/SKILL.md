---
name: evaluate_video
description: Run temporal evaluation (Tracking + HMR) on video datasets using PHALP.
---

# Evaluate Video (PHALP Tracking)

This skill wraps PHALP to perform temporal tracking and smoothing on video inputs, integrating custom NViT checkpoints.

## Usage

```bash
# Evaluate on a video folder or single video
python nvit/skills/evaluate_video/standard_eval_video.py --input /path/to/video.mp4 --checkpoint /path/to/model.ckpt
```

## Features
- **Temporal Consistency**: Uses PHALP tracker to associate identities across frames.
- **Smoothing**: Applies OneEuro filter or simple average (configurable).
- **Custom Predictor**: Supports injecting `GuidedHMR2` or `AdaptiveNViT` via `CustomRun9Predictor` logic.
- **Dual Mode**:
    - **Visual**: Renders tracking results.
    - **Metric**: (Todo) Computes MOTA/IDF1 if ground truth available.

## Dependencies
- `phalp` source code (in `nvit/external_models` or `PHALP-master`).
- `nvit/Code_Paper2_Implementation/nvit2_models`.
