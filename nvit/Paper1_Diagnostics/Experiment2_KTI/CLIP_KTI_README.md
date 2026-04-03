# CLIP KTI Evaluation Guide

## Quick Start

```bash
# 1. Install CLIP (if not already installed)
pip install transformers

# 2. Run CLIP KTI evaluation
python nvit/Paper1_Diagnostics/Experiment2_KTI/evaluate_clip_kmi.py \
    --model openai/clip-vit-base-patch32 \
    --image_folder datasets/3dpw/imageFiles \
    --num_batches 10
```

## Available CLIP Models

- `openai/clip-vit-base-patch32` (Default, 7x7 grid)
- `openai/clip-vit-base-patch16` (14x14 grid)
- `openai/clip-vit-large-patch14` (16x16 grid)

## Expected Results

Since CLIP is a **general vision model** (like SigLIP), it should show:
- **Low KTI scores** across all layers
- **Flat KTI curve** (no structural peaks)
- This serves as a **negative control** to validate that KTI specifically detects physical topology understanding

## Output

Results saved to:
```
nvit/Paper1_Diagnostics/Experiment2_KTI/results/CLIP_<model_variant>/layer_metrics_Control.json
```

## Integration with Existing Pipeline

The script follows the exact same "Two-Matrix Similarity" logic:
1. **Feature Matrix**: CLIP attention maps from `vision_model.encoder.layers`
2. **Topology Matrix**: Dummy SMPL skeleton (since CLIP has no topology knowledge)
3. **KTI Calculation**: `lab.calculate_physically_grounded_kmi(attn_map, dummy_kp)`

## Comparison

After running, use `visualize_kmi.py` to compare:
- HMR2 (High KTI - Specialist)
- CLIP/SigLIP (Low KTI - Generalist)
- Robot Scratch (Zero KTI - Overfitting)
- Robot Pretrained (Weak KTI - Weak Prior)
