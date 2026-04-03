# PROJECT_STATE.md - NViT Project Status

## 🎯 Current Phase: Controlled Ablation & Metric Verification
**Status**: ACTIVE
**Objective**: Validate the "Guided Head" architecture through rigorous controlled ablation experiments. No new architectural features or NAS are permitted.

## 🏗️ Fixed Main Version Definition
- **Backbone**: `AdaptiveNViT` (ViT + Mamba)
- **Transition Layer**: `HeatmapMapper` (Spatial Soft-Argmax mapping patches to 24 joint centers)
- **SMPL Head**: `GuidedSMPLHead` (TransformerDecoder with 2D coordinate guidance from heatmaps)
- **Key Indicators**: KPI (Kinematic Purity Index) and KTI (Knowledge Transfer Index) are the primary metrics for structural interpretation.

## 🚫 Restricted Activities
- **NO** open-ended Architecture Search (NAS).
- **NO** new complex modules or branches.
- **NO** creation of redundant scripts (v2, v3, etc.).
- **NO** automatic execution of training/testing without explicit user confirmation.

## 📁 Repository Hygiene
- **Main Entry Point**: `train_guided.py`
- **Core Models**: `nvit2_models/adaptive_nvit.py`, `nvit2_models/guided_head.py`
- **Configuration**: All ablations must be driven by `config` files or command-line overrides, NOT by duplicating scripts.
- **Archive Policy**: Legacy scripts are moved to `archive_legacy/` before deletion.
