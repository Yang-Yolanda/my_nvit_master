# ARCHITECTURE_CURRENT.md - NViT "Guided Head" Version

## 🧩 Structural Breakdown

### 1. Vision Perception Stage (Backbone Layers 0-7)
- **Module**: `ViTBlock`
- **Input**: 256x192 Image -> 16x16 Patches (196 total + 1 CLS).
- **Purpose**: Global feature extraction and visual understanding.

### 2. Topology/Kinematic Stage (Backbone Layers 8-11)
- **Module**: `MambaBlock` (Spiral or Bidirectional)
- **Purpose**: Global coordination and structural dependency modeling.
- **Note**: The original GCN has been removed in this version.

### 3. Structural Transition (Mapping)
- **Module**: [HeatmapMapper](file:///home/yangz/NViT-master/nvit/Code_Paper2_Implementation/nvit2_models/nvit_hybrid.py#L101-L197)
- **Logic**: 
  - Predicts 24 heatmaps (logits) from patch features.
  - Applies Soft-Argmax to derive 2D joint coordinates.
  - Samples patch features at predicted locations to initialize joint queries.

### 4. Guided SMPL Head
- **Module**: [GuidedSMPLHead](file:///home/yangz/NViT-master/nvit/Code_Paper2_Implementation/nvit2_models/guided_head.py#L87-L210)
- **Mechanism**:
  - **Dynamic Queries**: Static joint queries + Positional Encoding of 2D joint coordinates.
  - **Cross-Attention**: Joints (queries) attend to Patches (context) via `TransformerDecoder`.
  - **Regression**: Predictions for 24 rotations (6D), 10 shape params (betas), and 3 camera params.

## 📊 Loss Functions
- Standard SMPL Mesh/Joint losses.
- **Heatmap Consistency Loss**: Auxiliary supervision for the `HeatmapMapper`.
- **KTI/KPI Regularization**: Optional terms to enforce structural priors.
