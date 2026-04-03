# ABLATION_PLAN.md - 6-Group Titration Experiments

All experiments are based on the **Fixed Main Version (Guided Head)**.

| Exp ID | Title | Variable | Configuration Changes |
| :--- | :--- | :--- | :--- |
| **Exp-0** | Baseline (Legacy) | Model Topology | `MODEL.BACKBONE.gcn_variant=skeleton`, `MODEL.BACKBONE.switch_layer_2=10` |
| **Exp-1** | Guided Head (Main) | Model Topology | `MODEL.BACKBONE.gcn_variant=guided`, `MODEL.BACKBONE.switch_layer_2=10` |
| **Exp-2** | Position: Early | Mapping Depth | `MODEL.BACKBONE.switch_layer_2=4` (Mapping after Layer 4) |
| **Exp-3** | Position: Mid-Late | Mapping Depth | `MODEL.BACKBONE.switch_layer_2=8` (Mapping after Layer 8) |
| **Exp-4** | Component - Scan | Mamba Variant | `MODEL.BACKBONE.mamba_variant=spiral` vs `bi` |
| **Exp-5** | Component - Head | Guidance Type | `MODEL.SMPL_HEAD.GUIDANCE=indexing` vs `heatmap` |
| **Exp-6** | Indicator Effect | Loss Weights | `LOSS.KTI_WEIGHT=0.0`, `LOSS.KPI_WEIGHT=0.0` |

## 🚀 Execution Guide
- **Script**: `nvit/train_guided.py`
- **Base Command**: 
  ```bash
  python train_guided.py \
    trainer.devices=[0] \
    exp_name=NVIT_EXP_X \
    MODEL.BACKBONE.gcn_variant=guided \
    MODEL.BACKBONE.switch_layer_2=8 \
    FREEZE_DEPTH=8
  ```
