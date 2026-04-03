# 🧪 HMR2 ViT Layer Ablation: Research Protocol

## 📊 Current Status
- **Phase 1: Redundancy Mapping** - ✅ COMPLETED
- **Phase 2: Stable Pruning & Benchmarking** - ✅ COMPLETED
- **Phase 3: Accuracy Evaluation** - 📅 PENDING

## 🛠️ Methodology: The "Stable MLP" Path
1. **Backbone Isolation**: Target `model.backbone` to avoid tracing errors in the SMPL-Head.
2. **Structural Pruning**: Focus exclusively on `mlp.fc1` layers (Structural L1-Pruning). 
3. **MHA Preservation**: Avoid QKV pruning to maintain head-count alignment for multi-head attention.
4. **Iterative Sparsity**: 70% target sparsity applied in 5 steps.

## 📉 Redundancy Evidence (Hard Data)
- **Water Zone**: Blocks 8-23 (Mid-blocks).
- **Cosine Similarity**: Peak at **0.9936** (Between consecutive blocks in the Mid-zone).
- **Speedup**: **3.3x** (From 21.6 FPS to 72.1 FPS on GPU).

## 📂 Key Artifacts on Server
- Logic: `experiment_layer_ablation.py`
- Bench: `benchmark_pytorch.py`
- Analysis: `analyze_redundancy.py`
- Pruned Models: `output/ablation/hmr2_mid_heavy.pth` (1.6GB)

## 🚀 Next Mission
- Run precision tests on H3.6M dataset.
- Export stable `mid_heavy` model to ONNX.
