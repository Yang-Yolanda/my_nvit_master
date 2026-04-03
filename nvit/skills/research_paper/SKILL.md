---
name: Paper 2 Context Injection (Guided NViT)
description: Injects the core theoretical framework, hypotheses, and architectural details of Paper 2 ("Topologically Guided VLM") into the agent's context.
---

# Paper 2 Context Injection Skill

This skill ensures the agent "understands" the scientific mission, not just the code.

## 0. Core Development Principles

*   **No "Reinventing the Wheel" (不要反复造轮子)**: Prioritize reusing existing modules (`diagnostic_engine.py`, `GuidedHMR2Module`, etc.) over writing new standalone logic.
*   **Process Optimization (优化流程)**: Focus on improving the efficiency, stability, and diagnostic depth of existing code paths rather than creating redundant variants.
*   **Modular Extensions**: If new functionality is needed, implement it as a modular extension to the existing codebase to maintain consistency across the project.

## 1. Core Hypothesis: "The Spatial Gap"
Current VLMs (like SigLIP/CLIP) are trained on massive image-text pairs but lack explicit **3D spatial reasoning**.
*   **Hypothesis**: By injecting explicit **2D Keypoint Heatmaps** (from a lightweight detector) into a Transformer's query structure, we can bridge the gap between "Pixel Semantics" and "3D Topology".
*   **Goal**: Create a "Guided" architecture where a cheap signal (Heatmaps) guides a heavy feature extractor (ViT/Mamba) to focus on joints.

## 2. Architectural Design (The "Why")
### 2.1 AdaptiveNViT (`nvit_hybrid.py`)
*   **ViT Stage**: "Eyes". Sees the global image context.
*   **Mamba Stage**: "Nerves". Uses **Spiral Scan** to model bio-dynamic dependencies (energy flows from torso to limbs).
*   **HeatmapMapper**: "Translation". Converts visual patches into **24 Joint Tokens**. This is the critical "Guided" step.
*   **GCN Stage**: "Skeleton". Refines the 24 tokens using SMPL topology constraints.

### 2.2 GuidedSMPLHead (`guided_head.py`)
*   **Innovation**: Instead of standard queries, we use **Spatial Fourier Queries**.
*   **Mechanism**: The HeatmapMapper tells the Head *where* to look (Coords), and the Head uses Fourier Features to sample those locations endlessly.

## 3. Experimental Plan (The "What")
*   **Run 7 (Failed)**: FP16 Overflow. Taught us that "Soft-Argmax" needs numerical clamping.
*   **Run 8 (Failed)**: Pose Collapse (predicted mean pose). Taught us that Heatmap Loss (10.0) was too strong.
*   **Run 9 (CURRENT)**: 
    *   **Goal**: Single-GPU 0 stability. 
    *   **Fix**: Heatmap Loss (2.0) + SMPL Loss (2.0). 
    *   **Success Metric**: Non-zero variance in pose predictions.
*   **Ablation Studies**:
    *   Mamba Spiral vs. Sequential Scan.
    *   GCN Adjacencies (Topology vs. Random).

## 4. How to Use This Context
When analyzing a bug or result:
1.  **Ask**: "Does this align with the Spiral Scan logic?" (e.g., if limbs fail but torso works, Mamba might be forgetting long sequences).
2.  **Ask**: "Is the Heatmap guiding or confusing?" (e.g., if loss diverges, maybe the HeatmapMapper is pointing to empty space).
