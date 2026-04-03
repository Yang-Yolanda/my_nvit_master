# 🗺️ NViT Project Map (AI Memory & Topology)

> [!IMPORTANT]
> This document is the "Global Memory" for the AI. Before starting any task, read this to avoid redundant work and leverage existing findings.

## 1. 🏗️ Project Architecture (Topology)
- **Root**: `/home/yangz/NViT-master`
- **Source Code**: `/home/yangz/NViT-master/nvit`
- **External Dependencies**: `/home/yangz/4D-Humans` (Base HMR2 implementation)
- **Datasets (HDD)**: `/home/yangz/4D-Humans/data/` (Linked via `data@` and `data_loader.py`)
- **Key Models Folder**: `nvit/Code_Paper2_Implementation/nvit2_models` (Symlinked as `nvit2_models@`)

## 2. 🏆 Proven Milestones (Do Not Repeat)
- **Efficiency**: MLP Structural Pruning is **already proven** to deliver **3.3x speedup** (21 FPS -> 72 FPS).
- **Redundancy Theory**: KTI Peaks at Layer 0 (Parts) and Layer 7 (Assembly) are **validated**.
- **Adaptive Switching**: Hybridizing ViT (early), Mamba (mid), and GCN (late) is the **active winner** of Phase 2.
- **Run 9 Status**: `guided_hmr2.py` is the current state-of-the-art implementation for Paper 2.

## 3. 📂 Logic Registry (Where is what?)
| Component | Primary File / Folder | Purpose |
| :--- | :--- | :--- |
| **Model** | `nvit/Code_Paper2_Implementation/nvit2_models/guided_hmr2.py` | Adaptive NViT + Guided Head |
| **Data (3DPW)** | `nvit/datasets_3dpw.py` (referenced by `datasets_mixed.py`) | Core test-set loader |
| **Data (Tars)** | `nvit/datasets_mixed.py` | Large-scale training data manager |
| **Benchmarks** | `nvit/Code_Paper2_Implementation/Expt3_Comparative_Study.py` | Comparative logic |
| **Pruning** | `nvit/pruning_core/` | Structural pruning engine |

## 4. 🔕 Redundancy Registry (Ignore These)
- `archive/Paper1_Diagnostics/`: Legacy research.
- `logs/run10_startup_v*.log`: Transient/Fail logs.
- `weights_fragments/`: Debug intermediates.

---
*Created on 2026-01-29. This map is the AI's starting point for all navigation.*
