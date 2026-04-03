# 🧪 Paper 2 Experimental Protocol: Kinetic-Guided Hybrid HMR

## 1. 核心叙事逻辑 (Incremental Storyline)
本论文的实验设计遵循层层递进的 **3阶段逻辑** (Experiment Flow):

### Step 1: 确定最佳架构 (Partitioning)
**Goal**: 证明 "基于指标的自适应划分" (Adaptive Partitioning) 优于 "人工硬划分" (Hard Partitioning)。
- **Setup**: 使用 Experiment 3 确定的 KTI 阈值 (Start/Rigid) 来分配模块。
- **Comparison**: 
    - `Fixed (Mid-Heavy)` vs `Adaptive (KTI-Guided)`
- **Winner**: **Adaptive Partitioning** (作为后续实验的固定蓝图 Blueprint).

### Step 2: 确定最佳模块协同 (Module Synergy)
**Goal**: 在确定的蓝图内，证明 "Mamba + Guided GCN" (Hybrid) 优于单一模块。
- **Setup**: 固定 Adaptive 架构，仅替换模块实现。
- **Comparison**:
    - `Mamba Only` (Lack Structure)
    - `Mamba + Skeleton GCN` (Rigid Structure)
    - `Mamba + Guided GCN` (Hybrid Guidance - **Ours**)
- **Winner**: **Mamba + Guided GCN** (Hybrid).

### Step 3: 确定最佳组件设计 (Component Refinement)
**Goal**: 深入微调 Mamba 和 GCN 的内部机制。
- **Sub-step 3.1: Scan Strategy**:
    - `Spiral Scan` (Center-Out, Image-based)
    - `Kinetic Scan` (Anatomical, Structure-based) -> **Winner**
- **Sub-step 3.2: Directionality**:
    - `Bidirectional` (双向)
    - `Forward Only` (正向 - Root to Leaf) -> **User Hypothesis: Best for Kinematics**
- **Sub-step 3.3: GCN Refinement**:
    - `Classic Adj` (Binary)
    - `Anatomical Adj` (Highlighted Structure) -> **To be designed**

## 2. 实验矩阵 (Run 10 Series) - Revised

| Run ID | Step | Config Description | Hypothesis Check |
| :--- | :--- | :--- | :--- |
| **Run 10.1** | **step 2** | Kinetic-Bi + Grid GCN | **Baseline** (Running) |
| **Run 10.2** | **step 2** | Kinetic-Bi + Skeleton GCN | **Structure Control** (Running) |
| **Run 10.3** | **step 2** | **Kinetic-Bi + Guided GCN** | **Hybrid Synergy (Ours)** |
| **Run 10.4** | **step 3.1** | Spiral + Guided GCN | **Scan Ablation** (Prove Kinetic > Spiral) |
| **Run 10.5** | **step 3.2** | **Kinetic-Fwd + Guided GCN** | **Direction Ablation** (Prove Forward > Bi) |
| **Run 10.6** | **step 3.3** | Kinetic-Fwd + **Enhanced GCN** | **Final Best Model** |

## 3. 下一步行动 (Action Items)
1.  **Launch Run 10.3**: 验证混合架构的优越性 (Step 2 Winner)。
2.  **Launch Run 10.5**: 验证正向计算 (Forward) 的优越性 (Step 3 Winner)。
3.  **Design Enhanced GCN**: 研究如何进一步凸显人体结构 (Step 3.3)。
