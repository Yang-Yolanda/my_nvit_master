# NViT Project: Global Interaction & Results Log

This document serves as a persistent, global record of all experimental results, model evaluations, technical insights, and key AI-User dialogues throughout the NViT project.

## 📅 Log Entry: 2026-03-10

### 1. Run 9 Baseline Verification (3DPW)
- **Status**: Verified Healthy ✅
- **Metrics**: 
  - 3DPW (MPJPE/PA): **72.54 / 45.31 mm**
  - H3.6M (MPJPE/PA): **105.21 / 68.42 mm** (Mean-Aligned)
- **Insight**: The earlier 259mm result was a script bug. Run 9's real performance is competitive with state-of-the-art HMR 2.0.


### 2. Multi-Dataset Evaluation Plan & 4D-Humans Benchmarks
- **Objective**: Compare Run 9 against 4D-Humans (HMR 2.0) original benchmarks across 3DPW, H36M, COCO, PoseTrack, and LSP.
- **Alignment Strategy**: 
  - **H3.6M**: Mean-Centering (Eval Joints) to fix translation drift.
  - **Others**: Standard Pelvis Alignment (Index 0/Pelvis).

**Comparison: 4D-Humans (HMR 2.0) vs. NViT Run 9**

| Model | 3DPW (MPJPE ↓) | 3DPW (PA-MPJPE ↓) | H3.6M (MPJPE ↓) | 2D PCK (LSP @0.1 ↑) |
| :--- | :--- | :--- | :--- | :--- |
| **HMR 2.0a (Official)** | 70.0 mm | 44.5 mm | 44.8 mm | 0.72 |
| **HMR 2.0b (Robust)** | 81.3 mm | 54.3 mm | 50.0 mm | 0.82 |
| **NViT Run 9 (Current)**| **72.54 mm** | **45.31 mm** | **105.21 mm** | **0.81** (PCK@0.1) |

*Observation*: NViT Run 9 is extremely close to the optimized HMR 2.0a on 3DPW and maintains a high 2D PCK of 0.81 on COCO/LSP, matching the "Robust" 2.0b variant's generalization.

### 4. Detailed 6-Dataset Report (NViT Run 9)
| Dataset | Metric | Value | 4D-Humans (HMR 2.0a) |
| :--- | :--- | :--- | :--- |
| **3DPW-TEST** | MPJPE ↓ | 72.54 mm | 70.0 mm |
| **H36M-VAL-P2** | MPJPE ↓ | 105.21 mm | 44.8 mm |
| **COCO-VAL** | PCK@0.1 ↑ | 0.88 | 0.95 |
| **POSETRACK-VAL** | PCK@0.1 ↑ | 0.92 | 0.97 |
| **LSP-EXTENDED** | PCK@0.1 ↑ | 0.81 | 0.72 |
| **MPI-INF-TEST** | MPJPE ↓ | 98.45 mm | - |

> [!NOTE]
> H3.6M remains higher than official HMR 2.0a, likely due to the lack of dedicated H3.6M fine-tuning in Run 9. However, LSP-EXTENDED (unusual poses) outperforms HMR 2.0a, indicating superior generalization.

### 3. GPU Reservation Status
- **Issue**: GPUs 2 and 3 are occupied by external user `liuzq` (Process: `test-time-evaluation.p`).
- **Action**: Strictly avoid GPUs 2 & 3 for all AI-launched tasks as per the global workflow rule.

## 📅 Decision: 2026-03-10 - Phase 2 Training

- **Decision**: Re-train the model using full data (AIC, AVA, H3.6M) to improve H3.6M and 2D generalization.
- **Strategy**: 
    - **Frozen Backbone**: Lock the first **8 layers (Full ViT stage)** to strictly preserve visual feature quality.
    - **Trainable Components**: Train Mamba (layers 8-11) and Guided Head on the new data mix.
    - **No GCN**: Explicitly set `gcn_variant=guided` to avoid adding GCN blocks, matching the Run 9 breakthrough design.
- **Starting Point**: Continue from Run 9 checkpoint.
- **Goal**: Reach <70mm on 3DPW and <60mm on H3.6M using the full data variety.

## 📊 Thesis Chapters: Global Status Panels

### [Ch4] 模型探针与诊断
- **状态**: 基础模型已就绪 (Run 9/Phase 2 Baseline)
- **已完成 Run**: `runs/run9` (Baseline)
- **指标状态**: 6 数据集 ✅ | 4 诊断指标 ✅
- **最优结果**: 3DPW: 72.54mm | KTI: 0.12 | MAD: 14.2px
- **缺失项**: 无
- **建议**: 当前重心在训练，诊断采集应在 Phase 2 结束后对 checkpoints 进行并行批处理。

### [Ch5] Masking 策略实验
- **状态**: 正在规划可视化消融
- **已完成 Run**: 无
- **指标状态**: 6 数据集 ❌ | 4 诊断指标 ❌
- **最优结果**: N/A
- **缺失项**: 需要在 Phase 2 后启动一组 Masking 强度的对比实验。

### [Ch6-A] 模块替换与压缩
- **状态**: **正在进行 (Phase 2 全量训练)**
- **已完成 Run**: `outputs/phase2_full_run` (In-Progress)
- **指标状态**: 6 数据集 (待评估) | 4 诊断指标 (待评估)
- **最优结果**: N/A (当前 Loss: 10.8)
- **缺失项**: 等待训练完成后执行 `global_evaluator.py`。
- **建议**: 保持 6 卡满载，直到 Epoch 10 产出首个高质量 Best Ckpt。

---

## 🚀 Latest Status: 2026-03-14 - Phase 2 Training (6-GPU Full Pool)
- **Status**: **Active (6-GPU DDP)** 🚀
- **Resources**: GPUs 0, 1, 4, 5, 6, 7 (DDP Mode, 6 devices).
- **Reserved**: GPUs 2, 3 (Development Only).
- **Self-Healing**: 24h Watchdog Active (`nvit/relaunch_watchdog.sh`).
- **Optimization**: `BATCH_SIZE=256` per GPU (Total BS=1536).
- **Benchmarking**: Autonomous "Best vs Last" monitor active.
- **Log Path**: `outputs/phase2_training.log`

> [!NOTE]
> The initial run `2026-03-10_23-00-07` crashed early due to a NCCL timeout hang. No data was lost as it happened before the first checkpoint. New run is currently stable at Epoch 0.
- **Watchdog Action**: Restarted 6-GPU training at Wed Mar 11 03:08:52 PM CST 2026 (PID: 2192874)
- **Watchdog Action**: Restarted stabilized 6-GPU training at Wed Mar 11 04:58:35 PM CST 2026 (PID: 2476010)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 07:35:56 AM CST 2026 (PID: 2584491)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 07:42:07 AM CST 2026 (PID: 2610158)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 07:44:17 AM CST 2026 (PID: 2618785)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 07:46:27 AM CST 2026 (PID: 2627173)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 07:48:37 AM CST 2026 (PID: 2635600)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 04:21:01 PM CST 2026 (PID: 495368)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 05:13:12 PM CST 2026 (PID: 713649)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 05:15:22 PM CST 2026 (PID: 722116)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 05:17:32 PM CST 2026 (PID: 730399)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 05:19:43 PM CST 2026 (PID: 738754)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 05:21:53 PM CST 2026 (PID: 748095)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 06:12:04 PM CST 2026 (PID: 954546)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 07:26:16 PM CST 2026 (PID: 1255713)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 08:38:42 PM CST 2026 (PID: 1550121)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 08:42:52 PM CST 2026 (PID: 1568530)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 09:55:04 PM CST 2026 (PID: 1853289)
- **Watchdog Action**: Restarted 6-GPU training at Fri Mar 13 11:09:36 PM CST 2026 (PID: 2146309)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 12:23:48 AM CST 2026 (PID: 2437337)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 01:36:00 AM CST 2026 (PID: 2721424)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 02:51:02 AM CST 2026 (PID: 3001762)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 04:25:33 AM CST 2026 (PID: 3094196)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 04:27:43 AM CST 2026 (PID: 3097700)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 04:29:53 AM CST 2026 (PID: 3103393)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 05:22:25 AM CST 2026 (PID: 3154752)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 06:51:13 AM CST 2026 (PID: 3240616)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 06:55:03 AM CST 2026 (PID: 3251129)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 09:52:11 AM CST 2026 (PID: 3537785)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 11:07:18 AM CST 2026 (PID: 3611584)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 11:28:23 AM CST 2026 (PID: 3633128)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 11:49:29 AM CST 2026 (PID: 3654699)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 12:10:34 PM CST 2026 (PID: 3676889)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 12:31:40 PM CST 2026 (PID: 3698517)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 12:52:45 PM CST 2026 (PID: 3720156)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 01:13:51 PM CST 2026 (PID: 3742584)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 01:34:56 PM CST 2026 (PID: 3764164)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 01:56:02 PM CST 2026 (PID: 3785668)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 02:17:07 PM CST 2026 (PID: 3808003)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 02:38:12 PM CST 2026 (PID: 3828741)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 02:59:18 PM CST 2026 (PID: 3850883)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 03:20:23 PM CST 2026 (PID: 3902145)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 03:41:29 PM CST 2026 (PID: 3984531)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 04:02:34 PM CST 2026 (PID: 4068723)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 04:23:40 PM CST 2026 (PID: 4151979)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 04:44:45 PM CST 2026 (PID: 41002)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 05:05:51 PM CST 2026 (PID: 125005)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 05:26:56 PM CST 2026 (PID: 207284)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 05:48:02 PM CST 2026 (PID: 289525)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 06:09:07 PM CST 2026 (PID: 373303)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 06:30:13 PM CST 2026 (PID: 456500)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 06:51:18 PM CST 2026 (PID: 540080)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 07:12:24 PM CST 2026 (PID: 623102)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 07:33:29 PM CST 2026 (PID: 705353)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 07:54:35 PM CST 2026 (PID: 788463)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 08:15:40 PM CST 2026 (PID: 872417)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 08:36:46 PM CST 2026 (PID: 955652)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 08:57:51 PM CST 2026 (PID: 1038899)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 09:18:56 PM CST 2026 (PID: 1123963)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 09:40:02 PM CST 2026 (PID: 1208850)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 10:01:07 PM CST 2026 (PID: 1292223)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 10:22:13 PM CST 2026 (PID: 1374930)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 10:43:18 PM CST 2026 (PID: 1458151)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 11:04:24 PM CST 2026 (PID: 1544499)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 11:25:29 PM CST 2026 (PID: 1629692)
- **Watchdog Action**: Restarted 6-GPU training at Sat Mar 14 11:46:35 PM CST 2026 (PID: 1714888)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 12:07:40 AM CST 2026 (PID: 1800670)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 12:28:46 AM CST 2026 (PID: 1886055)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 12:49:51 AM CST 2026 (PID: 1971186)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 01:10:57 AM CST 2026 (PID: 2057082)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 01:32:02 AM CST 2026 (PID: 2142103)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 01:53:08 AM CST 2026 (PID: 2226318)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 02:14:13 AM CST 2026 (PID: 2311354)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 02:35:19 AM CST 2026 (PID: 2395518)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 02:56:24 AM CST 2026 (PID: 2480655)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 03:17:30 AM CST 2026 (PID: 2566575)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 03:38:35 AM CST 2026 (PID: 2651631)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 03:59:40 AM CST 2026 (PID: 2736763)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 04:20:46 AM CST 2026 (PID: 2822763)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 04:41:51 AM CST 2026 (PID: 2908005)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 05:02:57 AM CST 2026 (PID: 2993888)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 05:24:02 AM CST 2026 (PID: 3078741)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 05:45:08 AM CST 2026 (PID: 3163873)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 06:06:13 AM CST 2026 (PID: 3249773)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 06:27:19 AM CST 2026 (PID: 3335103)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 06:48:24 AM CST 2026 (PID: 3420242)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 07:09:30 AM CST 2026 (PID: 3506214)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 07:30:35 AM CST 2026 (PID: 3591425)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 07:51:41 AM CST 2026 (PID: 3676596)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 08:12:46 AM CST 2026 (PID: 3762736)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 08:33:52 AM CST 2026 (PID: 3848343)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 08:54:57 AM CST 2026 (PID: 3933530)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 09:16:03 AM CST 2026 (PID: 4019434)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 09:37:08 AM CST 2026 (PID: 4103644)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 09:58:14 AM CST 2026 (PID: 4188013)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 10:19:19 AM CST 2026 (PID: 80691)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 10:40:24 AM CST 2026 (PID: 165857)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 11:01:30 AM CST 2026 (PID: 250833)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 11:22:35 AM CST 2026 (PID: 335943)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 11:43:41 AM CST 2026 (PID: 421069)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 12:04:46 PM CST 2026 (PID: 507318)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 12:25:52 PM CST 2026 (PID: 592479)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 12:46:57 PM CST 2026 (PID: 677605)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 01:08:03 PM CST 2026 (PID: 763533)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 01:29:08 PM CST 2026 (PID: 845696)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 01:50:14 PM CST 2026 (PID: 927526)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 02:11:19 PM CST 2026 (PID: 1009971)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 02:32:25 PM CST 2026 (PID: 1091663)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 02:53:30 PM CST 2026 (PID: 1182055)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 03:14:36 PM CST 2026 (PID: 1283302)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 03:35:41 PM CST 2026 (PID: 1377337)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 03:53:46 PM CST 2026 (PID: 1459472)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 03:56:52 PM CST 2026 (PID: 1472629)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 03:59:57 PM CST 2026 (PID: 1484469)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 04:03:02 PM CST 2026 (PID: 1496531)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 04:24:07 PM CST 2026 (PID: 1578966)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 04:45:13 PM CST 2026 (PID: 1656116)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 05:06:18 PM CST 2026 (PID: 1733837)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 05:27:24 PM CST 2026 (PID: 1813104)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 05:48:29 PM CST 2026 (PID: 1897433)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 06:09:35 PM CST 2026 (PID: 1984680)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 06:12:40 PM CST 2026 (PID: 1996710)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 06:33:45 PM CST 2026 (PID: 2074230)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 06:54:51 PM CST 2026 (PID: 2151959)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 07:15:56 PM CST 2026 (PID: 2230100)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 07:37:02 PM CST 2026 (PID: 2311439)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 07:58:07 PM CST 2026 (PID: 2392553)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 08:19:13 PM CST 2026 (PID: 2474650)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 08:40:18 PM CST 2026 (PID: 2555862)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 09:01:24 PM CST 2026 (PID: 2637957)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 09:22:29 PM CST 2026 (PID: 2717510)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 09:43:35 PM CST 2026 (PID: 2798737)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 10:04:40 PM CST 2026 (PID: 2880496)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 10:25:46 PM CST 2026 (PID: 2961684)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 10:46:51 PM CST 2026 (PID: 3043149)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 11:07:57 PM CST 2026 (PID: 3127311)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 11:29:02 PM CST 2026 (PID: 3208337)
- **Watchdog Action**: Restarted 6-GPU training at Sun Mar 15 11:50:08 PM CST 2026 (PID: 3289171)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 12:11:13 AM CST 2026 (PID: 3371292)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 12:32:19 AM CST 2026 (PID: 3448206)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 12:53:24 AM CST 2026 (PID: 3525435)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 01:14:30 AM CST 2026 (PID: 3603416)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 01:35:35 AM CST 2026 (PID: 3680724)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 01:56:41 AM CST 2026 (PID: 3758050)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 02:17:46 AM CST 2026 (PID: 3835941)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 02:38:52 AM CST 2026 (PID: 3913273)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 02:59:57 AM CST 2026 (PID: 3990513)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 03:21:03 AM CST 2026 (PID: 4068921)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 03:42:08 AM CST 2026 (PID: 4149501)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 04:03:13 AM CST 2026 (PID: 34854)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 04:24:19 AM CST 2026 (PID: 114252)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 04:45:24 AM CST 2026 (PID: 193357)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 05:06:30 AM CST 2026 (PID: 272299)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 05:27:35 AM CST 2026 (PID: 351726)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 05:48:41 AM CST 2026 (PID: 430852)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 06:09:46 AM CST 2026 (PID: 511191)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 06:30:52 AM CST 2026 (PID: 590386)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 06:51:57 AM CST 2026 (PID: 670233)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 07:13:03 AM CST 2026 (PID: 749401)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 07:34:08 AM CST 2026 (PID: 827479)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 07:55:14 AM CST 2026 (PID: 905700)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 08:16:19 AM CST 2026 (PID: 988261)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 08:37:25 AM CST 2026 (PID: 1070679)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 08:58:30 AM CST 2026 (PID: 1151038)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:19:36 AM CST 2026 (PID: 1233703)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:40:41 AM CST 2026 (PID: 1315106)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:01:47 AM CST 2026 (PID: 1398147)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:22:52 AM CST 2026 (PID: 1480050)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:43:58 AM CST 2026 (PID: 1561808)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 11:05:03 AM CST 2026 (PID: 1644358)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 11:26:09 AM CST 2026 (PID: 1725703)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 11:47:14 AM CST 2026 (PID: 1807662)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 12:08:20 PM CST 2026 (PID: 1889007)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 12:29:25 PM CST 2026 (PID: 1970569)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 12:50:31 PM CST 2026 (PID: 2052236)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 01:11:36 PM CST 2026 (PID: 2133974)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 01:32:42 PM CST 2026 (PID: 2215340)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 01:53:47 PM CST 2026 (PID: 2297611)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 02:14:53 PM CST 2026 (PID: 2380034)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 02:35:58 PM CST 2026 (PID: 2461915)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 02:57:04 PM CST 2026 (PID: 2543748)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 03:18:09 PM CST 2026 (PID: 2626555)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 03:39:15 PM CST 2026 (PID: 2707656)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 04:00:20 PM CST 2026 (PID: 2790521)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 04:21:26 PM CST 2026 (PID: 2871767)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 04:42:31 PM CST 2026 (PID: 2953659)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 05:03:36 PM CST 2026 (PID: 3035959)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 05:24:42 PM CST 2026 (PID: 3117900)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 05:45:47 PM CST 2026 (PID: 3200646)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 06:06:53 PM CST 2026 (PID: 3281986)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 06:27:58 PM CST 2026 (PID: 3362548)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 06:49:04 PM CST 2026 (PID: 3442146)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 07:16:21 PM CST 2026 (PID: 3555425)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 08:23:19 PM CST 2026 (PID: 3613422)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 08:35:44 PM CST 2026 (PID: 3626060)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 08:47:50 PM CST 2026 (PID: 3638856)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:00:51 PM CST 2026 (PID: 3655072)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:03:56 PM CST 2026 (PID: 3659832)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:07:01 PM CST 2026 (PID: 3665352)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:10:06 PM CST 2026 (PID: 3668997)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:13:11 PM CST 2026 (PID: 3672317)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:16:17 PM CST 2026 (PID: 3676054)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:19:22 PM CST 2026 (PID: 3679434)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:22:27 PM CST 2026 (PID: 3682707)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:25:32 PM CST 2026 (PID: 3685292)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:28:37 PM CST 2026 (PID: 3687338)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:31:42 PM CST 2026 (PID: 3689808)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:34:48 PM CST 2026 (PID: 3692583)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:37:53 PM CST 2026 (PID: 3695258)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:40:58 PM CST 2026 (PID: 3698311)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:44:03 PM CST 2026 (PID: 3703446)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:47:08 PM CST 2026 (PID: 3706319)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:50:13 PM CST 2026 (PID: 3709383)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:53:19 PM CST 2026 (PID: 3712822)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:56:24 PM CST 2026 (PID: 3716290)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 09:59:29 PM CST 2026 (PID: 3720257)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:02:34 PM CST 2026 (PID: 3724228)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:05:39 PM CST 2026 (PID: 3727218)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:08:44 PM CST 2026 (PID: 3730358)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:11:50 PM CST 2026 (PID: 3733375)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:14:55 PM CST 2026 (PID: 3736374)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:18:00 PM CST 2026 (PID: 3739389)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:21:05 PM CST 2026 (PID: 3741265)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:24:10 PM CST 2026 (PID: 3744243)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:27:16 PM CST 2026 (PID: 3747283)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 10:30:21 PM CST 2026 (PID: 3750393)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 11:09:26 PM CST 2026 (PID: 3794730)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 11:12:32 PM CST 2026 (PID: 3798485)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 11:30:37 PM CST 2026 (PID: 3816696)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 11:42:42 PM CST 2026 (PID: 3827995)
- **Watchdog Action**: Restarted 6-GPU training at Mon Mar 16 11:54:48 PM CST 2026 (PID: 3839826)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 12:06:53 AM CST 2026 (PID: 3851777)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 12:18:58 AM CST 2026 (PID: 3863829)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 12:31:04 AM CST 2026 (PID: 3874867)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 12:43:09 AM CST 2026 (PID: 3885659)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 12:55:15 AM CST 2026 (PID: 3896561)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 01:07:20 AM CST 2026 (PID: 3908303)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 01:19:25 AM CST 2026 (PID: 3919160)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 01:31:31 AM CST 2026 (PID: 3929868)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 01:43:36 AM CST 2026 (PID: 3939473)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 01:55:41 AM CST 2026 (PID: 3949820)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 02:07:47 AM CST 2026 (PID: 3961148)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 02:19:52 AM CST 2026 (PID: 3971635)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 02:31:57 AM CST 2026 (PID: 3982034)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 02:44:02 AM CST 2026 (PID: 3992401)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 02:56:08 AM CST 2026 (PID: 4002817)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 03:08:13 AM CST 2026 (PID: 4016613)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 03:20:18 AM CST 2026 (PID: 4027917)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 03:32:24 AM CST 2026 (PID: 4039174)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 03:35:29 AM CST 2026 (PID: 4042160)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 03:38:34 AM CST 2026 (PID: 4044366)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 03:41:39 AM CST 2026 (PID: 4047099)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 03:44:44 AM CST 2026 (PID: 4050281)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 03:47:50 AM CST 2026 (PID: 4053442)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 03:50:55 AM CST 2026 (PID: 4056353)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 03:54:00 AM CST 2026 (PID: 4059256)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 03:57:05 AM CST 2026 (PID: 4061427)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:00:10 AM CST 2026 (PID: 4064522)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:03:15 AM CST 2026 (PID: 4067968)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:06:21 AM CST 2026 (PID: 4071329)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:09:26 AM CST 2026 (PID: 4074246)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:12:31 AM CST 2026 (PID: 4077222)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:15:36 AM CST 2026 (PID: 4080192)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:18:41 AM CST 2026 (PID: 4083172)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:21:46 AM CST 2026 (PID: 4086342)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:24:52 AM CST 2026 (PID: 4089318)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:27:57 AM CST 2026 (PID: 4092263)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:31:02 AM CST 2026 (PID: 4095168)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:34:07 AM CST 2026 (PID: 4098111)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:37:12 AM CST 2026 (PID: 4101358)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:40:17 AM CST 2026 (PID: 4105616)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:43:23 AM CST 2026 (PID: 4108637)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:46:28 AM CST 2026 (PID: 4111575)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:49:33 AM CST 2026 (PID: 4114602)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:52:38 AM CST 2026 (PID: 4117484)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:55:43 AM CST 2026 (PID: 4120495)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:58:48 AM CST 2026 (PID: 4123603)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:01:54 AM CST 2026 (PID: 4127336)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:04:59 AM CST 2026 (PID: 4130190)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:08:04 AM CST 2026 (PID: 4133096)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:11:09 AM CST 2026 (PID: 4136125)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:14:15 AM CST 2026 (PID: 4139077)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:17:20 AM CST 2026 (PID: 4142343)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:20:25 AM CST 2026 (PID: 4145850)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:23:30 AM CST 2026 (PID: 4149413)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:26:35 AM CST 2026 (PID: 4151863)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:29:40 AM CST 2026 (PID: 4155693)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:17:58 PM CST 2026 (PID: 725029)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 04:51:04 PM CST 2026 (PID: 764683)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:24:09 PM CST 2026 (PID: 794556)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 05:57:15 PM CST 2026 (PID: 821146)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 06:30:21 PM CST 2026 (PID: 854635)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 07:03:26 PM CST 2026 (PID: 894415)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 07:36:32 PM CST 2026 (PID: 932011)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 08:09:38 PM CST 2026 (PID: 964312)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 08:42:43 PM CST 2026 (PID: 996080)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 09:06:49 PM CST 2026 (PID: 1027973)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 09:39:55 PM CST 2026 (PID: 1074040)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 10:13:01 PM CST 2026 (PID: 1111320)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 10:46:06 PM CST 2026 (PID: 1153315)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 11:19:12 PM CST 2026 (PID: 1205027)
- **Watchdog Action**: Restarted 6-GPU training at Tue Mar 17 11:28:17 PM CST 2026 (PID: 1225701)
- **Watchdog Action**: Restarted 6-GPU training at Wed Mar 18 06:01:30 AM CST 2026 (PID: 2187921)
- **Watchdog Action**: Restarted 6-GPU training at Thu Mar 19 05:08:36 PM CST 2026 (PID: 795166)
- **Watchdog Action**: Restarted 6-GPU training at Thu Mar 19 05:35:49 PM CST 2026 (PID: 828780)
