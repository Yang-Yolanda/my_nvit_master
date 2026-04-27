# External baselines audit (CH5)

This note accompanies `artifacts/external_baselines/SMPLer/results.csv` and the SMPLer integration under `nvit/external_baselines/smpler_adapter.py`.

## 1. SMPLer (reproducible in this stack)

| Item | Detail |
|------|--------|
| **Repository** | `https://github.com/xuxy09/SMPLer` |
| **Local clone** | `/home/yangz/external_baselines/SMPLer` (created here via GitHub `codelink` zip when `git clone` was blocked; re-clone with git if preferred) |
| **Conda / deps** | README: `conda create -n smpler python=3.8`, then PyTorch 1.8+cu111, packages listed in README, PyTorch3D 0.5.0 (see README). CH5 eval only needs inference; PyTorch3D is **not** required for `SMPLerCH5Wrapper` forward (demo uses it only for optional mesh vis). |
| **meta_data (SMPL + regressors)** | Weiyun: `https://share.weiyun.com/8zWRIAN1` → extract to `$SMPLER_ROOT/meta_data/` (`basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`, `J_regressor_h36m_correct.npy`, `J_regressor_3dpw.npy`, `smpl_mean_params.npz`, … per `config.py`). |
| **Pretrained weights** | Weiyun: `https://share.weiyun.com/o65G2zHE` → extract to `$SMPLER_ROOT/pretrained/` (includes HRNet backbones under `pretrained/hrnet/` and SMPLer checkpoints). README evaluation commands reference: `pretrained/SMPLer_h36m.pt`, `pretrained/SMPLer_3dpw.pt`, and SMPLer-L variants (`SMPLer-L_h36m.pt`, `SMPLer-L_3dpw.pt`). |
| **Grep for ckpt names** | `rg 'pretrained/.*\\.pt' README.md` in the SMPLer repo lists all shipped evaluation checkpoints. |

### Adapter: `SMPLerCH5Wrapper`

- **Reuses** the CH5 / HMR2 pipeline: `hmr2.datasets.create_dataset(model.cfg, dataset_cfg, …)` exactly like `nvit/skills/evaluate_model/standard_eval.py` (same crops and NPZs).
- **Input**: `batch['img']` from that dataloader (ImageNet-normalized tensor). Images are resized **224×224** inside the wrapper to match SMPLer’s HRNet training resolution.
- **Output** (aligned with `hmr2.utils.pose_utils.Evaluator`): `pred_keypoints_3d`, `pred_keypoints_2d`, plus `pred_cam`, `pred_smpl_params`, `pred_vertices` for debugging. SMPL decode uses the **same** `hmr2.models.SMPL` instance as a reference NViT/HMR2 checkpoint so joint indices match `KEYPOINT_LIST` in `datasets_eval.yaml`.
- **Reference checkpoint**: `--ckpt` / `HMR2_CFG_REFERENCE_CKPT` must be any NViT/HMR2 Lightning ckpt loadable by `nvit.utils.model_io.load_model_from_ckpt` (provides `cfg` + `smpl` only; SMPLer weights come from `--smpler_ckpt`).

### How to run

```bash
export PYTHONPATH=/path/to/NViT-master:/path/to/4D-Humans:$PYTHONPATH
export SMPLER_ROOT=/home/yangz/external_baselines/SMPLer
export HMR2_CFG_REFERENCE_CKPT=/path/to/your_nvit_or_hmr2.ckpt
export HMR2_EVAL_DATA_DIR=/cpfs_infra/shared/yangz/4D-Humans/hmr2_evaluation_data
bash /path/to/NViT-master/run_eval_suite_final.sh
```

Per SMPLer README, use **`SMPLer_3dpw.pt` + `--data_mode 3dpw`** for 3DPW-TEST and **`SMPLer_h36m.pt` + `--data_mode h36m`** for H36M-VAL-P2. Mixing checkpoint and `data_mode` breaks the intended training/eval pairing.

`artifacts/external_baselines/SMPLer/results.csv` is regenerated from JSON logs by `nvit/external_baselines/aggregate_smpler_results.py`. If metrics are empty, check `artifacts/external_baselines/SMPLer/logs/*.log` for missing NPZs, missing `meta_data/`, or missing `pretrained/hrnet/*.pth`.

---

## 2. TransSMPL (★ reported only — not re-run here)

Source: **TransSMPL: Efficient Human Pose Estimation with Pruned and Quantized Transformer Networks**, *Electronics* (MDPI), [DOI landing](https://www.mdpi.com/2079-9292/13/24/4980). Table 1 (paper text and summary) reports:

| Dataset | MPJPE (mm) ★ | PA-MPJPE (mm) ★ |
|---------|----------------|-----------------|
| Human3.6M | 48.5 | 33.2 |
| 3DPW | 77.8 | 47.6 |

**Footnote (non apples-to-apples):** These numbers use the paper’s own training/eval and SMPLer comparison settings (MobileNetV3 backbone, pruning/quantization, etc.), not necessarily identical preprocessing, cropping, or checkpoint pairing as our CH5 `hmr2_evaluation_data` pipeline. Treat as **literature-reported** benchmarks only.

---

## 3. ICIC 2025 (★ reported placeholder — not re-run here)

A **single canonical “ICIC 2025” SMPL / HMR paper with standard 3DPW + H36M MPJPE/PA-MPJPE** could not be resolved unambiguously from open web search in this environment. If **ICIC** was a typo for **ICCV**, a nearby 2025 reference with public numbers is **PersPose** (ICCV 2025); the official README reports (★):

| Dataset | MPJPE (mm) ★ | PA-MPJPE (mm) ★ |
|---------|----------------|-----------------|
| Human3.6M | 43.0 | 28.3 |
| 3DPW | 60.1 | 39.1 |

**Footnote (non apples-to-apples):** PersPose uses perspective encoding and its own data preparation; metrics are **not** directly comparable to our CH5 evaluator without re-running their released code on our NPZ protocol. Replace this block with your exact **ICIC 2025** paper title, table, and page if you had a specific venue/paper in mind.

---

## 4. SMPLer paper table (reference, ★ reported)

From the SMPLer README / TPAMI-linked materials, **SMPLer** (HRNet-W32) reports roughly: H36M MPJPE **47.0** / PA-MPJPE **32.8**; 3DPW MPJPE **75.7** / PA-MPJPE **45.2**. **SMPLer-L** reports H36M **45.2** / **32.4**; 3DPW **73.7** / **43.4**. Use as cross-check only; our `results.csv` row reflects **our** CH5 re-evaluation when `run_eval_suite_final.sh` completes successfully.
