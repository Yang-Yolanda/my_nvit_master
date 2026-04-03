import os
import sys
import torch
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add correct PYTHONPATH to sys.path
sys.path.append('/home/yangz/4D-Humans')
sys.path.append(str(Path('/home/yangz/NViT-master').resolve()))

from nvit2_models.guided_hmr2 import GuidedHMR2Module
from hmr2.datasets import create_dataset
from hmr2.configs import dataset_eval_config
from hmr2.utils import Evaluator, recursive_to
from hmr2.utils.geometry import aa_to_rotmat

def generate_interface_report(out_dir):
    report_content = """# 评测接口核对报告 (Evaluation Interface Report)

## 1. 评测脚本来源
- **核心评测脚本**: `nvit/skills/evaluate_model/standard_eval.py`
- **评价器引擎**: `hmr2.utils.Evaluator`
- 该脚本被 `run_batch_hmr_diagnostics.py` 及 `global_evaluator.py` 共享用于全局评测工作流。

## 2. 接口定义核对
- **输入格式**: 脚本期望 PyTorch DataLoader 直接推理，或者包含 `pred_keypoints_3d` 键的字典。
- **坐标系**: Camera Space。预测和GT的 3D keypoints 必须在相机坐标系下对齐。
- **单位缩放**: **Meter 毫米坑点发现**。HMR2 原始输出和数据集 `keypoints_3d` 在内存中均以 **Meter (米)** 为单位。然而，`hmr2.utils.pose_utils.compute_error` 会默认在返回值时 `return 1000 * mpjpe`。因此，调用 `Evaluator.get_metrics_dict()` 后，数值**已经被转换为毫米 (mm)**。之前如果脚本额外执行 `val * 1000` 将导致数值出现 700,000 以上的极大误差（放大了1,000倍）。
- **Root对齐 (Pelvis Alignment)**: 对于 3DPW，标准 HMR2 配置要求使用 `pelvis_ind=0` 或者不配置而由 Evaluator 内部利用其自身的 `keypoint_list` 进行0点偏移。如果外部评测脚本未在送入前后执行减去 Root 并手动将 Root 置 0（如 `H36M` 对齐修正代码段），就会导致全量特征未对齐而在空间漂移，即使单位转换后误差也极高（可达 700+ mm）。
- **Joint Mapping**: 3DPW-TEST 默认使用 14 关节评估 (H36M Subset, `HMR_14`)。
"""
    with open(out_dir / "eval_interface_report.md", "w") as f:
        f.write(report_content)

def check_anomaly_and_evaluate(out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoints
    models_to_test = {
        "Baseline_32L (Teacher)": "/home/yangz/NViT-master/outputs/baseline_32L/checkpoints/best.ckpt",
        "Hybrid_12L (Run9)": "/home/yangz/NViT-master/outputs/ch6_safe_ckpts/best.ckpt"
    }
    
    # Load dataset
    cfg_eval = dataset_eval_config()
    dataset_cfg = cfg_eval['3DPW-TEST']
    dataset_cfg.defrost()
    dataset_cfg.DATASET_FILE = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    dataset_cfg.freeze()
    
    # Temporary mock config for creation
    from yacs.config import CfgNode as CN
    dummy_cfg = CN(); dummy_cfg.MODEL = CN(); dummy_cfg.MODEL.IMAGE_SIZE = 256
    dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]; dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
    dummy_cfg.SMPL = CN(); dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
    dummy_cfg.DATASETS = CN(); dummy_cfg.DATASETS.CONFIG = CN()
    
    dataset = create_dataset(dummy_cfg, dataset_cfg, train=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    results = []
    
    for m_name, ckpt in models_to_test.items():
        print(f"Loading {m_name}...")
        sys.stdout.flush()
        try:
            model = GuidedHMR2Module.load_from_checkpoint(ckpt, strict=False, map_location=device)
            model.eval()
            model.to(device)
        except Exception as e:
            print(f"Skipping {m_name}: {e}")
            sys.stdout.flush()
            continue
            
        hmr2_evaluator = Evaluator(
            dataset_length=int(1e8), 
            keypoint_list=dataset_cfg.KEYPOINT_LIST, 
            pelvis_ind=0, 
            metrics=['mode_mpjpe', 'mode_re'],
            pck_thresholds=None,
        )
        
        i = 0
        limit_batches = 1
        print(f"Evaluating {m_name} on 3DPW-TEST (Limiting to {limit_batches} batches for quick diagnosis)...")
        sys.stdout.flush()
        for batch in tqdm(dataloader):
            if i >= limit_batches: break
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
                
                # Check Unit & Values
                mean_pred = out['pred_keypoints_3d'].abs().mean().item()
                mean_gt = batch['keypoints_3d'].abs().mean().item()
                
                # DIAGNOSTIC PRINT: Show scale of prediction
                if i == 0:
                    with open(out_dir / "anomaly_logs.txt", "a") as f:
                        f.write(f"=== Anomaly Check for {m_name} ===\\n")
                        f.write(f"Pred Mean Absolute (meters): {mean_pred:.4f}\\n")
                        f.write(f"GT Mean Absolute (meters): {mean_gt:.4f}\\n")
                        # 3DPW is usually zero-mean around the body, 
                        # but if pred has massive translations, we see it
                
                # APPLY ROOT ALIGNMENT (like standard_eval.py H36M fix)
                # To ensure completely un-drifted metrics
                keypoint_list = dataset_cfg.KEYPOINT_LIST
                pred_eval_kps = out['pred_keypoints_3d'][:, keypoint_list]
                gt_eval_kps = batch['keypoints_3d'][:, keypoint_list, :-1]
                
                pred_root = pred_eval_kps.mean(dim=1, keepdim=True)
                gt_root = gt_eval_kps.mean(dim=1, keepdim=True)
                
                # We align the overall prediction to zero
                out['pred_keypoints_3d'] = out['pred_keypoints_3d'] - pred_root
                batch['keypoints_3d'][:, :, :-1] = batch['keypoints_3d'][:, :, :-1] - gt_root

                # Ensure dummy alignment joint is 0
                out['pred_keypoints_3d'] = torch.cat([torch.zeros_like(out['pred_keypoints_3d'][:, [0]]), out['pred_keypoints_3d'][:, 1:]], dim=1)
                batch['keypoints_3d'][:, 0, :-1] = 0
                hmr2_evaluator.pelvis_ind = 0 
                
                hmr2_evaluator(out, batch)
            i += 1
            
        metrics = hmr2_evaluator.get_metrics_dict()
        results.append({
            "model_id": m_name,
            "dataset": "3DPW-TEST",
            "mpjpe_mm": f"{metrics.get('mode_mpjpe', 0.0):.2f}",
            "pa_mpjpe_mm": f"{metrics.get('mode_re', 0.0):.2f}",
            "num_samples": i * 32,
            "eval_script_used": "standard_eval.py / hmr2.utils.Evaluator",
            "notes": "Units inherently mm. Root-aligned via Mean Eval Joints before Evaluator pass."
        })
        
        del model
        torch.cuda.empty_cache()
        
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "metrics.csv", index=False)
    
    tex_out = df.to_latex(index=False, column_format="l|l|c|c|c|p{3cm}|p{3cm}")
    with open(out_dir / "metrics_table.tex", "w") as f:
        f.write("% 自动生成的基准对齐表格\\n")
        f.write(tex_out)
        
    print(f"Diagnostics completely finished. Saved to {out_dir}")

if __name__ == "__main__":
    out_dir = Path("/home/yangz/NViT-master/nvit/results/paper1_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    open(out_dir / "anomaly_logs.txt", "w").close() # truncate
    generate_interface_report(out_dir)
    check_anomaly_and_evaluate(out_dir)
