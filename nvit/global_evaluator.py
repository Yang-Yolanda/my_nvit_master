import os
from nvit.utils.path_utils import get_humans_root, get_project_root, resolve_data_path

import sys
import argparse
import torch
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / 'nvit'))
sys.path.append(str(get_humans_root())) # 4D-Humans root

from nvit.skills.evaluate_model.standard_eval import EvaluatorSkill
from nvit.skills.evaluate_model.scientific_diagnostics import ViTDiagnosticLab, HMR2Wrapper
from nvit2_models.guided_hmr2 import GuidedHMR2Module
from hmr2.datasets import create_dataset
from hmr2.configs import dataset_eval_config
from hmr2.utils import Evaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GlobalEvaluator")

def get_checkpoint(run_path):
    """Find best.ckpt if available, else last.ckpt. Also supports direct .ckpt file path."""
    path = Path(run_path)
    if path.is_file() and path.suffix == '.ckpt':
        return str(path), "direct_file"

    ckpt_dir = path / "checkpoints"
    # Fallback for some structures where checkpoints is not a sub-folder
    if not ckpt_dir.exists() and path.is_dir():
         # Check if the folder itself contains .ckpt files
         pattern = list(path.glob("*.ckpt"))
         if pattern:
             latest = max(pattern, key=os.path.getmtime)
             return str(latest), "folder_direct"
         return None, None

    best_ckpt = ckpt_dir / "best.ckpt"
    last_ckpt = ckpt_dir / "last.ckpt"
    
    if best_ckpt.exists():
        return str(best_ckpt), "best"
    elif last_ckpt.exists():
        return str(last_ckpt), "last"
    else:
        # Check for any epoch=*-step=*.ckpt
        pattern = list(ckpt_dir.glob("epoch=*-step=*.ckpt"))
        if pattern:
            latest = max(pattern, key=os.path.getmtime)
            return str(latest), "latest_step"
    return None, None


def run_human_suite(ckpt_path, output_dir, gpu="0", datasets="ALL", parent_args=None):
    """Runs the 6 standard datasets using EvaluatorSkill"""
    logger.info(f"🚀 Starting Human Metric Suite on GPU {gpu}...")
    skill = EvaluatorSkill(gpu=gpu)
    
    # Setup dummy args for skill
    class Args:
        pass
    args = Args()
    args.ckpt = ckpt_path
    args.dataset = datasets
    args.batch_size = 32
    args.num_workers = 8
    args.limit_batches = getattr(parent_args, 'limit_batches', None)
    args.skip_errors = True
    args.use_mean_alignment = True # Critical for H3.6M
    args.data_dir = str(get_humans_root() / "hmr2_evaluation_data")
    args.output = str(Path(output_dir) / "metrics_suite.json")
    
    skill.run_eval(args)
    return args.output

# def run_diagnostics(ckpt_path, output_dir, gpu="0", num_batches=10, chapter="Ch4"):
#     logger.info(f"🔬 Starting 4-Metric Scientific Diagnostics for Chapter [{chapter}]...")
    
#     device = torch.device(f'cuda:{gpu}')
    
#     # Intelligently detect architecture from checkpoint instead of hardcoding by Chapter!
#     try:
#         checkpoint = torch.load(ckpt_path, map_location='cpu')
#         state_dict = checkpoint.get('state_dict', checkpoint)
#         decpose_weight = state_dict.get('smpl_head.decpose.weight')
#         if decpose_weight is not None and decpose_weight.shape[0] == 144:
#             from hmr2.models.hmr2 import HMR2
#             model = HMR2.load_from_checkpoint(ckpt_path, strict=False, map_location=device)
#             logger.info("Detected Legacy [144] SMPLHead in checkpoint. Loaded standard HMR2.")
#         else:
#             from nvit2_models.guided_hmr2 import GuidedHMR2Module
#             model = GuidedHMR2Module.load_from_checkpoint(ckpt_path, strict=False, map_location=device)
#             logger.info("Detected Guided [6] SMPLHead in checkpoint. Loaded GuidedHMR2Module.")
#     except Exception as e:
#         logger.warning(f"Inspection failed, defaulting to GuidedHMR2Module: {e}")
#         from nvit2_models.guided_hmr2 import GuidedHMR2Module
#         model = GuidedHMR2Module.load_from_checkpoint(ckpt_path, strict=False, map_location=device)
        
#     model.to(device)
#     model.eval()
    
#     wrapper = HMR2Wrapper(model)
#     # output_root here will be used to create [model_name] subfolder
#     lab = ViTDiagnosticLab(wrapper, model_name="diagnostics", output_root=output_dir)
    
#     # Only keep 'Control' group for diagnostics
#     lab.groups = {'Control': {'mask_layers': [], 'mode': 'none'}}
    
#     # 在 dataloader = ... 之前插入
#     print("-" * 30)
#     print(f"Dataset: {dataset_cfg.DATASET_FILE}")
#     print(f"Keypoint List Length: {len(dataset_cfg.KEYPOINT_LIST)}")
#     print(f"Keypoint List: {dataset_cfg.KEYPOINT_LIST}")

#     # 查找 'Pelvis' 或 'Hip' 在列表中的位置
#     try:
#         actual_pelvis_idx = dataset_cfg.KEYPOINT_LIST.index('Pelvis')
#         print(f"Detected Pelvis Index: {actual_pelvis_idx}")
#     except ValueError:
#         print("Warning: 'Pelvis' not found in Keypoint List!")
#     print("-" * 30)



#     # Load 3DPW-TEST for diagnostics
#     cfg_eval = dataset_eval_config()
#     dataset_cfg = cfg_eval['3DPW-TEST']
#     dataset_cfg.defrost()
#     dataset_cfg.DATASET_FILE = str(get_humans_root() / 'hmr2_evaluation_data' / '3dpw_test.npz')
#     dataset_cfg.freeze()
    
#     from hmr2.datasets import ImageDataset
#     dataset = ImageDataset(
#         cfg=model.cfg,
#         dataset_file=dataset_cfg.DATASET_FILE,
#         img_dir=str(resolve_data_path('3DPW')),
#         train=False
#     )
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    
#     evaluator = Evaluator(
#         dataset_length=len(dataset),
#         keypoint_list=dataset_cfg.KEYPOINT_LIST,
#         pelvis_ind=39,
#         metrics=['mode_mpjpe']
#     )
    
def run_diagnostics(
    ckpt_path,
    output_root,
    gpu="0",
    run_name="Diagnostics",
    model=None,
    num_batches=20,
    chapter="Ch5",
    kti_mode="edge_ratio",
):
    """Scientific Diagnostics (Entropy, Rank, KTI)"""
    logger.info(
        f"🔬 Starting 4-Metric Scientific Diagnostics (num_batches={num_batches}) for Chapter [{chapter}]..."
    )
    
    device = torch.device(f'cuda:{gpu}')
    
    # 1. 加载模型逻辑 (使用统一的 Path Patcher 加载器)
    if model is None:
        from nvit.utils.model_io import load_model_from_ckpt
        model = load_model_from_ckpt(ckpt_path, device=device)
    else:
        logger.info("♻️ Reusing model instance from previous stage.")
        model.to(device)
        
    model.to(device)
    model.eval()
    
    wrapper = HMR2Wrapper(model)
    lab = ViTDiagnosticLab(
        wrapper,
        model_name=run_name,
        output_root=output_root,
        kti_mode=kti_mode,
    )
    lab.groups = {'Control': {'mask_layers': [], 'mode': 'none'}}
    
    # 2. 加载数据集配置 (关键点！)
    cfg_eval = dataset_eval_config()
    dataset_cfg = cfg_eval['3DPW-TEST']
    dataset_cfg.defrost()
    dataset_cfg.DATASET_FILE = str(get_humans_root() / 'hmr2_evaluation_data' / '3dpw_test.npz')
    dataset_cfg.freeze()

    # --- 这里是你插入的打印代码，必须放在 dataset_cfg 定义之后 ---
    print("-" * 30)
    print(f"Keypoint List: {dataset_cfg.KEYPOINT_LIST}")
    
    # 【修复重点 1】动态寻找盆骨索引
    # 你的列表是 [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 43]
    # 43 才是真正的对齐点（H36M Pelvis），它在列表中的位置是第 13 个（索引 13）
    try:
        p_idx = dataset_cfg.KEYPOINT_LIST.index(43)
        print(f"✅ Detected Pelvis (43) at List Index: {p_idx}")
    except ValueError:
        p_idx = 0
        print(f"⚠️ Pelvis (43) not found, fallback to Index 0")
    print("-" * 30)

    # 3. 数据加载 (保持不变...)
    from hmr2.datasets import ImageDataset
    dataset = ImageDataset(
        cfg=model.cfg,
        dataset_file=dataset_cfg.DATASET_FILE,
        img_dir=str(resolve_data_path('3DPW')),
        train=False
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    
    # 【修复重点 2】把硬编码的 39 改成 p_idx
    # 之前填 39 必崩，因为你的列表一共才 14 个元素，索引 39 越界直接触发 CUDA Assert
    evaluator = Evaluator(
        dataset_length=len(dataset),
        keypoint_list=dataset_cfg.KEYPOINT_LIST,
        pelvis_ind=p_idx, # <--- 改成动态索引
        metrics=['mode_mpjpe']
    )
    
    lab.run_experiment(dataloader, evaluator, dataset_cfg, num_batches=num_batches)
    return str(lab.output_dir)

def summarize_results(chapter, run_name, suite_json, diag_dir, output_root):
    """Combines metrics into a single summary.csv"""
    logger.info(f"📊 Summarizing Chapter {chapter} results for {run_name}...")
    
    summary = {"Run": run_name, "Chapter": chapter}
    
    # 1. Load Human Suite
    if os.path.exists(suite_json):
        with open(suite_json, 'r') as f:
            data = json.load(f)
            results = data.get('results', {})
            for ds, m in results.items():
                if 'mode_mpjpe' in m:
                    summary[f"{ds}_MPJPE"] = m['mode_mpjpe']
                if 'mode_re' in m:
                    summary[f"{ds}_PA_MPJPE"] = m['mode_re']
                if 'mode_kpl2' in m:
                    summary[f"{ds}_KPL2"] = m['mode_kpl2']

    # 2. Load Diagnostics
    # Look for the last run's metrics in diag_dir
    if diag_dir and os.path.exists(diag_dir):
        diag_results_csv = Path(diag_dir) / "results.csv"
        if diag_results_csv.exists():
            df = pd.read_csv(diag_results_csv)
            # We take the mean across all layers for the summary, 
            # but the full curve is preserved in the run's folder.
            if 'Avg_MAD' in df: summary['MAD'] = float(df['Avg_MAD'].iloc[-1])
            if 'Avg_KTI' in df: summary['KTI'] = float(df['Avg_KTI'].iloc[-1])
            if 'Avg_KTI_ER' in df: summary['KTI_ER'] = float(df['Avg_KTI_ER'].iloc[-1])
            if 'Avg_KTI_Corr' in df: summary['KTI_Corr'] = float(df['Avg_KTI_Corr'].iloc[-1])
            if 'KTI_Mode' in df: summary['KTI_Mode'] = str(df['KTI_Mode'].iloc[-1])
            if 'Avg_Rank' in df: summary['EffectiveRank'] = float(df['Avg_Rank'].iloc[-1])
            if 'Avg_Entropy' in df: summary['Entropy'] = float(df['Avg_Entropy'].iloc[-1])

    # Save summary
    summary_file = Path(output_root) / "summary.csv"
    df_summary = pd.DataFrame([summary])
    if summary_file.exists():
        df_old = pd.read_csv(summary_file)
        df_summary = pd.concat([df_old, df_summary]).drop_duplicates(subset=['Run'], keep='last')
    
    df_summary.to_csv(summary_file, index=False)
    logger.info(f"✅ Summary written to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="NViT Global Evaluator (Ch4-Ch6)")
    parser.add_argument("--chapter", type=str, required=True, choices=['Ch4', 'Ch5', 'Ch6A', 'Ch6B'])
    parser.add_argument("--run_path", type=str, default=None, help="Path to the training run outputs")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Direct path to a .ckpt file")
    parser.add_argument(
        "--run_label",
        type=str,
        default=None,
        help="Name for this run in summary.csv and eval_global output dir (default: basename of run_path or checkpoint_path)",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--diag_batches",
        type=int,
        default=20,
        help="Batches of 3DPW-TEST (bs=1) for internal metrics: entropy, MAD, KTI, effective rank.",
    )
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit number of batches per dataset for quick testing")
    parser.add_argument("--datasets", type=str, default="ALL", help="Comma-separated list of datasets or ALL")
    parser.add_argument(
        "--kti_mode",
        type=str,
        default="edge_ratio",
        choices=["edge_ratio", "dist_corr", "cosine"],
        help="Internal KTI metric mode for diagnostics.",
    )
    args = parser.parse_args()

    # 1. Checkpoint Resolution
    effective_path = args.checkpoint_path if args.checkpoint_path else args.run_path
    if not effective_path:
        logger.error("❌ Either --run_path or --checkpoint_path must be provided.")
        return

    run_name = args.run_label or os.path.basename(effective_path.rstrip("/"))
    # Sanitize folder name: avoid path separators / odd chars from experiment strings
    run_name = run_name.replace("/", "_").replace("\\", "_")
    output_root = BASE_DIR / "outputs" / "eval_global" / args.chapter / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    ckpt_path, ckpt_type = get_checkpoint(effective_path)
    if not ckpt_path:
        logger.error(f"❌ No checkpoint found at {effective_path}")
        return

    # 1. 加载模型 (共享实例)
    from nvit.utils.model_io import load_model_from_ckpt
    device = f"cuda:{args.gpu}"
    model = load_model_from_ckpt(ckpt_path, device=device)
    model.eval()

    # 2. Human Metric Suite
    torch.cuda.empty_cache()
    suite_json = run_human_suite(ckpt_path, output_root, gpu=args.gpu, datasets=args.datasets, parent_args=args)

    # 3. Diagnostic Metrics
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    diag_dir = None
    if args.chapter in ["Ch5", "Ch6A", "Ch6B"]:
        # Pass the parent of output_root so the lab creates its own subfolder (which is output_root)
        diag_dir = run_diagnostics(
            ckpt_path,
            output_root.parent,
            gpu=args.gpu,
            run_name=run_name,
            model=model,
            num_batches=args.diag_batches,
            chapter=args.chapter,
            kti_mode=args.kti_mode,
        )

    # 4. Final Aggregation
    summarize_results(args.chapter, run_name, suite_json, diag_dir, BASE_DIR / "outputs" / "eval_global" / args.chapter)
    
    # 5. Layer-Wise Diagnostic Plotting (NViT Auto-Gen)
    from nvit.skills.evaluate_model.layer_plotter import generate_comparative_plots
    _orun = (os.environ.get("NVIT_LAYER_PLOT_ONLY_RUNS") or "").strip()
    _only = {p.strip() for p in _orun.split(",") if p.strip()} if _orun else None
    generate_comparative_plots(
        args.chapter, BASE_DIR / "outputs" / "eval_global", only_runs=_only
    )
    logger.info(f"✨ Workflow Finalized. Layer Comparison visuals generated for {args.chapter}.")

if __name__ == "__main__":
    main()
