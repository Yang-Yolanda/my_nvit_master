import os
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
sys.path.append('/home/yangz/4D-Humans') # 4D-Humans root

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
    """Find best.ckpt if available, else last.ckpt"""
    ckpt_dir = Path(run_path) / "checkpoints"
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

def run_human_suite(ckpt_path, output_dir, gpu="0", datasets="ALL"):
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
    args.limit_batches = None
    args.skip_errors = True
    args.use_mean_alignment = True # Critical for H3.6M
    args.data_dir = '/home/yangz/4D-Humans/hmr2_evaluation_data'
    args.output = str(Path(output_dir) / "metrics_suite.json")
    
    skill.run_eval(args)
    return args.output

def run_diagnostics(ckpt_path, output_dir, gpu="0", num_batches=10, chapter="Ch4"):
    logger.info(f"🔬 Starting 4-Metric Scientific Diagnostics for Chapter [{chapter}]...")
    
    device = torch.device(f'cuda:{gpu}')
    
    # Intelligently detect architecture from checkpoint instead of hardcoding by Chapter!
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        decpose_weight = state_dict.get('smpl_head.decpose.weight')
        if decpose_weight is not None and decpose_weight.shape[0] == 144:
            from hmr2.models.hmr2 import HMR2
            model = HMR2.load_from_checkpoint(ckpt_path, strict=False, map_location=device)
            logger.info("Detected Legacy [144] SMPLHead in checkpoint. Loaded standard HMR2.")
        else:
            from nvit2_models.guided_hmr2 import GuidedHMR2Module
            model = GuidedHMR2Module.load_from_checkpoint(ckpt_path, strict=False, map_location=device)
            logger.info("Detected Guided [6] SMPLHead in checkpoint. Loaded GuidedHMR2Module.")
    except Exception as e:
        logger.warning(f"Inspection failed, defaulting to GuidedHMR2Module: {e}")
        from nvit2_models.guided_hmr2 import GuidedHMR2Module
        model = GuidedHMR2Module.load_from_checkpoint(ckpt_path, strict=False, map_location=device)
        
    model.to(device)
    model.eval()
    
    wrapper = HMR2Wrapper(model)
    # output_root here will be used to create [model_name] subfolder
    lab = ViTDiagnosticLab(wrapper, model_name="diagnostics", output_root=output_dir)
    
    # Only keep 'Control' group for diagnostics
    lab.groups = {'Control': {'mask_layers': [], 'mode': 'none'}}
    
    # Load 3DPW-TEST for diagnostics
    cfg_eval = dataset_eval_config()
    dataset_cfg = cfg_eval['3DPW-TEST']
    dataset_cfg.defrost()
    dataset_cfg.DATASET_FILE = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    dataset_cfg.freeze()
    
    from hmr2.datasets import ImageDataset
    dataset = ImageDataset(
        cfg=model.cfg,
        dataset_file=dataset_cfg.DATASET_FILE,
        img_dir='/home/yangz/4D-Humans/data/3DPW',
        train=False
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    
    evaluator = Evaluator(
        dataset_length=len(dataset),
        keypoint_list=dataset_cfg.KEYPOINT_LIST,
        pelvis_ind=39,
        metrics=['mode_mpjpe']
    )
    
    lab.run_experiment(dataloader, evaluator, dataset_cfg, num_batches=num_batches)
    
    # The lab results are in lab.output_dir / results.csv
    # We should also ensure the 4 metrics are easily accessible
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
    diag_results_csv = Path(diag_dir) / "results.csv"
    if diag_results_csv.exists():
        df = pd.read_csv(diag_results_csv)
        # We take the mean across all layers for the summary, 
        # but the full curve is preserved in the run's folder.
        if 'Avg_MAD' in df: summary['MAD'] = float(df['Avg_MAD'].iloc[-1])
        if 'Avg_KTI' in df: summary['KTI'] = float(df['Avg_KTI'].iloc[-1])
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
    parser.add_argument("--run_path", type=str, required=True, help="Path to the training run outputs")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--diag_batches", type=int, default=20)
    parser.add_argument("--datasets", type=str, default="ALL", help="Comma-separated list of datasets or ALL")
    args = parser.parse_args()

    run_name = os.path.basename(args.run_path)
    output_root = BASE_DIR / "outputs" / "eval_global" / args.chapter / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    # Hard Safety Check for GPU 2/3
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', "")
    if args.gpu in ["2", "3"] or "2" in visible_devices.split(',') or "3" in visible_devices.split(','):
        logger.error(f"❌ [GPU-VIOLATION] Attempted to use reserved GPU 2/3. Aborting evaluator.")
        sys.exit(1)

    # 1. Checkpoint Resolution
    ckpt_path, ckpt_type = get_checkpoint(args.run_path)
    if not ckpt_path:
        logger.error(f"❌ No checkpoint found in {args.run_path}")
        return

    logger.info(f"🎯 Evaluating {run_name} [Chapter: {args.chapter}] using {ckpt_type} checkpoint: {ckpt_path}")

    # 2. Human Metric Suite
    suite_json = run_human_suite(ckpt_path, output_root, gpu=args.gpu, datasets=args.datasets)

    # 3. Diagnostic Metrics (Ch6B might skip if non-human)
    diag_dir = None
    if args.chapter != 'Ch6B':
        diag_dir = run_diagnostics(ckpt_path, output_root, gpu=args.gpu, num_batches=args.diag_batches, chapter=args.chapter)

    # 4. Final Aggregation
    summarize_results(args.chapter, run_name, suite_json, diag_dir, BASE_DIR / "outputs" / "eval_global" / args.chapter)
    
    # 5. Layer-Wise Diagnostic Plotting (NViT Auto-Gen)
    from nvit.skills.evaluate_model.layer_plotter import generate_comparative_plots
    generate_comparative_plots(args.chapter, BASE_DIR / "outputs" / "eval_global")
    logger.info(f"✨ Workflow Finalized. Layer Comparison visuals generated for {args.chapter}.")

if __name__ == "__main__":
    main()
