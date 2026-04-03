#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import sys
import os
import argparse
from pathlib import Path
import logging

# Ensure we can import from the parent core directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper

# Logic for HMR2 import (Current Target Model)
try:
    from hmr2.models import load_hmr2
    from hmr2.utils import Evaluator
    from hmr2.configs import dataset_eval_config
    from hmr2.datasets import ImageDataset
except ImportError:
    print("Error: HMR2/4D-Humans dependencies not found.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_target_model(model_name, ckpt_path=None):
    """Generalized model loader."""
    if '4d-humans' in model_name.lower() or 'hmr2' in model_name.lower():
        if ckpt_path is None:
             ckpt_path = '/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
        model, _ = load_hmr2(ckpt_path)
        return model
    elif 'hsmr' in model_name.lower():
        # TODO: Implement HSMR loading logic here
        logger.warning("HSMR loading logic pending repo-specific setup.")
        return None
    else:
        logger.error(f"Unsupported model: {model_name}")
        return None

def run_experiment_2(args):
    """
    Experiment 2: Attention Entropy & Gaze (Globality vs locality)
    """
    model_name = args.model
    output_root = "results"
    
    logger.info(f"Running Experiment 2 (KTI) for model: {model_name}")
    
    # 1. Load Model
    model = load_target_model(model_name, args.ckpt)
    if model is None: return
    model.eval()
    
    # 2. Setup Diagnostic Lab via Wrapper
    wrapper = get_wrapper(model, model_name)
    lab = ViTDiagnosticLab(wrapper, model_name=model_name, output_root=output_root)
    
    # 3. Setup Dataset & Evaluator (Generic enough for cross-model benchmarks)
    # Note: Using HMR2 dataset for standardized comparison.
    dataset_cfg = dataset_eval_config()['3DPW-TEST']
    dataset = ImageDataset(
        cfg=dataset_cfg,
        dataset_file='/home/yangz/4D-Humans/data/3dpw_test.npz',
        img_dir='/home/yangz/4D-Humans/data/3DPW',
        train=False
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    
    evaluator = Evaluator(
        dataset_length=len(dataset),
        keypoint_list=dataset_cfg.KEYPOINT_LIST,
        pelvis_ind=39,
        metrics=['mode_mpjpe']
    )
    
    # 4. Filter groups for Experiment 2 (Usually just Control is enough for diagnostic curves)
    # We run the whole suite to see how masking affects this specific model.
    lab.run_experiment(data_loader, evaluator, dataset_cfg, num_batches=args.num_batches)
    
    logger.info(f"Experiment 2 Complete. Results in {lab.output_dir}/results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paper 1 Experiment 2: Kinematic Mutual Information')
    parser.add_argument('--model', type=str, default='4D-Humans', help='Model to diagnose')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to run')
    args = parser.parse_args()
    
    run_experiment_2(args)
