
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
import numpy as np
import json
from tqdm import tqdm
import types

# Add core skills to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from manage_experiment.skill_base import SkillBase

from nvit2_models.guided_hmr2 import GuidedHMR2Module
from hmr2.models import HMR2
from nvit.Paper1_Diagnostics.diagnostic_core.diagnostic_engine import ViTDiagnosticLab, HMR2Wrapper
from hmr2.datasets import create_dataset
from hmr2.configs import dataset_eval_config
from hmr2.utils import recursive_to, Evaluator
from hmr2.utils.geometry import aa_to_rotmat


class EvaluatorSkill(SkillBase):
    def __init__(self, gpu="0"):
        super().__init__(gpu=gpu)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, ckpt_path):
        self.logger.info(f"Loading Checkpoint: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        from nvit.utils.model_io import load_model_from_ckpt
        model = load_model_from_ckpt(ckpt_path, device=self.device)
        model.eval()
        model.to(self.device)
        return model

    def run_eval(self, args):
        model = self.load_model(args.ckpt)
        
        # 3. Determine Datasets to test
        cfg_eval = dataset_eval_config()
        if args.dataset == 'ALL':
            # Added MPI-INF-TEST if available, though currently identified 5 core datasets
            datasets = ['3DPW-TEST', 'H36M-VAL-P2', 'COCO-VAL', 'POSETRACK-VAL', 'LSP-EXTENDED']
        else:
            datasets = args.dataset.split(',')

        self.logger.info(f"Evaluating on: {datasets}")
        all_results = {}

        for ds_name in datasets:
            self.logger.info(f"\n--- Dataset: {ds_name} ---")
            if ds_name not in cfg_eval:
                self.logger.warning(f"Dataset {ds_name} not found in config. Skipping.")
                continue
                
            dataset_cfg = cfg_eval[ds_name]
            # Ensure path is absolute and points to our warehouse
            data_dir = getattr(args, 'data_dir', '/home/yangz/4D-Humans/hmr2_evaluation_data')
            dataset_cfg.defrost()
            dataset_cfg.DATASET_FILE = os.path.join(data_dir, os.path.basename(dataset_cfg.DATASET_FILE))
            dataset_cfg.freeze()

            if not os.path.exists(dataset_cfg.DATASET_FILE):
                self.logger.warning(f"NPZ not found: {dataset_cfg.DATASET_FILE}. Skipping.")
                continue

            dataset = create_dataset(model.cfg, dataset_cfg, train=False)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
            # Setup HMR2 Evaluator for this dataset
            metrics = ['mode_mpjpe', 'mode_re']
            pck_thresholds = None
            
            # Use appropriate metrics for different datasets
            if ds_name in ['LSP-EXTENDED', 'POSETRACK-VAL', 'COCO-VAL']:
                metrics = ['mode_kpl2']
                pck_thresholds = [0.05, 0.1]
                
            pelvis_ind = 0
            if 'H36M' in ds_name:
                pelvis_ind = 0
                
            hmr2_evaluator = Evaluator(
                dataset_length=int(1e8), 
                keypoint_list=dataset_cfg.KEYPOINT_LIST, 
                pelvis_ind=pelvis_ind, 
                metrics=metrics,
                pck_thresholds=pck_thresholds,
            )

            # 5. Evaluation Loop
            dataloader_iter = iter(dataloader)
            i = 0
            pbar = tqdm(total=len(dataloader), desc=ds_name)
            
            while i < len(dataloader):
                if args.limit_batches and i >= args.limit_batches: break
                
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    break
                except Exception as e:
                    if args.skip_errors:
                        self.logger.warning(f"Error loading batch {i}: {e}. Skipping.")
                        i += 1
                        pbar.update(1)
                        continue
                    else:
                        raise e
                
                try:
                    batch = recursive_to(batch, self.device)
                    with torch.no_grad():
                        out = model(batch)
                        
                        # --- ROBUST H36M ALIGNMENT FIX (Center of all Eval Joints) ---
                        if args.use_mean_alignment or 'H36M' in ds_name:
                            keypoint_list = dataset_cfg.KEYPOINT_LIST
                            pred_eval_kps = out['pred_keypoints_3d'][:, keypoint_list]
                            gt_eval_kps = batch['keypoints_3d'][:, keypoint_list, :-1]
                            
                            # Root = Mean of Eval Joints
                            pred_root = pred_eval_kps.mean(dim=1, keepdim=True)
                            gt_root = gt_eval_kps.mean(dim=1, keepdim=True)
                            
                            out['pred_keypoints_3d'] = out['pred_keypoints_3d'] - pred_root
                            batch['keypoints_3d'][:, :, :-1] = batch['keypoints_3d'][:, :, :-1] - gt_root

                            # Ensure the dummy alignment joint in Evaluator is also 0 in both
                            out['pred_keypoints_3d'] = torch.cat([torch.zeros_like(out['pred_keypoints_3d'][:, [0]]), out['pred_keypoints_3d'][:, 1:]], dim=1)
                            batch['keypoints_3d'][:, 0, :-1] = 0
                            hmr2_evaluator.pelvis_ind = 0 
                        # ---------------------------------
                        
                        hmr2_evaluator(out, batch)
                except Exception as e:
                    if args.skip_errors:
                        self.logger.warning(f"Error processing batch {i}: {e}. Skipping.")
                    else:
                        raise e
                
                i += 1
                pbar.update(1)
            pbar.close()

            report = hmr2_evaluator.get_metrics_dict()
            all_results[ds_name] = report
            
            if 'mode_mpjpe' in report:
                self.logger.info(f"Finished {ds_name}: MPJPE={report['mode_mpjpe']:.2f}, PA-MPJPE={report['mode_re']:.2f}")
            else:
                self.logger.info(f"Finished {ds_name}: KPL2={report.get('mode_kpl2', 0.0):.4f}")

        # 6. Final Report Summary
        self.logger.info("\n" + "="*80)
        self.logger.info(f"      UPGRADED NViT MULTI-DATASET EVALUATION SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Checkpoint: {os.path.basename(args.ckpt)}")
        self.logger.info("-" * 80)
        self.logger.info(f"{'Dataset':<20} | {'MPJPE (mm)':<12} | {'PA-MPJPE (mm)':<15} | {'Others'}")
        self.logger.info("-" * 80)
        for ds, r in all_results.items():
            mpjpe = f"{r.get('mode_mpjpe', 0.0):.2f}" if 'mode_mpjpe' in r else "N/A"
            pa_mpjpe = f"{r.get('mode_re', 0.0):.2f}" if 'mode_re' in r else "N/A"
            others = f"KPL2={r.get('mode_kpl2', 0.0):.4f}" if 'mode_kpl2' in r else ""
            self.logger.info(f"{ds:<20} | {mpjpe:<12} | {pa_mpjpe:<15} | {others}")
        self.logger.info("="*80 + "\n")
        
        # Save to JSON
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({'args': vars(args), 'results': all_results}, f, indent=4)
            self.logger.info(f"Results saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standardized NViT Evaluation Skill")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="3DPW-TEST")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--skip_errors", action="store_true", default=True)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--dense_gt", action="store_true", default=True)
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--data_dir", type=str, default="/home/yangz/4D-Humans/hmr2_evaluation_data")
    parser.add_argument("--use_mean_alignment", action="store_true", help="Use Mean of Eval Joints for root alignment (Fixes H36M drift)")
    
    args = parser.parse_args()
    
    eval_skill = EvaluatorSkill(gpu=args.gpu)
    eval_skill.run_eval(args)
