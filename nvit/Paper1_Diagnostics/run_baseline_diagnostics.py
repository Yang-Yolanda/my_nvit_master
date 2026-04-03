import torch
import sys
import os
import argparse
import logging
import pandas as pd
from pathlib import Path

# Setup paths
curr_dir = Path(__file__).resolve().parent
sys.path.append(str(curr_dir))
sys.path.append(str(curr_dir / 'diagnostic_core'))
sys.path.append('/home/yangz/4D-Humans')

# --- Mocks for compatibility ---
from unittest.mock import MagicMock
sys.modules['wis3d'] = MagicMock()
# Mock 'skel' for HSMR
skel_mock = MagicMock()
skel_mock.__path__ = []
sys.modules['skel'] = skel_mock
sys.modules['skel.osim_rot'] = MagicMock()
sys.modules['skel.skel_model'] = MagicMock()
# ------------------------------

from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper

# Logic for HMR2/4D-Humans dependencies
try:
    from hmr2.models import load_hmr2
    from hmr2.utils import Evaluator
    from hmr2.configs import dataset_eval_config
    from hmr2.datasets import ImageDataset
except ImportError:
    print("Error: HMR2/4D-Humans dependencies not found.")

def load_target_model(model_name, ckpt_path=None):
    """Generalized model loader. Returns (model, model_cfg)"""
    if '4d-humans' in model_name.lower() or 'hmr2' in model_name.lower():
        if ckpt_path is None:
             ckpt_path = '/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
        return load_hmr2(ckpt_path)
    
    elif 'hsmr' in model_name.lower():
        hsmr_root = '/home/yangz/NViT-master/nvit/external_models/HSMR'
        if hsmr_root not in sys.path:
            sys.path.insert(0, hsmr_root)
            sys.path.insert(0, os.path.join(hsmr_root, 'lib'))
        from modeling.pipelines.hsmr import build_inference_pipeline
        if ckpt_path is None:
            ckpt_path = os.path.join(hsmr_root, 'data_inputs/released_models/HSMR-ViTH-r1d1/checkpoints/hsmr.ckpt')
            model_dir = Path(hsmr_root) / 'data_inputs/released_models/HSMR-ViTH-r1d1'
        else:
            model_dir = Path(ckpt_path).parent.parent
        model = build_inference_pipeline(model_root=model_dir, ckpt_fn=ckpt_path, device='cuda')
        model.eval()
        from yacs.config import CfgNode as CN
        dummy_cfg = CN()
        dummy_cfg.MODEL = CN()
        dummy_cfg.MODEL.IMAGE_SIZE = 256
        dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
        dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
        dummy_cfg.SMPL = CN()
        dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
        dummy_cfg.DATASETS = CN()
        dummy_cfg.DATASETS.CONFIG = CN()
        return model, dummy_cfg

    elif 'prompthmr' in model_name.lower():
        phmr_root = '/home/yangz/NViT-master/nvit/external_models/PromptHMR'
        if phmr_root not in sys.path:
            sys.path.insert(0, phmr_root)
        from prompt_hmr import load_model_from_folder
        if ckpt_path is None:
            ckpt_path = os.path.join(phmr_root, 'data/pretrain/phmr_vid')
        model = load_model_from_folder(ckpt_path)
        model.eval()
        from yacs.config import CfgNode as CN
        dummy_cfg = CN()
        dummy_cfg.MODEL = CN()
        dummy_cfg.MODEL.IMAGE_SIZE = 256
        dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
        dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
        dummy_cfg.SMPL = CN()
        dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
        dummy_cfg.DATASETS = CN()
        dummy_cfg.DATASETS.CONFIG = CN()
        return model, dummy_cfg
    elif 'camerahmr' in model_name.lower():
        camerahmr_root = '/home/yangz/NViT-master/nvit/external_models/CameraHMR'
        if camerahmr_root not in sys.path:
            sys.path.insert(0, camerahmr_root)
        from core.camerahmr_model import CameraHMR
        if ckpt_path is None:
            ckpt_path = os.path.join(camerahmr_root, 'data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt')
        model = CameraHMR.load_from_checkpoint(ckpt_path, strict=False, model_type='smpl', map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        from yacs.config import CfgNode as CN
        dummy_cfg = CN()
        dummy_cfg.MODEL = CN()
        dummy_cfg.MODEL.IMAGE_SIZE = 256
        dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
        dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
        dummy_cfg.SMPL = CN()
        dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
        dummy_cfg.DATASETS = CN()
        dummy_cfg.DATASETS.CONFIG = CN()
        return model, dummy_cfg
    
    return None, None

def run_diagnostics(models_to_run, num_batches=10):
    dataset_file = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    output_root = curr_dir / 'Experiment1_Entropy' / 'results'
    output_root.mkdir(parents=True, exist_ok=True)

    models_to_run_std = [m.replace('prompthmr', 'PromptHMR').replace('camerahmr', 'CameraHMR').replace('hmr2', 'HMR2').replace('hsmr', 'HSMR') for m in models_to_run]
    for model_name in models_to_run_std:
        print(f"\n🚀 Running diagnostics for {model_name}...")
        model, model_cfg = load_target_model(model_name)
        if model is None:
            print(f"❌ Failed to load {model_name}")
            continue
        
        wrapper = get_wrapper(model, model_name)
        lab = ViTDiagnosticLab(wrapper, model_name=model_name, output_root=output_root)
        
        # Only keep 'Control' group for Phase 1 initial diagnostics
        lab.groups = {'Control': {'mask_layers': [], 'mode': 'none'}}
        
        dataset_cfg = dataset_eval_config()['3DPW-TEST']
        dataset = ImageDataset(
            cfg=model_cfg,
            dataset_file=dataset_file,
            img_dir=img_dir,
            train=False
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
        
        evaluator = Evaluator(
            dataset_length=len(dataset),
            keypoint_list=dataset_cfg.KEYPOINT_LIST,
            pelvis_ind=39,
            metrics=['mode_mpjpe']
        )
        
        lab.run_experiment(data_loader, evaluator, dataset_cfg, num_batches=num_batches)
        print(f"✅ Completed {model_name}. Results saved to {lab.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['HMR2', 'PromptHMR', 'HSMR', 'CameraHMR'])
    parser.add_argument('--num_batches', type=int, default=10)
    args = parser.parse_args()
    
    run_diagnostics(args.models, args.num_batches)
