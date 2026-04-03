#!/home/yangz/.conda/envs/4D-humans/bin/python
import os
import sys
import torch
import numpy as np
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from unittest.mock import MagicMock

# Setup paths
repo_root = Path(__file__).resolve().parent.parent.parent
diag_dir = repo_root / 'nvit/Paper1_Diagnostics'
sys.path.append(str(diag_dir))
sys.path.append(str(diag_dir / 'diagnostic_core'))
sys.path.append('/home/yangz/4D-Humans')

# Mocks
class NamedMock(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__qualname__ = "Mock"
    def _get_child_mock(self, **kwargs):
        return NamedMock(**kwargs)

sys.modules['wis3d'] = NamedMock()
skel_mock = NamedMock()
skel_mock.__path__ = []
sys.modules['skel'] = skel_mock
sys.modules['skel.osim_rot'] = NamedMock()
sys.modules['skel.skel_model'] = NamedMock()

try:
    from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper
    from hmr2.models import load_hmr2
    from hmr2.utils import Evaluator
    from hmr2.configs import dataset_eval_config
    from hmr2.datasets import ImageDataset
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Model Loading Logic ---
def load_target_model(model_name, ckpt_path=None):
    if 'hmr2' in model_name.lower():
        if ckpt_path is None:
            ckpt_path = '/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
        return load_hmr2(ckpt_path)
    
    elif 'hsmr' in model_name.lower():
        hsmr_root = repo_root / 'nvit/external_models/HSMR'
        if str(hsmr_root) not in sys.path:
            sys.path.insert(0, str(hsmr_root))
            sys.path.insert(0, str(hsmr_root / 'lib'))
        from modeling.pipelines.hsmr import build_inference_pipeline
        if ckpt_path is None:
            ckpt_path = str(hsmr_root / 'data_inputs/released_models/HSMR-ViTH-r1d1/checkpoints/hsmr.ckpt')
            model_dir = hsmr_root / 'data_inputs/released_models/HSMR-ViTH-r1d1'
        else:
            model_dir = Path(ckpt_path).parent.parent
        model = build_inference_pipeline(model_root=model_dir, ckpt_fn=ckpt_path, device='cuda')
        model.eval()
        from yacs.config import CfgNode as CN
        dummy_cfg = CN(); dummy_cfg.MODEL = CN(); dummy_cfg.MODEL.IMAGE_SIZE = 256
        dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]; dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
        dummy_cfg.SMPL = CN(); dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
        dummy_cfg.DATASETS = CN(); dummy_cfg.DATASETS.CONFIG = CN()
        return model, dummy_cfg

    elif 'prompthmr' in model_name.lower():
        phmr_root = repo_root / 'nvit/external_models/PromptHMR'
        if str(phmr_root) not in sys.path: sys.path.insert(0, str(phmr_root))
        from prompt_hmr import load_model_from_folder
        if ckpt_path is None: ckpt_path = str(phmr_root / 'data/pretrain/phmr_vid')
        model = load_model_from_folder(ckpt_path)
        model.eval()
        from yacs.config import CfgNode as CN
        dummy_cfg = CN(); dummy_cfg.MODEL = CN(); dummy_cfg.MODEL.IMAGE_SIZE = 256
        dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]; dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
        dummy_cfg.SMPL = CN(); dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
        dummy_cfg.DATASETS = CN(); dummy_cfg.DATASETS.CONFIG = CN()
        return model, dummy_cfg

    elif 'camerahmr' in model_name.lower():
        camerahmr_root = repo_root / 'nvit/external_models/CameraHMR'
        if str(camerahmr_root) not in sys.path: sys.path.insert(0, str(camerahmr_root))
        from core.camerahmr_model import CameraHMR
        if ckpt_path is None: ckpt_path = str(camerahmr_root / 'data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt')
        model = CameraHMR.load_from_checkpoint(ckpt_path, strict=False, model_type='smpl', map_location='cuda')
        model.eval()
        from yacs.config import CfgNode as CN
        dummy_cfg = CN(); dummy_cfg.MODEL = CN(); dummy_cfg.MODEL.IMAGE_SIZE = 256
        dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]; dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
        dummy_cfg.SMPL = CN(); dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
        dummy_cfg.DATASETS = CN(); dummy_cfg.DATASETS.CONFIG = CN()
        return model, dummy_cfg
    
    return None, None

def save_metrics(lab, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for l, v in lab.layer_metrics.items():
        serializable[l] = {k: [float(x) for x in vals] for k, vals in v.items()}
    with open(output_dir / 'layer_metrics.json', 'w') as f:
        json.dump(serializable, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['HMR2', 'PromptHMR', 'HSMR', 'CameraHMR'])
    parser.add_argument('--num_batches', type=int, default=10)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()

    dataset_file = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    output_root = Path('outputs/diagnostics')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        torch.manual_seed(seed); np.random.seed(seed)
        
        for model_name in args.models:
            seed_out_dir = output_root / model_name / f'seed_{seed}'
            if (seed_out_dir / 'layer_metrics.json').exists():
                print(f"⏩ Skipping {model_name} Seed {seed} (already exists)")
                continue

            print(f"\n🚀 Evaluating {model_name}...")
            # More aggressive cleanup before loading
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            model, model_cfg = load_target_model(model_name)
            if model is None: continue
            model.to(device)
            
            wrapper = get_wrapper(model, model_name)
            lab = ViTDiagnosticLab(wrapper, model_name=model_name, output_root=seed_out_dir)
            lab.groups = {'Control': {'mask_layers': [], 'mode': 'none'}}
            
            dataset_cfg = dataset_eval_config()['3DPW-TEST']
            dataset = ImageDataset(cfg=model_cfg, dataset_file=dataset_file, img_dir=img_dir, train=False)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
            
            evaluator = Evaluator(dataset_length=len(dataset), keypoint_list=dataset_cfg.KEYPOINT_LIST, pelvis_ind=39, metrics=['mode_mpjpe'])
            
            lab.run_experiment(data_loader, evaluator, dataset_cfg, num_batches=args.num_batches)
            save_metrics(lab, seed_out_dir)
            print(f"✅ Saved to {seed_out_dir}")
            
            # Extremely aggressive cleanup to avoid OOM
            del lab
            del wrapper
            if hasattr(model, 'cpu'): model.cpu()
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

if __name__ == "__main__":
    main()
