#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import sys
import os
import argparse
from pathlib import Path
import logging
import numpy as np
import random
from yacs.config import CfgNode as CN

# --- Patch for xformers compatibility with older torch ---
import sys
from unittest.mock import MagicMock
if not hasattr(torch.backends.cuda, 'is_flash_attention_available'):
    torch.backends.cuda.is_flash_attention_available = lambda: False

# Aggressively mock xformers if it fails to import
try:
    import xformers
except ImportError:
    pass
except Exception:
    pass
    
# Force mock to prevent dynamic link errors even if import succeeded partially
sys.modules['xformers'] = MagicMock()
sys.modules['xformers.ops'] = MagicMock()
sys.modules['xformers.ops.fmha'] = MagicMock()
sys.modules['xformers.ops.fmha.flash'] = MagicMock()
sys.modules['wis3d'] = MagicMock()

# Also mock torch.backends.cuda attributes that might trigger checks
if not hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
    torch.backends.cuda.flash_sdp_enabled = lambda: False
    
# --- Mock 'skel' module for HSMR ---
# This is required because 'skel' seems to be a missing compiled extension
try:
    import skel
except ImportError:
    skel_mock = MagicMock()
    osim_rot = MagicMock()
    # Mock classes used in definition.py
    for cls_name in ['ConstantCurvatureJoint', 'CustomJoint', 'EllipsoidJoint', 'PinJoint', 'WalkerKnee']:
        setattr(osim_rot, cls_name, MagicMock)
    skel_mock.osim_rot = osim_rot
    # Add skel_model mock
    skel_model = MagicMock()
    skel_mock.skel_model = skel_model
    # Ensure skel is treated as a package
    skel_mock.__path__ = []
    
    sys.modules['skel'] = skel_mock
    sys.modules['skel.osim_rot'] = osim_rot
    sys.modules['skel.skel_model'] = skel_model
# ---------------------------------------------------------

# --- HSMR RUNTIME PATCH for Mock compatibility ---
# If to_tensor receives a Mock, it crashes. We patch it to return a dummy tensor.
def patched_to_tensor(x, device=None, temporary=False):
    import torch
    import numpy as np
    from typing import List
    
    # Check for Mock FIRST
    if isinstance(x, MagicMock) or 'Mock' in str(type(x)):
        # Return a dummy tensor of shape (B, 44, 3) or similar? 
        # Actually standard usage in HSMR is for keypoints.
        # Let's return a safe zero tensor.
        # We don't know the shape easily, but (1, 44, 3) is a safe guess for batch 1.
        return torch.zeros((1, 44, 3), device=device if device else 'cuda')
        
    # Original logic (copied from types.py essentially)
    if isinstance(x, torch.Tensor):
        device = x.device if device is None else device
        if temporary:
            recover_type_back = lambda x_: x_.to(x.device)
            return x.to(device), recover_type_back
        else:
            return x.to(device)

    device = 'cpu' if device is None else device
    if isinstance(x, np.ndarray):
        if temporary:
            recover_type_back = lambda x_: x_.detach().cpu().numpy()
            return torch.from_numpy(x).to(device), recover_type_back
        else:
            return torch.from_numpy(x).to(device)
    if isinstance(x, list):
         return torch.from_numpy(np.array(x)).to(device)
         
    # Fallback
    return torch.tensor(x, device=device)

# We need to inject this patch AFTER HSMR modules are loaded but BEFORE model runs.
# We will do it inside the HSMR loading block.
# -------------------------------------------------

# Add 4D-Humans to path
sys.path.append('/home/yangz/4D-Humans')

# Add diagnostic core to path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'diagnostic_core'))
from diagnostic_engine import ViTDiagnosticLab, get_wrapper

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
    """Generalized model loader. Returns (model, model_cfg)"""
    if '4d-humans' in model_name.lower() or 'hmr2' in model_name.lower():
        if ckpt_path is None:
             ckpt_path = '/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
        return load_hmr2(ckpt_path) # Returns model, cfg
    elif 'camerahmr' in model_name.lower():
        # Setup CameraHMR path
        camerahmr_root = '/home/yangz/NViT-master/nvit/external_models/CameraHMR'
        if camerahmr_root not in sys.path:
            sys.path.insert(0, camerahmr_root)
            
        try:
            from core.camerahmr_model import CameraHMR
            from core.constants import CHECKPOINT_PATH
            
            # Load default checkpoint if not provided
            if ckpt_path is None:
                ckpt_path = os.path.join(camerahmr_root, 'data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt')
            
            if not os.path.exists(ckpt_path):
                logger.error(f"CameraHMR Checkpoint NOT FOUND at: {ckpt_path}")
                # Try to find it?
                # fallback
                return None, None
                
            logger.info(f"Loading CameraHMR from {ckpt_path}")
            model = CameraHMR.load_from_checkpoint(ckpt_path, strict=False, model_type='smpl')
            model.eval()
            
            # CameraHMR doesn't expose a config object easily, use HMR2's for dataset
            # Assuming CameraHMR input size is 224 or 256. 
            # Looking at mesh_estimator.py, it uses IMAGE_SIZE constant.
            from core.constants import IMAGE_SIZE
            
            # Create a dummy config for dataset loader
            from yacs.config import CfgNode as CN
            dummy_cfg = CN()
            dummy_cfg.MODEL = CN()
            dummy_cfg.MODEL.IMAGE_SIZE = IMAGE_SIZE # Usually 224
            dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
            dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
            
            # --- FIX: INJECT SMPL CONFIG ---
            dummy_cfg.SMPL = CN()
            dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
            
            # --- FIX: INJECT DATASETS CONFIG ---
            dummy_cfg.DATASETS = CN()
            dummy_cfg.DATASETS.CONFIG = CN()
            # -----------------------------------
            
            sys.path.pop(0) # Cleanup path
            return model, dummy_cfg
            
        except ImportError as e:
            logger.error(f"Failed to import CameraHMR: {e}")
            if camerahmr_root in sys.path:
                sys.path.pop(0)
            return None, None
            
    elif 'hsmr' in model_name.lower():
        # Setup HSMR path
        hsmr_root = '/home/yangz/NViT-master/nvit/external_models/HSMR'
        # Add hsmr root AND lib to path to find 'skel' and other modules
        if hsmr_root not in sys.path:
            sys.path.insert(0, hsmr_root)
            sys.path.insert(0, os.path.join(hsmr_root, 'lib'))
            
        try:
            # --- APPLY PATCH ---
            try:
                import lib.utils.data.types
                lib.utils.data.types.to_tensor = patched_to_tensor
                logger.info("Successfully patched lib.utils.data.types.to_tensor")
            except ImportError as e:
                logger.warning(f"Could not patch lib.utils.data.types: {e}")
            # -------------------
            
            # --- FIX HSMR CONFIG ---
            # from lib.core.config import cfg  # CAUSES CRASH: lib.core missing
            # abs_skel_path = '/home/yangz/NViT-master/external_models/HSMR/data_inputs/body_models/skel'
            # (Patch applied directly to hsmr.py instead)
            # -----------------------

            from modeling.pipelines.hsmr import build_inference_pipeline
            # Also try to import skel to ensure it works
            # import skel
            
            # Default checkpoint path logic inside build_inference_pipeline is strictly relative
            # We must pass the FOLDER containing .hydra as model_root
            if ckpt_path is None:
                ckpt_path = os.path.join(hsmr_root, 'data_inputs/released_models/HSMR-ViTH-r1d1/checkpoints/hsmr.ckpt')
                model_dir = Path(hsmr_root) / 'data_inputs/released_models/HSMR-ViTH-r1d1'
            else:
                 # Derive model root from ckpt path: parent of parent (checkpoints -> experiment_root)
                 if os.path.exists(ckpt_path):
                     model_dir = Path(ckpt_path).parent.parent
                 else:
                     model_dir = Path(hsmr_root) / 'data_inputs/released_models/HSMR-ViTH-r1d1'

            model = build_inference_pipeline(model_root=model_dir, ckpt_fn=ckpt_path, device='cuda')
            model.eval()
            
            # HSMR typically uses 256x256 input patch (then backbone crops to 256x192)
            # We use HMR2 dataset loader which outputs 256x256
            from yacs.config import CfgNode as CN
            dummy_cfg = CN()
            dummy_cfg.MODEL = CN()
            dummy_cfg.MODEL = CN()
            # 256 for HMR2/ViT-H (16 patch), 224 for PromptHMR (14 patch)
            if 'prompthmr' in model_name.lower():
                dummy_cfg.MODEL.IMAGE_SIZE = 224
            else:
                dummy_cfg.MODEL.IMAGE_SIZE = 256
            dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
            dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]

            # --- FIX: INJECT SMPL CONFIG ---
            dummy_cfg.SMPL = CN()
            dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
            
            # --- FIX: INJECT DATASETS CONFIG ---
            dummy_cfg.DATASETS = CN()
            dummy_cfg.DATASETS.CONFIG = CN()
            # -----------------------------------
            
            sys.path.pop(0)
            return model, dummy_cfg
            
        except ImportError as e:
            logger.error(f"Failed to import HSMR: {e}")
            sys.path.pop(0)
            return None, None

    elif 'prompthmr' in model_name.lower():
        # Setup PromptHMR path
        phmr_root = '/home/yangz/NViT-master/nvit/external_models/PromptHMR'
        if phmr_root not in sys.path:
            sys.path.insert(0, phmr_root)
            
        try:
            from prompt_hmr import load_model_from_folder
            
            # Checkpoint handling
            # PromptHMR usually expects a folder containing 'checkpoint.ckpt' and 'config.yaml'
            if ckpt_path is None:
                ckpt_path = os.path.join(phmr_root, 'data/pretrain/phmr_vid')
            
            if not os.path.exists(ckpt_path):
                 logger.warning(f"PromptHMR checkpoint folder not found at {ckpt_path}")
            
            # Using folder loader as seen in demo
            model = load_model_from_folder(ckpt_path)
            model.eval()
            
            # Config for dataset (PromptHMR uses 256 or 896? demo uses 896 or 256. 
            # image_encoder.py uses dinov2_vitb14 which expects patchified input.
            # DINOv2 is flexible. PromtHMR config usually handles it.
            # We'll use 256 for standardization with HMR2 unless it fails.
            from yacs.config import CfgNode as CN
            dummy_cfg = CN()
            dummy_cfg.MODEL = CN()
            dummy_cfg.MODEL.IMAGE_SIZE = 256
            
            dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
            dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
            
            # --- FIX: INJECT SMPL CONFIG ---
            dummy_cfg.SMPL = CN()
            dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
            
            # --- FIX: INJECT DATASETS CONFIG ---
            dummy_cfg.DATASETS = CN()
            dummy_cfg.DATASETS.CONFIG = CN()
            # -----------------------------------
            
            sys.path.pop(0)
            return model, dummy_cfg
            
        except ImportError as e:
            logger.error(f"Failed to import PromptHMR: {e}")
            sys.path.pop(0)
            return None, None

    elif any(m in model_name.lower() for m in ['spin', 'pare', 'cliff', 'hybrik', 'mmhuman3d']):
        # MMHuman3D Models
        mm_root = '/home/yangz/mmhuman3d'
        p3d_root = os.path.join(mm_root, 'pytorch3d')
        
        for p in [mm_root, p3d_root]:
            if p not in sys.path:
                sys.path.insert(0, p)
            
        try:
            from mmhuman3d.apis import init_model
            
            # Map model name to config
            # This is a heuristic mapping based on common file names
            config_map = {
                'spin': 'configs/spin/resnet50_spin_pw3d.py',
                'pare': 'configs/pare/hrnet_w32_conv_pare_coco.py', # Example
                'cliff': 'configs/cliff/resnet50_cliff_gt-bbox_coco.py' # Example
            }
            
            # Find best match
            cfg_path = None
            for k, v in config_map.items():
                if k in model_name.lower():
                    cfg_path = os.path.join(mm_root, v)
                    break
            
            if cfg_path is None or not os.path.exists(cfg_path):
                logger.warning(f"Could not automatically find config for {model_name}. Please rely on default or provide specific name.")
                # Fallback to SPIN hardcoded for now if generic
                cfg_path = os.path.join(mm_root, 'configs/spin/resnet50_spin_pw3d.py')
            
            logger.info(f"Loading MMHuman3D model using config: {cfg_path}")
            
            # Checkpoint
            # If ckpt_path is None, init_model will load weights if defined in config or random init?
            # User must provide checkpoint or rely on MMHuman3D auto-download (if supported)
            # MMHuman3D often puts 'load_from' in config.
            
            model, _ = init_model(cfg_path, checkpoint=ckpt_path, device='cuda')
            model.eval()
            
            # Create dummy config for HMR2 dataset loader
            from yacs.config import CfgNode as CN
            dummy_cfg = CN()
            dummy_cfg.MODEL = CN()
            dummy_cfg.MODEL.IMAGE_SIZE = 224 # SPIN uses 224 usually
            
            sys.path.pop(0)
            return model, dummy_cfg
            
        except ImportError as e:
            logger.error(f"Failed to import MMHuman3D: {e}")
            sys.path.pop(0)
            return None, None
            
    else:
        logger.error(f"Unsupported model: {model_name}")
        return None, None

def run_experiment_3(args):
    """
    Experiment 3: Attention Entropy & Gaze (Globality vs locality)
    """
    model_name = args.model
    model_name = args.model
    # Output to Experiment3_Masking/results/ModelName
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / "results"
    
    logger.info(f"Running Experiment 3 (Masking) for model: {model_name}")
    
    # 1. Load Model
    model, model_cfg = load_target_model(model_name, args.ckpt)
    if model is None: return
    model.eval()
    
    # 2. Setup Diagnostic Lab via Wrapper
    wrapper = get_wrapper(model, model_name)
    lab = ViTDiagnosticLab(wrapper, model_name=model_name, output_root=output_root)
    
    # 3. Setup Dataset & Evaluator (Generic enough for cross-model benchmarks)
    # Note: Using HMR2 dataset for standardized comparison.
    dataset_cfg = dataset_eval_config()['3DPW-TEST']
    
    # DEBUG: Check size
    print(f"DEBUG Runner: model_cfg.MODEL.IMAGE_SIZE = {model_cfg.MODEL.IMAGE_SIZE}", flush=True)
    
    dataset = ImageDataset(
        cfg=model_cfg,
        dataset_file='/mnt/hdd_toshiba_1/yangz_data/4D-Humans/data/metadata/3dpw_test.npz',
        img_dir='/mnt/hdd_toshiba_1/yangz_data/4D-Humans/data/3DPW',
        train=False
    )
    print(f"DEBUG Runner: Dataset IMG_SIZE = {getattr(dataset, 'IMG_SIZE', 'Unknown')}", flush=True)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    
    evaluator = Evaluator(
        dataset_length=len(dataset),
        keypoint_list=dataset_cfg.KEYPOINT_LIST,
        pelvis_ind=39,
        metrics=['mode_mpjpe']
    )
    
    # --- Phase 1: Comprehensive Combinatorial Sweep (Run by Default) ---
    logger.info("--- Starting Phase 1: Comprehensive Combinatorial Sweep (1/4 & 1/3 Splits, Hybrid Mixes) ---")
    logger.info(f"Generated {len(lab.groups)} groups.")
    lab.run_experiment(data_loader, evaluator, dataset_cfg, num_batches=args.num_batches)
    
    # --- Phase 2: Adaptive Split (Optional Check) ---
    logger.info("--- Starting Phase 2: Adaptive Optimization Check ---")
    # 1. Ask engine for optimal layers based on previous run metrics
    split_1, split_2 = lab.analyze_metrics_and_suggest_split()
    logger.info(f"Adaptive Analysis suggests optimal splits at Layer {split_1} (Mamba) and {split_2} (GCN)")
    
    # 2. Reset groups to run ONLY the adaptive one
    lab.groups = {} 
    lab.add_adaptive_group(split_1, split_2)
    
    # 3. Run Adaptive
    lab.run_experiment(data_loader, evaluator, dataset_cfg, num_batches=args.num_batches)
    
    logger.info(f"Experiment 3 Complete. Results in {lab.output_dir}/results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paper 1 Experiment 3: Kinematic Masking Sweep')
    parser.add_argument('--model', type=str, default='4D-Humans', help='Model to diagnose')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to run')
    args = parser.parse_args()
    
    run_experiment_3(args)
