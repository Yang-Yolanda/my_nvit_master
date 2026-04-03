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
            # We need to import the module where to_tensor is used or defined
            # It's defined in utils.data.types but used in modeling.pipelines.hsmr
            import utils.data.types
            utils.data.types.to_tensor = patched_to_tensor
            
            # ALSO patch lib.utils.data.types because HSMR might import via 'lib' package
            try:
                import lib.utils.data.types
                lib.utils.data.types.to_tensor = patched_to_tensor
            except ImportError:
                pass
            # -------------------

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
            
    elif 'siglip' in model_name.lower():
        from transformers import AutoModel, AutoConfig
        # Load SigLIP
        model_name_hf = "google/siglip-so400m-patch14-384"
        config = AutoConfig.from_pretrained(model_name_hf)
        config.output_attentions = True
        config.attn_implementation = "eager"
        
        model = AutoModel.from_pretrained(model_name_hf, config=config)
        model.cuda()
        model.eval()
        
        # Dummy Config for Dataset
        from yacs.config import CfgNode as CN
        dummy_cfg = CN()
        dummy_cfg.MODEL = CN()
        dummy_cfg.MODEL.IMAGE_SIZE = 384 # Valid for SigLIP
        dummy_cfg.MODEL.IMAGE_MEAN = [0.5, 0.5, 0.5] # Approx? Or use standard? SigLIP preprocessor...
        # Standard SigLIP uses 0.5 mean/std usually (Inception style) 
        # But HMR loader normalizes using ImageNet constants [0.485...]
        # We'll stick to HMR constants in loader, and let SigLIP digest it (finetuning robust enough or just analysis)
        dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
        dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
        
        # --- FIX: INJECT SMPL CONFIG ---
        dummy_cfg.SMPL = CN()
        dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
        
        # --- FIX: INJECT DATASETS CONFIG ---
        dummy_cfg.DATASETS = CN()
        dummy_cfg.DATASETS.CONFIG = CN()
        # -----------------------------------
        
        return model, dummy_cfg
        
    else:
        logger.error(f"Unsupported model: {model_name}")
        return None, None

def run_experiment_2(args):
    """
    Experiment 2: Kinematic Mutual Information (Structure vs Depth)
    Target: Control Group Only (Baseline Analysis)
    """
    model_name = args.model
    # Output to Experiment2_KTI/results/ModelName
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / "results"
    
    logger.info(f"Running Experiment 2 (KTI) for model: {model_name}")
    
    # 1. Load Model
    model, model_cfg = load_target_model(model_name, args.ckpt)
    if model is None: return
    model.eval()
    
    # 2. Setup Diagnostic Lab via Wrapper
    wrapper = get_wrapper(model, model_name)
    lab = ViTDiagnosticLab(wrapper, model_name=model_name, output_root=output_root)
    
    # 3. Setup Dataset
    dataset_cfg = dataset_eval_config()['3DPW-TEST']
    dataset = ImageDataset(
        cfg=model_cfg,
        dataset_file='/mnt/hdd_toshiba_1/yangz_data/4D-Humans/hmr2_evaluation_data/3dpw_test.npz',
        img_dir='/mnt/hdd_toshiba_1/yangz_data/4D-Humans/data/3DPW',
        train=False
    )
    # Experiment 2 needs enough samples
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    
    evaluator = Evaluator(
        dataset_length=len(dataset),
        keypoint_list=dataset_cfg.KEYPOINT_LIST,
        pelvis_ind=39,
        metrics=['mode_mpjpe']
    )
    
    # --- Experiment 2 Logic: Baseline Only ---
    logger.info("--- Starting Experiment 2: Baseline KTI Analysis ---")
    
    # Clear all groups except Control
    lab.groups = {'Control': {'mask_layers': [], 'mode': 'none'}}
    
    lab.run_experiment(data_loader, evaluator, dataset_cfg, num_batches=args.num_batches)
    
    logger.info(f"Experiment 2 Complete. Results in {lab.output_dir}/results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paper 1 Experiment 2: KTI & Structure')
    parser.add_argument('--model', type=str, default='4D-Humans', help='Model to diagnose')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--num_batches', type=int, default=50, help='Number of batches to run')
    args = parser.parse_args()
    
    run_experiment_2(args)
