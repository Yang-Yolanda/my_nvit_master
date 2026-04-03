import torch
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
nvit_dir = Path("/home/yangz/NViT-master/nvit")
sys.path.append(str(nvit_dir))
sys.path.append(str(nvit_dir / 'Paper1_Diagnostics'))
sys.path.append(str(nvit_dir / 'Paper1_Diagnostics' / 'diagnostic_core'))
sys.path.append('/home/yangz/4D-Humans')

from hmr2.configs import dataset_eval_config
from hmr2.utils import Evaluator, recursive_to
from diagnostic_core.diagnostic_engine import get_wrapper

def custom_load_hmr2(checkpoint_path, device):
    from hmr2.models import HMR2
    from hmr2.configs import get_config
    
    # Get config (we copied it to Paper1_Diagnostics/model_config.yaml)
    model_cfg_path = "/home/yangz/NViT-master/nvit/Paper1_Diagnostics/model_config.yaml"
    model_cfg = get_config(model_cfg_path)
    
    # Override some config values, to crop bbox correctly (borrowed from hmr2.models.load_hmr2)
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Initialize model
    model = HMR2(model_cfg)
    
    # Load weights
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
        
    model = model.to(device)
    model.eval()
    return model, model_cfg

def run_eval_for_ckpt(ckpt_path, group_tag, device, num_batches=None):
    logger.info(f"Evaluating {group_tag} from {ckpt_path}...")
    
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}")
        return None
        
    try:
        model, cfg = custom_load_hmr2(ckpt_path, device)
    except Exception as e:
        logger.error(f"Failed to load {group_tag}: {e}")
        return None
    
    wrapper = get_wrapper(model, 'HMR2')
    cfg_eval = dataset_eval_config()['3DPW-TEST']
    
    # 3DPW paths
    dataset_file = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    
    from hmr2.datasets import ImageDataset
    dataset = ImageDataset(
        cfg=cfg,
        dataset_file=dataset_file,
        img_dir=img_dir,
        train=False
    )
    
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    total_length = len(dataset)
    if num_batches is not None:
        total_length = min(total_length, num_batches * batch_size)
        
    evaluator = Evaluator(
        dataset_length=total_length,
        keypoint_list=cfg_eval.KEYPOINT_LIST,
        pelvis_ind=39,
        metrics=['mode_mpjpe', 'mode_re']
    )
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Eval {group_tag}")):
            if num_batches is not None and i >= num_batches: break
            batch = recursive_to(batch, device)
            
            # Simple resize if needed
            if 'img' in batch and (batch['img'].shape[-1] != 256 or batch['img'].shape[-2] != 256):
                 import torch.nn.functional as F
                 batch['img'] = F.interpolate(batch['img'], size=(256, 256), mode='bilinear', align_corners=False)
            
            out = wrapper(batch)
            evaluator(out, batch)
            
    # Extract results
    mpjpes = getattr(evaluator, 'mode_mpjpe')[:evaluator.counter]
    pa_mpjpes = getattr(evaluator, 'mode_re')[:evaluator.counter]
    
    # Calculate stats
    mean_val = np.mean(mpjpes)
    std_val = np.std(mpjpes)
    
    # Extreme cases (Top 5% worst)
    threshold = np.percentile(mpjpes, 95)
    extreme_mpjpes = mpjpes[mpjpes >= threshold]
    extreme_mean = np.mean(extreme_mpjpes)
    extreme_std = np.std(extreme_mpjpes)
    
    results = {
        'Group': group_tag,
        'Mean MPJPE': mean_val,
        'Std MPJPE': std_val,
        'Extreme Mean (Top 5%)': extreme_mean,
        'Extreme Std': extreme_std,
        'PA-MPJPE': np.mean(pa_mpjpes)
    }
    
    # Save raw errors for later plotting (distribution)
    results_dir = Path("outputs/ch5_eval")
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / f"errors_{group_tag}.npy", mpjpes)
    
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_root = Path("Paper1_Diagnostics/checkpoints")
    
    exp_mapping = {
        'Exp0': ckpt_root / 'ft_Control.ckpt',
        'Exp1': ckpt_root / 'ft_T2-Static-Mid.ckpt',
        'Exp2': ckpt_root / 'ft_T2-Static-Late.ckpt',
        'Exp3': ckpt_root / 'ft_T2-A-S-Baseline.ckpt',
        'Exp4': ckpt_root / 'ft_T2-A-H-Baseline.ckpt',
        'Exp5': ckpt_root / 'ft_T2-KTI-Adaptive.ckpt'
    }
    
    overall_results = []
    
    # For quick testing, num_batches=10. For full, set to None.
    # User said they want a systematic evaluation, so full evaluation is preferred.
    NUM_BATCHES = None 
    
    for tag, path in exp_mapping.items():
        res = run_eval_for_ckpt(str(path), tag, device, num_batches=NUM_BATCHES)
        if res:
            overall_results.append(res)
            
    df = pd.DataFrame(overall_results)
    output_path = Path("outputs/ch5_eval/ch5_supplemental_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("Chapter 5 Supplemental Experiment Results")
    print("="*50)
    print(df.to_markdown(index=False))
    print("="*50)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
