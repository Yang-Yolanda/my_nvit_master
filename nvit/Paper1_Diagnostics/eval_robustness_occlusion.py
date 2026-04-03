import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# Add paths
sys.path.insert(0, '/home/yangz/4D-Humans')
sys.path.insert(0, '/home/yangz/NViT-master/nvit/Paper1_Diagnostics')

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.datasets import HMR2DataModule
from hmr2.configs import dataset_eval_config
from hmr2.utils import Evaluator
from hmr2.datasets.image_dataset import ImageDataset
from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper

def apply_random_occlusion(img_tensor, occlusion_ratio=0.2, seed=42):
    """
    Apply a random black box occlusion to a batch of images.
    img_tensor: [B, 3, H, W]
    occlusion_ratio: ratio of the image side to be occluded (e.g. 0.2 means 20% of width/height)
    """
    B, C, H, W = img_tensor.shape
    occ_h = int(H * occlusion_ratio)
    occ_w = int(W * occlusion_ratio)
    
    # Use a local random generator with a fixed seed for reproducibility across groups
    rng = np.random.RandomState(seed)
    
    masked_img = img_tensor.clone()
    for i in range(B):
        top = rng.randint(0, H - occ_h)
        left = rng.randint(0, W - occ_w)
        masked_img[i, :, top:top+occ_h, left:left+occ_w] = 0.0
        
    return masked_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--group', type=str, required=True, help='Intervention group')
    parser.add_argument('--occlusion', type=float, default=0.2, help='Ratio of image side to occlude (0.0 to 1.0)')
    parser.add_argument('--num_samples', type=int, default=None, help='Limit evaluation to N samples')
    parser.add_argument('--num_batches', type=int, default=None, help='Limit evaluation to N batches')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model Config
    _, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    
    # Dataset Setup (3DPW Test)
    dataset_cfg = dataset_eval_config()
    dataset_file = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    
    val_ds = ImageDataset(m_cfg, dataset_file, img_dir=img_dir, train=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    # Load Fine-tuned Checkpoint
    ckpt_path = f'checkpoints/ft_{args.group}.ckpt'
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    model, _ = load_hmr2(DEFAULT_CHECKPOINT)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    model.to(device)
    model.eval()

    # Diagnostic Wrapper (Apply Masking)
    wrapper = get_wrapper(model, 'HMR2')
    lab = ViTDiagnosticLab(wrapper, model_name=f"Robust_{args.group}", track_metrics=False)
    lab.apply_single_intervention(args.group)

    # Evaluator
    evaluator = Evaluator(dataset_length=len(val_ds), 
                          keypoint_list=dataset_cfg['3DPW-TEST'].KEYPOINT_LIST, 
                          pelvis_ind=m_cfg.EXTRA.PELVIS_IND, 
                          metrics=['mode_mpjpe', 'mode_re'])

    print(f"Running Robustness Test for {args.group} (Occlusion={args.occlusion}) on GPU {args.gpu}")
    
    all_mpjpes = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Robustness {args.group}")):
            if args.num_batches is not None and batch_idx >= args.num_batches:
                break
            if args.num_samples is not None and batch_idx * 16 >= args.num_samples:
                break
                
            batch = wrapper.to_device(batch, device)
            
            # Apply fixed-seed occlusion per batch for cross-model fairness
            if args.occlusion > 0:
                batch['img'] = apply_random_occlusion(batch['img'], occlusion_ratio=args.occlusion, seed=batch_idx)
            
            lab.update_batch_state(batch)
            out = model(batch)
            eval_out = evaluator(out, batch)
            
            # Extract batch MPJPEs
            if 'mode_mpjpe' in eval_out:
                val = eval_out['mode_mpjpe']
                if hasattr(val, 'detach'):
                    all_mpjpes.append(val.detach().cpu().numpy())
                else:
                    all_mpjpes.append(val)

    metrics = evaluator.get_metrics_dict()
    mpjpe = metrics.get('mode_mpjpe', 0)
    pa_mpjpe = metrics.get('mode_re', 0)
    
    # Flatten all_mpjpes
    if all_mpjpes:
        all_mpjpes = np.concatenate(all_mpjpes)
    else:
        all_mpjpes = np.array([])
    
    print(f"Robust Results ({args.occlusion}): MPJPE={mpjpe:.2f}, PA-MPJPE={pa_mpjpe:.2f}")

    # ---------------- 增加 DDP 层面的中间数据导出逻辑 ----------------
    if hasattr(lab, 'export_intermediate_metrics'):
        lab.export_intermediate_metrics(run_id=f"{args.group}_Occ{args.occlusion}", dataset_split="3DPW_TEST", ckpt_name=args.group)
    # -----------------------------------------------------------

    # Save to a dedicated robustness CSV
    res_file = 'robustness_results.csv'
    res_data = pd.DataFrame([{
        'Group': args.group,
        'Occlusion': args.occlusion,
        'MPJPE': mpjpe,
        'PA-MPJPE': pa_mpjpe
    }])
    
    if os.path.exists(res_file):
        existing = pd.read_csv(res_file)
        # Remove entry if exact group/occlusion combo exists
        existing = existing[~((existing.Group == args.group) & (existing.Occlusion == args.occlusion))]
        df = pd.concat([existing, res_data], ignore_index=True)
    else:
        df = res_data
        
    df.to_csv(res_file, index=False)
    
    # Save raw errors for distribution plots
    out_dir = Path("outputs/ch5_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"robust_errors_{args.group}_occ{args.occlusion}.npy", all_mpjpes)

if __name__ == '__main__':
    main()
