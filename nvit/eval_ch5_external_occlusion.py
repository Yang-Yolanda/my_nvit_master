import os
import sys
import torch
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/home/yangz/4D-Humans')
sys.path.insert(0, '/home/yangz/NViT-master/nvit/Paper1_Diagnostics')

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.configs import dataset_eval_config
from hmr2.utils import Evaluator
from hmr2.datasets.image_dataset import ImageDataset
from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper

def apply_random_occlusion(img_tensor, imgnames, occlusion_ratio=0.2):
    B, C, H, W = img_tensor.shape
    occ_h = int(H * occlusion_ratio)
    occ_w = int(W * occlusion_ratio)
    
    masked_img = img_tensor.clone()
    for i in range(B):
        # Use md5 digest for deterministic per-sample seeding across processes/runs
        import hashlib
        s = str(imgnames[i]).encode("utf-8")
        seed = int(hashlib.md5(s).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        
        top = rng.randint(0, H - occ_h)
        left = rng.randint(0, W - occ_w)
        masked_img[i, :, top:top+occ_h, left:left+occ_w] = 0.0
        
    return masked_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ckpt', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--group', type=str, required=True, help='Intervention group (M0-M6)')
    parser.add_argument('--output_json', type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    
    dataset_cfg = dataset_eval_config()
    dataset_file = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    
    val_ds = ImageDataset(m_cfg, dataset_file, img_dir=img_dir, train=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found at {args.ckpt}")
        return

    model, _ = load_hmr2(DEFAULT_CHECKPOINT)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=False)
    model.to(device)
    model.eval()

    wrapper = get_wrapper(model, 'HMR2')
    # Use lab to bypass any missing forward components if needed, though standard evaluator is usually enough.
    # But since Paper 1 uses ViTDiagnosticLab to wrap it, we'll keep it to be safe.
    lab = ViTDiagnosticLab(wrapper, model_name=f"Robust_{args.group}", track_metrics=True)
    # lab.apply_single_intervention(args.group) -> Not needed because M0-M6 checkpoints ALREADY contain the soft/hard masks during normal forward

    results = {}
    occlusion_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for occ in occlusion_levels:
        evaluator = Evaluator(dataset_length=len(val_ds), 
                              keypoint_list=dataset_cfg['3DPW-TEST'].KEYPOINT_LIST, 
                              pelvis_ind=m_cfg.EXTRA.PELVIS_IND, 
                              metrics=['mode_mpjpe', 'mode_re'])

        print(f"Running Occlusion {occ} for {args.group}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader)):
                batch = wrapper.to_device(batch, device)
                if occ > 0:
                    imgnames = batch.get('imgname', [str(batch_idx * 16 + i) for i in range(batch['img'].shape[0])])
                    batch['img'] = apply_random_occlusion(batch['img'], imgnames, occlusion_ratio=occ)

                out = model(batch)
                evaluator(out, batch)

        metrics = evaluator.get_metrics_dict()
        mpjpe = metrics.get('mode_mpjpe', 0)
        pa_mpjpe = metrics.get('mode_re', 0)
        
        # If evaluator returns rank-0 tensors
        if torch.is_tensor(mpjpe):
            mpjpe = mpjpe.item()
        if torch.is_tensor(pa_mpjpe):
            pa_mpjpe = pa_mpjpe.item()

        results[str(occ)] = {'MPJPE': mpjpe, 'PA-MPJPE': pa_mpjpe}
        print(f"[{args.group}] Occ={occ} -> MPJPE: {mpjpe:.1f}")

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Occlusion results saved to {args.output_json}")

if __name__ == '__main__':
    main()
