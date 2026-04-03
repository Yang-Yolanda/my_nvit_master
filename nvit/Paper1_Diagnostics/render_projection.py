import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, '/home/yangz/4D-Humans')
sys.path.insert(0, '/home/yangz/NViT-master/nvit/Paper1_Diagnostics')

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.datasets.image_dataset import ImageDataset
from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--img_idx', type=int, default=150)
    parser.add_argument('--occlusion', type=float, default=0.3)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Models
    _, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model_control, _ = load_hmr2(DEFAULT_CHECKPOINT)
    model_control.load_state_dict(torch.load('checkpoints/ft_Control.ckpt', map_location='cpu'), strict=False)
    model_control.to(device).eval()
    
    model_adaptive, _ = load_hmr2(DEFAULT_CHECKPOINT)
    model_adaptive.load_state_dict(torch.load('checkpoints/ft_T2-KTI-Adaptive.ckpt', map_location='cpu'), strict=False)
    model_adaptive.to(device).eval()

    # Apply Interventions
    get_wrapper(model_adaptive, 'HMR2').apply_single_intervention('T2-KTI-Adaptive')
    get_wrapper(model_control, 'HMR2').apply_single_intervention('Control')

    # Dataset
    dataset_file = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    val_ds = ImageDataset(m_cfg, dataset_file, img_dir=img_dir, train=False)
    batch = val_ds[args.img_idx]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)

    # Occlusion
    if args.occlusion > 0:
        _, C, H, W = batch['img'].shape
        occ_h, occ_w = int(H * args.occlusion), int(W * args.occlusion)
        rng = np.random.RandomState(args.img_idx) 
        top, left = rng.randint(0, H - occ_h), rng.randint(0, W - occ_w)
        batch['img'][:, :, top:top+occ_h, left:left+occ_w] = 0.0

    # Inference
    with torch.no_grad():
        out_control = model_control(batch)
        out_adaptive = model_adaptive(batch)

    # 2D Projection (Simple)
    def project_and_draw(img_tensor, out_dict, color=(0, 255, 0)):
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img = (img_tensor[0].cpu().permute(1,2,0).numpy() * std + mean) * 255
        img = np.clip(img, 0, 255).astype(np.uint8).copy()
        
        # Simple projection: SMPL usually gives [B, 44, 2] or similar keypoints
        # HMR2 out['pred_keypoints_2d'] is [B, N, 2] in normalized coords? 
        # Actually it's [B, 44, 2] in pixels of the patch (224x224)
        kp2d = out_dict['pred_keypoints_2d'][0].detach().cpu().numpy()
        for i in range(kp2d.shape[0]):
            x, y = int(kp2d[i, 0]), int(kp2d[i, 1])
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                cv2.circle(img, (x, y), 2, color, -1)
        return img

    viz_control = project_and_draw(batch['img'], out_control, color=(0, 0, 255)) # Red for Control
    viz_adaptive = project_and_draw(batch['img'], out_adaptive, color=(0, 255, 0)) # Green for Adaptive

    combined = np.concatenate([viz_control, viz_adaptive], axis=1)
    out_path = f'projection_compare_idx{args.img_idx}_occ{args.occlusion}.png'
    cv2.imwrite(out_path, combined[:, :, ::-1])
    print(f"Saved projection visualization to {out_path}")

if __name__ == '__main__':
    main()
