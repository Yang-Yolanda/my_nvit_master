import os

# Set EGL for headless rendering BEFORE any other imports that might touch OpenGL
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/home/yangz/4D-Humans')
sys.path.insert(0, '/home/yangz/NViT-master/nvit/Paper1_Diagnostics')

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils.renderer import Renderer
from hmr2.datasets.image_dataset import ImageDataset
from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--img_idx', type=int, default=100, help='Index of image in 3DPW test set')
    parser.add_argument('--occlusion', type=float, default=0.3, help='Occlusion ratio')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Models
    print(">>> Stage 1: Loading models...")
    _, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    
    print(f">>> Loading checkpoints on {device}...")
    model_control, _ = load_hmr2(DEFAULT_CHECKPOINT)
    model_control.load_state_dict(torch.load('checkpoints/ft_Control.ckpt', map_location='cpu'), strict=False)
    model_control.to(device).eval()
    
    model_adaptive, _ = load_hmr2(DEFAULT_CHECKPOINT)
    model_adaptive.load_state_dict(torch.load('checkpoints/ft_T2-KTI-Adaptive.ckpt', map_location='cpu'), strict=False)
    model_adaptive.to(device).eval()

    print(">>> Applying interventions...")
    wrapper_adaptive = get_wrapper(model_adaptive, 'HMR2')
    lab_adaptive = ViTDiagnosticLab(wrapper_adaptive, model_name="Render_Adaptive", track_metrics=False)
    lab_adaptive.apply_single_intervention('T2-KTI-Adaptive')

    wrapper_control = get_wrapper(model_control, 'HMR2')
    lab_control = ViTDiagnosticLab(wrapper_control, model_name="Render_Control", track_metrics=False)
    lab_control.apply_single_intervention('Control')

    # 2. Setup Dataset & Get Image
    print(">>> Stage 2: Loading 3DPW dataset...")
    dataset_file = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    val_ds = ImageDataset(m_cfg, dataset_file, img_dir=img_dir, train=False)
    
    batch = val_ds[args.img_idx]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)

    # 3. Apply Occlusion
    if args.occlusion > 0:
        print(f">>> Applying {args.occlusion} occlusion...")
        # batch['img'] is [1, 3, H, W] here
        _, C, H, W = batch['img'].shape
        occ_h = int(H * args.occlusion)
        occ_w = int(W * args.occlusion)
        rng = np.random.RandomState(args.img_idx) 
        top = rng.randint(0, H - occ_h)
        left = rng.randint(0, W - occ_w)
        batch['img'][:, :, top:top+occ_h, left:left+occ_w] = 0.0

    # 4. Inference
    print(">>> Stage 3: Inference...")
    with torch.no_grad():
        lab_control.update_batch_state(batch)
        out_control = model_control(batch)
        
        lab_adaptive.update_batch_state(batch)
        out_adaptive = model_adaptive(batch)

    # 5. Rendering
    print(">>> Stage 4: Rendering (EGL Backend)...")
    try:
        renderer = Renderer(m_cfg, faces=model_control.smpl.faces)
        print(">>> Renderer initialized successfully.")
    except Exception as e:
        print(f">>> Failed to initialize Renderer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    def get_render(out_dict, batch_dict):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = batch_dict['img'][0].cpu().permute(1,2,0).numpy()
        input_patch = img_np * std + mean
        input_patch = np.clip(input_patch, 0, 1)
        
        regression_img = renderer(out_dict['pred_vertices'][0].detach().cpu().numpy(),
                                 out_dict['pred_cam_t'][0].detach().cpu().numpy(),
                                 batch_dict['img'][0],
                                 mesh_base_color=(0.65, 0.74, 0.86),
                                 scene_bg_color=(1, 1, 1))
        return input_patch, regression_img

    patch, render_control = get_render(out_control, batch)
    _, render_adaptive = get_render(out_adaptive, batch)

    # Side views
    def get_side_view(out_dict, batch_dict):
        white_img = torch.ones_like(batch_dict['img'][0])
        side_img = renderer(out_dict['pred_vertices'][0].detach().cpu().numpy(),
                           out_dict['pred_cam_t'][0].detach().cpu().numpy(),
                           white_img,
                           mesh_base_color=(0.65, 0.74, 0.86),
                           scene_bg_color=(1, 1, 1),
                           side_view=True)
        return side_img

    side_control = get_side_view(out_control, batch)
    side_adaptive = get_side_view(out_adaptive, batch)

    # Assemble final image
    top_row = np.concatenate([patch, render_control, render_adaptive], axis=1)
    bottom_row = np.concatenate([patch, side_control, side_adaptive], axis=1)
    final_img = np.concatenate([top_row, bottom_row], axis=0)

    out_path = f'comparison_render_idx{args.img_idx}_occ{args.occlusion}.png'
    cv2.imwrite(out_path, 255 * final_img[:, :, ::-1])
    print(f"Success! Saved comparison to {out_path}")

if __name__ == '__main__':
    main()
