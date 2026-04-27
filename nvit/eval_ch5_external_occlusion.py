import os
from nvit.utils.path_utils import get_humans_root, get_project_root, resolve_data_path

import sys
import torch
import numpy as np
import json
import argparse
from typing import Any
from tqdm import tqdm

_PAPER1 = get_project_root() / "nvit" / "Paper1_Diagnostics"
sys.path.insert(0, str(get_humans_root()))
sys.path.insert(0, str(_PAPER1))

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.configs import dataset_eval_config
from hmr2.utils import Evaluator
from hmr2.datasets.image_dataset import ImageDataset
from diagnostic_core.diagnostic_engine import get_wrapper
from nvit.utils.model_io import load_model_from_ckpt, patch_hmr2_config

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


def _dataset_cfg_for_model(loaded_model) -> Any:
    """HMR2 / GuidedHMR2: use checkpoint cfg when available."""
    h = getattr(loaded_model, "hparams", None)
    if h is not None:
        c = getattr(h, "cfg", None)
        if c is not None:
            return patch_hmr2_config(c)
    _, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    return m_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ckpt', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--group', type=str, required=True, help='Intervention group (M0-M6)')
    parser.add_argument('--output_json', type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Only use the first N samples of 3DPW-TEST (faster ablation; default: full set).",
    )
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Stop after this many dataloader batches (overrides max_samples in practice if smaller).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint not found at {args.ckpt}", file=sys.stderr)
        return 1
    if args.gpu is not None and str(args.gpu).lower() not in ("none", ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dload = "cuda:0"
    else:
        device = torch.device("cpu")
        dload = "cpu"

    model = load_model_from_ckpt(args.ckpt, device=dload)
    model = model.to(device)
    model.eval()
    m_cfg = _dataset_cfg_for_model(model)

    dataset_cfg = dataset_eval_config()
    dataset_file = str(get_humans_root() / 'hmr2_evaluation_data' / '3dpw_test.npz')
    img_dir = str(resolve_data_path('3DPW'))
    
    val_ds = ImageDataset(m_cfg, dataset_file, img_dir=img_dir, train=False)
    if args.max_samples is not None and args.max_samples < len(val_ds):
        n = int(args.max_samples)
        val_ds = torch.utils.data.Subset(val_ds, range(n))

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    wrapper = get_wrapper(model, "HMR2")
    # Do not use ViTDiagnosticLab here: its attention hooks assume NViT token layout and break plain ViT.

    results = {}
    occlusion_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    eval_len = len(val_loader.dataset)
    if args.limit_batches is not None:
        print(
            f"Warning: --limit_batches={args.limit_batches} limits throughput; "
            f"MPJPE may be biased vs full-set Evaluator length={eval_len}."
        )

    for occ in occlusion_levels:
        evaluator = Evaluator(dataset_length=eval_len, 
                              keypoint_list=dataset_cfg['3DPW-TEST'].KEYPOINT_LIST, 
                              pelvis_ind=m_cfg.EXTRA.PELVIS_IND, 
                              metrics=['mode_mpjpe', 'mode_re'])

        print(f"Running Occlusion {occ} for {args.group}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"occ={occ}")):
                if args.limit_batches is not None and batch_idx >= args.limit_batches:
                    break
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

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Occlusion results saved to {args.output_json}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main() or 0)
