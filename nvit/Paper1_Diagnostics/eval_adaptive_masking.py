import os
import sys
import torch
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/home/yangz/4D-Humans')
sys.path.insert(0, '/home/yangz/NViT-master/nvit/Paper1_Diagnostics')

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.datasets import HMR2DataModule
from hmr2.configs import dataset_eval_config
from hmr2.utils import Evaluator
from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper
from tqdm import tqdm

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--group', type=str, required=True, help='Which group to evaluate')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    group = args.group
    
    # Dataset
    dataset_cfg = dataset_eval_config()
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({'TRAIN': {'BATCH_SIZE': 16, 'NUM_WORKERS': 4}, 'GENERAL': {'NUM_WORKERS': 4}})
    
    from hmr2.datasets.image_dataset import ImageDataset
    dataset_file = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    
    try:
        _, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        val_ds = ImageDataset(m_cfg, dataset_file, img_dir=img_dir, train=False)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    ckpt_path = f'checkpoints/ft_{group}.ckpt'
    if not os.path.exists(ckpt_path):
        print(f"Skipping {group}: Checkpoint not found at {ckpt_path}")
        return

    print(f"Evaluating {group} on GPU {args.gpu}...")
    model, _ = load_hmr2(DEFAULT_CHECKPOINT)
    
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    model.to(device)
    model.eval()

    wrapper = get_wrapper(model, 'HMR2')
    lab = ViTDiagnosticLab(wrapper, model_name=f"Eval_{group}", output_root='eval_results', track_metrics=False)
    lab.apply_single_intervention(group)

    evaluator = Evaluator(dataset_length=int(1e8), keypoint_list=dataset_cfg['3DPW-TEST'].KEYPOINT_LIST, pelvis_ind=m_cfg.EXTRA.PELVIS_IND, metrics=['mode_mpjpe', 'mode_re'])

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Eval {group}"):
            batch = wrapper.to_device(batch, device)
            lab.update_batch_state(batch)
            out = model(batch)
            evaluator(out, batch)

    metrics = evaluator.get_metrics_dict()
    mpjpe = metrics.get('mode_mpjpe', 0)
    pa_mpjpe = metrics.get('mode_re', 0)
    
    print(f"Results for {group}: MPJPE={mpjpe:.2f}, PA-MPJPE={pa_mpjpe:.2f}")

    # Append to CSV safely
    csv_path = 'finetune_eval_results.csv'
    new_data = pd.DataFrame([{'Group': group, 'MPJPE': mpjpe, 'PA-MPJPE': pa_mpjpe}])
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[existing.Group != group] # Remove old entry if strictly appending
        df = pd.concat([existing, new_data], ignore_index=True)
    else:
        df = new_data
    df.to_csv(csv_path, index=False)
    print(f"Saved {group} results to finetune_eval_results.csv")

if __name__ == '__main__':
    main()
