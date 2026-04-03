import os
import sys
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/home/yangz/4D-Humans')
sys.path.insert(0, '/home/yangz/NViT-master/nvit/Paper1_Diagnostics')

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper
from hmr2.datasets.image_dataset import ImageDataset

def run_diagnostic_analysis(ckpt_path, name, gpu_id, num_batches=50):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Analyzing {name} on GPU {gpu_id} using {ckpt_path}")
    
    # 1. Load Model
    model, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    model.to(device).eval()
    
    # 2. Setup Dataset
    dataset_file = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    val_ds = ImageDataset(m_cfg, dataset_file, img_dir=img_dir, train=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
    
    # 3. Setup Diagnostic Lab
    wrapper = get_wrapper(model, 'HMR2')
    lab = ViTDiagnosticLab(wrapper, model_name=name, output_root='diagnostic_comparison', track_metrics=True)
    
    # 4. Run Loop
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc=f"Diag {name}", total=num_batches)):
            if i >= num_batches: break
            batch = wrapper.to_device(batch, device)
            lab.update_batch_state(batch)
            _ = model(batch)
            
    # 5. Export Results
    os.makedirs(f'diagnostic_comparison/{name}', exist_ok=True)
    results_path = f'diagnostic_comparison/{name}/layer_metrics.csv'
    lab.results_df.to_csv(results_path, index=False)
    print(f"Finished diagnostics for {name}. Results in {results_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['Control', 'Adaptive'], required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    if args.mode == 'Control':
        run_diagnostic_analysis('checkpoints/ft_Control.ckpt', 'FT_Control', args.gpu)
    else:
        run_diagnostic_analysis('checkpoints/ft_T2-KTI-Adaptive.ckpt', 'FT_Adaptive', args.gpu)
