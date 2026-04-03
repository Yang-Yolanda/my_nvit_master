#!/home/yangz/.conda/envs/4D-humans/bin/python
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
import json
import argparse
import numpy as np

# Add HMR2 to path
sys.path.append('/home/yangz/4D-Humans')
from hmr2.configs import dataset_eval_config
from hmr2.datasets import create_dataset
from hmr2.utils import Evaluator, recursive_to
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='original')
    parser.add_argument('--dataset', type=str, default='H36M-VAL-P2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda')

    # 1. Load Baseline to get Config
    print('Loading baseline config...')
    _, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

    # 2. Load Target Model
    if args.mode == 'original':
        from model_manager import ModelManager
        mm = ModelManager({'device': 'cuda', 'enable_detector': False})
        mm.load_model()
        model = mm.model
    else:
        from experiment_layer_ablation import LayerAblationPruner
        config = {
            'device': 'cuda',
            'enable_detector': False,
            'ablation_mode': args.mode,
            'heavy_sparsity': 0.7
        }
        pruner = LayerAblationPruner(config)
        model = pruner.prune()

    model = model.to(device).eval()

    # 3. Setup Dataset
    dataset_cfg = dataset_eval_config()[args.dataset]
    dataset = create_dataset(model_cfg, dataset_cfg, train=False)
    


    if args.limit > 0:
        indices = torch.arange(min(args.limit, len(dataset)))
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 4. Evaluation Loop
    evaluator = Evaluator(
        dataset_length=len(dataset),
        keypoint_list=dataset_cfg.KEYPOINT_LIST,
        pelvis_ind=model_cfg.EXTRA.PELVIS_IND,
        metrics=['mode_mpjpe', 'mode_re']
    )

    print(f'Evaluating {args.mode} on {args.dataset} (Limit: {args.limit})...')
    for i, batch in enumerate(tqdm(dataloader)):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        
        # Use correct key 'pred_keypoints_3d' instead of 'joints'
        # Use __call__ to accumulate metrics
        evaluator(out, batch)
        # evaluator.log() # Optional: print current progress

    metrics = evaluator.get_metrics_dict()
    print(f'\nResults for [{args.mode}]:')
    print(json.dumps(metrics, indent=4))

    # Save results
    out_file = f'eval_res_{args.mode}_{args.dataset.lower()}.json'
    with open(out_file, 'w') as f:
        json.dump(metrics, f)
    print(f'Results saved to {out_file}')

if __name__ == "__main__":
    main()
