import torch
import sys
import os
import argparse
import pandas as pd
from pathlib import Path

def main():
    print("Main Started", flush=True)
    
    script_dir = Path(__file__).resolve().parent
    sys.path.append(str(script_dir))
    sys.path.append(str(script_dir.parent))
    
    print("Importing...", flush=True)
    from run_baseline_diagnostics import load_target_model
    from diagnostic_core.diagnostic_engine import get_wrapper
    from hmr2.datasets import create_dataset
    from hmr2.configs import dataset_eval_config
    from hmr2.utils import Evaluator, recursive_to
    from tqdm import tqdm
    
    print("Imports Success", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['HMR2', 'PromptHMR', 'HSMR', 'CameraHMR'])
    parser.add_argument('--num_batches', type=int, default=10)
    args = parser.parse_args()
    
    print(f"Args: {args}", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_eval = dataset_eval_config()['3DPW-TEST']
    
    results = []
    
    for model_name in args.models:
        print(f"\nEvaluating {model_name}...", flush=True)
        model, model_cfg = load_target_model(model_name)
        if model is None: 
            print(f"Skipping {model_name}", flush=True)
            continue
        model = model.to(device)
        model.eval()
        
        wrapper = get_wrapper(model, model_name)
        dataset = create_dataset(model_cfg, cfg_eval, train=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        evaluator = Evaluator(
            dataset_length=len(dataset) if args.num_batches is None else args.num_batches * 32,
            keypoint_list=cfg_eval.KEYPOINT_LIST,
            pelvis_ind=39,
            metrics=['mode_mpjpe', 'mode_re'],
            pck_thresholds=[50, 100, 150]
        )
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=model_name)):
                if args.num_batches is not None and i >= args.num_batches: break
                batch = recursive_to(batch, device)
                if 'img' in batch and (batch['img'].shape[-1] != 256 or batch['img'].shape[-2] != 256):
                    import torch.nn.functional as F
                    batch['img'] = F.interpolate(batch['img'], size=(256, 256), mode='bilinear', align_corners=False)
                
                try:
                    out = wrapper(batch)
                    B = batch['img'].shape[0]
                    if 'pred_keypoints_3d' not in out or out['pred_keypoints_3d'] is None:
                        out['pred_keypoints_3d'] = torch.zeros(B, 44, 3, device=device)
                    if 'pred_keypoints_2d' not in out or out['pred_keypoints_2d'] is None:
                        out['pred_keypoints_2d'] = torch.zeros(B, 44, 2, device=device)
                    
                    evaluator(out, batch)
                except Exception as e:
                    print(f"Error on {model_name} batch {i}: {e}", flush=True)
                    
        res = evaluator.get_metrics_dict()
        mpjpe = res.get('mode_mpjpe', 0.0)
        pa_mpjpe = res.get('mode_re', 0.0)
        pck50 = res.get('kpAvg_pck_50', 0.0) * 100
        pck100 = res.get('kpAvg_pck_100', 0.0)
        pck150 = res.get('kpAvg_pck_150', 0.0)
        auc = ((pck50/100 + pck100)/2 * 50 + (pck100 + pck150)/2 * 50) / 150.0 * 100
        
        results.append({
            'Model': model_name,
            'MPJPE (mm)': f"{mpjpe:.1f}",
            'PA-MPJPE (mm)': f"{pa_mpjpe:.1f}",
            'PCK@50mm (%)': f"{pck50:.1f}",
            'AUC (%)': f"{auc:.1f}"
        })
        print(f"Results for {model_name}: MPJPE={mpjpe:.1f}, PA-MPJPE={pa_mpjpe:.1f}, PCK50={pck50:.1f}%, AUC={auc:.1f}%", flush=True)

    df = pd.DataFrame(results)
    out_dir = Path('/home/yangz/NViT-master/nvit/Paper1_Diagnostics/Experiment1_Entropy/results')
    df.to_csv(out_dir / "comprehensive_evaluation.csv", index=False)
    print("\n--- FINAL EVALUATION METRICS ---\n")
    print(df.to_markdown(index=False))

if __name__ == '__main__':
    main()
