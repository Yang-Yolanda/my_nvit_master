import os
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import logging

# Add paths
sys.path.insert(0, '/home/yangz/4D-Humans')
sys.path.insert(0, '/home/yangz/NViT-master/nvit/Paper1_Diagnostics')

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.datasets import HMR2DataModule
from hmr2.configs import dataset_config
from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='T2-KTI-Adaptive', help='Intervention group name')
    parser.add_argument('--steps', type=int, default=5000, help='Number of fine-tuning steps')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model
    model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model.to(device)
    
    # 2. Setup Lab & Apply Masking
    wrapper = get_wrapper(model, 'HMR2')
    lab = ViTDiagnosticLab(wrapper, model_name=f"FT_{args.group}", output_root='ft_results')
    
    if not lab.apply_single_intervention(args.group):
        logger.error(f"Group {args.group} not found!")
        return

    # 3. Setup Dataset
    # Use standard training mixed dataset
    ds_cfg = dataset_config('datasets_tar.yaml')
    # Update config for small-scale sweep
    cfg.defrost()
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.LR = args.lr
    from yacs.config import CfgNode as CN
    train_ds = CN()
    train_ds['H36M-TRAIN-WMASK'] = CN({'WEIGHT': 1.0})
    train_ds['MPII-TRAIN-WMASK'] = CN({'WEIGHT': 1.0})
    train_ds['COCO-TRAIN-2014-WMASK-PRUNED'] = CN({'WEIGHT': 1.0})
    train_ds['MPI-INF-TRAIN-PRUNED'] = CN({'WEIGHT': 1.0})
    # If TRAIN exists but restricts type, bypass by reassigning DATASETS or popping
    if 'TRAIN' in cfg.DATASETS:
        # We can't always just delete from CfgNode if it's schema-locked without new_allowed.
        # But HMR2 uses new_allowed=True. 
        _ = cfg.DATASETS.pop('TRAIN')
    cfg.DATASETS.TRAIN = train_ds
    cfg.freeze()


        
    datamodule = HMR2DataModule(cfg, ds_cfg)
    datamodule.setup()
    loaders = datamodule.train_dataloader()
    train_loader = loaders['img'] if isinstance(loaders, dict) else loaders


    # 4. Simple Fine-tuning Loop (avoiding full PL overhead for quick sweep)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    
    logger.info(f"Starting Fine-tuning for {args.group} on GPU {args.gpu}")
    
    global_step = 0
    pbar = tqdm(total=args.steps, desc=f"FT: {args.group}")
    
    while global_step < args.steps:
        for batch in train_loader:
            if global_step >= args.steps: break
            
            batch = wrapper.to_device(batch, device)
            optimizer.zero_grad()
            
            # Diagnostic engine patches block.forward, so we just run standard model forward
            out = model(batch)
            
            # Loss computation
            # We use the built-in compute_loss if available, or a simple one
            # HMR2 uses a complex loss, let's call it via wrapper if possible
            loss = model.compute_loss(batch, out, train=True)
            
            loss.backward()
            optimizer.step()
            
            global_step += 1
            pbar.update(1)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if global_step % 500 == 0:
                logger.info(f"Step {global_step}/{args.steps} - Loss: {loss.item():.4f}")
                
    pbar.close()
    
    # 5. Save Checkpoint
    save_path = Path(f'checkpoints/ft_{args.group}.ckpt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved checkpoint to {save_path}")

if __name__ == '__main__':
    from tqdm import tqdm
    main()
