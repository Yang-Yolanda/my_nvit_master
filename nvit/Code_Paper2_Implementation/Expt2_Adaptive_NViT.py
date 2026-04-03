#!/home/yangz/.conda/envs/4D-humans/bin/python

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import sys
import os
import logging
from pathlib import Path

# Setup Paths
NVIT_ROOT = '/home/yangz/NViT-master/nvit'
HUMANS_DIR = '/home/yangz/4D-Humans'
sys.path.append(NVIT_ROOT)
sys.path.append(HUMANS_DIR)
# Also append current dir for models import
sys.path.append(os.getcwd())

# Imports
from hmr2.models import HMR2, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from datasets_3dpw import create_dataset
from nvit2_models.nvit_hybrid import AdaptiveNViT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# Adaptive Lightning Module
# ==============================================================================

class AdaptiveHMR2(HMR2):
    def __init__(self, cfg, init_renderer=False, switch_layers=(2, 8)):
        # Init Parent (HMR2)
        # Pass init_renderer=False to avoid headless rendering issues initially
        super().__init__(cfg=cfg, init_renderer=init_renderer)
        
        # Override Backbone
        logger.info(f"Initializing AdaptiveNViT Backbone with switches: {switch_layers}")
        self.backbone = AdaptiveNViT(
            depth=32,
            embed_dim=1280,
            num_heads=16,
            switch_layer_1=switch_layers[0],
            switch_layer_2=switch_layers[1],
            img_size=(256, 256),
            patch_size=16
        )
        
        # Renderer Handling
        # If init_renderer is False, self.renderer and self.mesh_renderer are None.
        # This is handled in tensorboard_logging override.

    def tensorboard_logging(self, batch, output, step_count, train=True, write_to_summary_writer=True):
        # Override to prevent crash if renderer is None
        if self.renderer is None and self.mesh_renderer is None:
            # Just log scalars (handled by parent? No, logic is mixed in parent method)
            # We will copy scalar logging part and skip image rendering
            mode = 'train' if train else 'val'
            losses = output['losses']
            if write_to_summary_writer:
                summary_writer = self.logger.experiment
                for loss_name, val in losses.items():
                    summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
            return
        
        # If renderer exists, delegate to parent (or copy logic if super call is tricky due to decorators)
        super().tensorboard_logging(batch, output, step_count, train, write_to_summary_writer)

    def configure_optimizers(self):
        # HMR2 always returns (opt_g, opt_d). 
        # But if we disable Adv, training_step fails to unpack.
        # We must fix this here.
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL <= 0:
            param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]
            optimizer = torch.optim.AdamW(params=param_groups, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
            return optimizer
        else:
            return super().configure_optimizers()

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    logger.info("Starting Expt2: Adaptive NViT Training Loop Check...")
    
    # 1. Load Reference Config & Weights
    logger.info("Loading HMR2 Reference...")
    ref_model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    
    # Check what switch points we want (Based on Expt1)
    # Expt1 said: Switch 1 @ L0 (Max) -> ViT L0..1? 
    # Decision: ViT L0-L1 (2 layers), Mamba L2-L7 (6 layers), GCN L8+ (4 layers)
    SWITCH_LAYERS = (2, 8) 
    
    # 2. Init Adaptive Model
    logger.info("Instantiating AdaptiveHMR2...")
    model = AdaptiveHMR2(model_cfg, init_renderer=False, switch_layers=SWITCH_LAYERS)
    
    # 3. Transfer Weights (Surgical)
    logger.info("Transplanting Weights...")
    ref_state = ref_model.state_dict()
    model_state = model.state_dict()
    
    transferred = 0
    skipped = 0
    
    for name, param in ref_state.items():
        # Handle Backbone separately
        if 'backbone' in name:
            # Check if compatible
            # Blocks: backbone.blocks.0. ...
            if name in model_state and model_state[name].shape == param.shape:
                 model_state[name].copy_(param)
                 transferred += 1
            else:
                 # Mismatched architecture (Mamba/GCN vs ViT)
                 # Or mismatched parameter names
                 # e.g. backbone.blocks.2.mamba... vs backbone.blocks.2.attn...
                 skipped += 1
        else:
            # Heads, SMPL, etc. -> Copy directly
            if name in model_state and model_state[name].shape == param.shape:
                model_state[name].copy_(param)
                transferred += 1
            else:
                logger.warning(f"Skipping head param mismatch: {name}")
                
    logger.info(f"Weights Transferred: {transferred}, Skipped: {skipped} (Expected due to Hybrid Arch)")
    
    model.load_state_dict(model_state)
    
    # 4. Data Setup (Use 3DPW Test for validation loop)
    class Args:
        batch_size = 4
        num_workers = 2
        pin_mem = True
        data_path = '/home/yangz/4D-Humans/data' 
        
    dataset = create_dataset(Args(), split='test')
    # Use a small subset to ensure quick loop
    subset_dataset = torch.utils.data.Subset(dataset, range(20)) 
    dataloader = torch.utils.data.DataLoader(subset_dataset, batch_size=4, shuffle=True)
    
    # 5. dummy Mocap Batch (HMR2 training step expects 'mocap' in 'joint_batch')
    # HMR2 dataset loaders usually return mixed batches. 
    # 3DPW dataset returns dicts.
    # Lightning training_step signature: (joint_batch, batch_idx)
    # joint_batch['img'] -> Image Batch
    # joint_batch['mocap'] -> Mocap Batch (for Discriminator)
    # If we don't have Mocap data in this loader, we might crash if Adversarial Loss > 0.
    
    # Quick fix: Disable Adversarial Loss in CFG
    model.cfg.defrost()
    model.cfg.LOSS_WEIGHTS.ADVERSARIAL = 0.0
    logger.info("Disabled Adversarial Loss for Diagnostic Run.")
    
    # Also, hmr2.py training_step unpacks: batch = joint_batch['img']
    # If our loader yields 'batch' directly, we need to wrap it.
    
    class WrappedLoader:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                # Wrap as 'img' key
                yield {'img': batch, 'mocap': None}
        def __len__(self):
            return len(self.loader)
            
    train_loader = WrappedLoader(dataloader)

    # 6. Run Trainer
    logger.info("Creating Lightning Trainer...")
    trainer = Trainer(
        max_epochs=1,
        limit_train_batches=5, # Only run 5 batches to verify
        limit_val_batches=0,
        accelerator='auto',
        devices=1,
        logger=False, # Disable file logging for speed
        enable_checkpointing=False,
    )
    
    logger.info("Fitting Model...")
    try:
        trainer.fit(model, train_dataloaders=train_loader)
        logger.info("✅ Training Loop Completed Successfully!")
        with open("Expt2_Adaptive_NViT.log", "w") as f:
            f.write("Status: Success\n")
            f.write("Message: AdaptiveNViT forward/backward pass verified.\n")
            
    except Exception as e:
        logger.error(f"❌ Training Loop Failed: {e}")
        with open("Expt2_Adaptive_NViT.log", "w") as f:
            f.write("Status: Failed\n")
            f.write(f"Error: {e}\n")
        raise e

if __name__ == "__main__":
    main()
