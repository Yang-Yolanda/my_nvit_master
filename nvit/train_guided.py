#!/usr/bin/env python
from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
import sys
from nvit.utils.path_utils import get_humans_root, get_project_root
from pathlib import Path

# Add 4D-Humans and NViT-master to path
sys.path.insert(0, str(get_humans_root()))
sys.path.insert(0, str(get_project_root()))

import hydra
import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# [SPEED FIX] Enable dynamic cuDNN kernel optimization
torch.backends.cudnn.benchmark = True

# [DeepSeek Trick]: Bypass `/dev/shm` RAM limit for DDP DataLoader tensors 
# by using the actual physical file system, drastically lowering CPU Host RAM usage for large batches.
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# [DeepSeek Trick]: Limit OpenCV and OpenMP CPU thread allocation per DataLoader worker 
# preventing exponential memory explosion in Thread Stacks during DDP.
import cv2
cv2.setNumThreads(0)
os.environ['OMP_NUM_THREADS'] = '1'

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from yacs.config import CfgNode
from hmr2.configs import dataset_config
from hmr2.datasets import HMR2DataModule
# Using Guided Module instead of standard HMR2
from nvit2_models.guided_hmr2 import GuidedHMR2Module
from hmr2.utils.pylogger import get_pylogger
from hmr2.utils.misc import task_wrapper, log_hyperparameters

# [NEW] Import BioMambaDataset for Robust Sanity Check
from nvit.bio_dataset import BioMambaDataset
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from nvit.masking_utils import MaskingPatcher
from pathlib import Path

import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)

log = get_pylogger(__name__)

import time
import psutil
class SystemHealthMonitor(pl.Callback):
    def __init__(self, log_interval=30):
        super().__init__()
        self.log_interval = log_interval
        self.last_log_time = time.time()

    @pl.utilities.rank_zero.rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if time.time() - self.last_log_time >= self.log_interval:
            # System Metrics
            cpu_pct = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            
            # GPU metrics
            gpu_str = ""
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        free, total = torch.cuda.mem_get_info(i)
                        used = total - free
                        util = used / total * 100
                        gpu_str += f" GPU{i}:{util:.1f}%({used/1024**3:.1f}GB)"
                    except:
                        pass

            # [NEW] DataLoader Diagnostic Header when RAM is high
            diag_str = ""
            if mem.percent > 90:
                try:
                    dl = trainer.train_dataloader
                    # Extract workers and batch size even if wrapped
                    while hasattr(dl, 'loader'): dl = dl.loader
                    diag_str = f" | ⚠️ [OOM-Risk] Workers:{getattr(dl, 'num_workers', 'N/A')} B:{getattr(dl, 'batch_size', 'N/A')}"
                except:
                    pass

            log.info(f"❤️ [Health] Step:{trainer.global_step} | Host CPU:{cpu_pct}% | Host RAM:{mem.percent}% ({mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB){diag_str} |{gpu_str}")
            
            if mem.percent > 95:
                log.error("🛑 CRITICAL Host RAM usage (>95%)! Watchdog may kill process soon.")
            
            self.last_log_time = time.time()

class GuidedDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Project-relative paths
        project_root = pyrootutils.find_root()
        self.dataset_file = str(project_root.parent / '4D-Humans' / 'data/metadata/3dpw_test.npz')
        self.img_dir = str(project_root.parent / '4D-Humans' / 'data/3DPW')
        
    def setup(self, stage=None):
        # Load Model Config for Dataset preprocessing
        _, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        
        self.train_ds = BioMambaDataset(m_cfg, dataset_file=self.dataset_file, img_dir=self.img_dir, train=True)
        # Use same for val/test in sanity check
        self.val_ds = BioMambaDataset(m_cfg, dataset_file=self.dataset_file, img_dir=self.img_dir, train=False)

        # [ULTIMATE RAM FIX]: Freeze the Garbage Collector!
        # This completely prevents Linux 'Copy-on-Write' from duplicating 
        # the entire dataset object 80 times across all DataLoader workers!
        import gc
        gc.freeze()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, 
            batch_size=self.cfg.TRAIN.BATCH_SIZE, 
            shuffle=True, 
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            prefetch_factor=getattr(self.cfg.GENERAL, 'PREFETCH_FACTOR', 2),
            persistent_workers=True, # [GPU LIMIT FIX]: Keep workers alive to avoid CPU respawn starvation
            pin_memory=True # [GPU LIMIT FIX]: Essential for 100% GPU utilization (Async PCIe DMA)
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, 
            batch_size=self.cfg.TRAIN.BATCH_SIZE, 
            shuffle=False, 
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            persistent_workers=True,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=self.cfg.GENERAL.NUM_WORKERS)

@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    """Save config files to rootdir."""
    try:
        Path(rootdir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'))
    except OSError as e:
        log.warning(f"Failed to save configs to {rootdir} (possibly OSS mount conflict): {e}")
    # Dataset cfg is now None or simplified
    # with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
    #    f.write(dataset_cfg.dump())

@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

    # [Hardened] Undermind Rule 1.1: Set seed for full reproducibility
    seed = cfg.get('seed', 1234)
    pl.seed_everything(seed, workers=True)

    # Load dataset config
    # [NEW] Allow overriding the dataset config file via Hydra (e.g. data.config_file)
    # Default is 'datasets_tar.yaml'
    ds_conf_name = cfg.get('DATASETS_CONFIG_FILE', 'datasets_tar.yaml')
    dataset_cfg = dataset_config(ds_conf_name)

    # Save configs
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    # Setup training and validation datasets
    # Setup training and validation datasets
    # [Production] Use Standard HMR2 DataModule
    # Ensure 'hmr2_training_data/cmu_mocap.npz' and 'vitpose_backbone.pth' are available or paths configured.
    datamodule = HMR2DataModule(cfg, dataset_cfg)

    # [Debug/Pilot] Use GuidedDataModule for Sanity Check on subset (Pilot Training)
    # datamodule = GuidedDataModule(cfg)
    
    # [Fix] Patch missing GENERAL keys that cause InterpolationError
    if 'GENERAL' not in cfg:
        cfg.GENERAL = CfgNode()
    if 'LOG_STEPS' not in cfg.GENERAL:
        cfg.GENERAL.LOG_STEPS = 10
    if 'CHECKPOINT_STEPS' not in cfg.GENERAL:
        cfg.GENERAL.CHECKPOINT_STEPS = 1000
    if 'CHECKPOINT_SAVE_TOP_K' not in cfg.GENERAL:
        cfg.GENERAL.CHECKPOINT_SAVE_TOP_K = 1
        
    # [Fix] Override trainer log steps to static value to avoid broken interpolation
    cfg.trainer.log_every_n_steps = cfg.GENERAL.LOG_STEPS

    # [Case-Tolerant Config Helper]
    # Works for both Hydra (model.backbone) and YACS/Manual edits (MODEL.BACKBONE)
    def get_sub_cfg(cfg, keys):
        for k in keys:
            if k in cfg and cfg[k] is not None:
                return cfg[k]
        return None

    model_cfg = get_sub_cfg(cfg, ['model', 'MODEL'])
    if model_cfg is None:
        log.warning("⚠️ Model configuration not found in cfg. Using defaults.")
        # Create a dummy one to avoid crashes, though this shouldn't happen with Hydra
        from omegaconf import DictConfig
        model_cfg = DictConfig({'backbone': {}, 'smpl_head': {}})

    backbone_cfg = get_sub_cfg(model_cfg, ['backbone', 'BACKBONE'])
    smpl_head_cfg = get_sub_cfg(model_cfg, ['smpl_head', 'SMPL_HEAD'])

    # Setup model (Using Guided HMR2)
    model = GuidedHMR2Module(cfg)
    
    # [NEW: KTI-Guided Surgical Freezing]
    freeze_depth = cfg.get('FREEZE_DEPTH', 0)
    if freeze_depth > 0:
        log.info(f"Surgically freezing first {freeze_depth} layers of backbone (ViT stage)...")
        # Find the backbone container
        bb = getattr(model, 'nvit_backbone', getattr(model, 'backbone', None))
        
        if bb is not None:
            if hasattr(bb, 'surgical_freeze'):
                # Use class-specific fine-grained freezing first
                bb.surgical_freeze(freeze_depth=freeze_depth)
            elif hasattr(bb, 'blocks'):
                # Universal fallback: traverse standard transformer blocks
                log.info(f"Applying universal block-level freeze to {type(bb).__name__}")
                # 1. Freeze embeddings to maintain pre-trained feature distribution
                for p_name, p in bb.named_parameters():
                    if any(x in p_name for x in ['patch_embed', 'pos_embed', 'cls_token']):
                        p.requires_grad = False
                # 2. Freeze blocks up to specified depth
                for i, blk in enumerate(bb.blocks):
                    if i < freeze_depth:
                        for p in blk.parameters(): p.requires_grad = False
                        log.info(f"❄️  [Manual] Layer {i} FROZEN")
            else:
                log.warning(f"Backbone found but has no '.blocks' or 'surgical_freeze'. Unsafe to freeze depth {freeze_depth}.")
        else:
            log.error("CRITICAL: No backbone found in model. Freezing failed.")

    # [NEW: Trainable Parameter Summary]
    # Moved AFTER freezing/masking to accurately reflect training scale.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model Parameters: Total={total_params:,} | Trainable={trainable_params:,} ({trainable_params/total_params:.1%})")

    # [NEW: Attention Masking (Paper 1 Baselines)]
    mask_config = cfg.get('MASK_CONFIG', None)
    if mask_config is not None:
        log.info(f"Applying Attention Masking (Mode: {mask_config.mode}) to backbone...")
        # MaskingPatcher expects model.backbone.blocks or specifically handled
        patcher = MaskingPatcher(model, mask_config)
        # Handle nvit_backbone naming in GuidedHMR2Module
        if hasattr(model, 'nvit_backbone'):
             patcher.att_modules = []
             for i, blk in enumerate(model.nvit_backbone.blocks):
                 if hasattr(blk, 'attn'):
                     patcher.att_modules.append((i, blk.attn))
                 elif hasattr(blk, 'block') and hasattr(blk.block, 'attn'):
                     patcher.att_modules.append((i, blk.block.attn))
        patcher.apply()
        model.mask_patcher = patcher

    # Setup Tensorboard logger
    logger = TensorBoardLogger(os.path.join(cfg.paths.output_dir, 'tensorboard'), name='', version='', default_hp_metric=False)
    loggers = [logger]

    # Setup Checkpoint Wrapper to allow multiple ModelCheckpoint instances
    class UniqueModelCheckpoint(ModelCheckpoint):
        def __init__(self, *args, **kwargs):
            self._state_key = kwargs.pop('state_key', None)
            super().__init__(*args, **kwargs)
        @property
        def state_key(self) -> str:
            return self._state_key if self._state_key else super().state_key

    # Setup checkpoint saving
    # [Optimized Checkpointing Strategy]
    # 1. Weights-only snapshot for every epoch - for archive/evaluation (approx 1.3GB)
    weights_only_callback = UniqueModelCheckpoint(
        state_key='weights',
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'), 
        save_weights_only=True,
        save_top_k=-1,            # Save every epoch
        every_n_epochs=1,
        filename='epoch_{epoch:02d}', 
        monitor=None,
    )
    
    # 2. Master Resumption Node - contains Optimizer States for "Perfect Resume" (approx 2.5GB+)
    # Updates 'last.ckpt' at the end of every epoch.
    last_state_callback = UniqueModelCheckpoint(
        state_key='last',
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'), 
        save_last=True,
        save_top_k=0,             # Only keep 'last.ckpt'
        every_n_epochs=1,
        monitor=None,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    health_monitor = SystemHealthMonitor(log_interval=30)
    
    callbacks = [
        weights_only_callback, 
        last_state_callback,
        lr_monitor,
        health_monitor,
    ]
    
    # Convert DictConfig to dict to allow popping
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    if 'strategy' in trainer_cfg:
        trainer_cfg.pop('strategy')
        
    # [Fix] Absolutely Disable Validation according to user request
    trainer_cfg['limit_val_batches'] = 0.0
    trainer_cfg['check_val_every_n_epoch'] = None
    trainer_cfg['num_sanity_val_steps'] = 0
    
    # [Fix] Enforce an Epoch boundary for Infinite DataLoaders
    # The WebDataset inherently amplifies its with_epoch() counter by num_workers and num_nodes.
    # We must enforce a hard limit on batches per epoch so epochs remain a reasonable conceptual unit.
    if 'limit_train_batches' not in trainer_cfg and 'CHECKPOINT_STEPS' in cfg.GENERAL:
        trainer_cfg['limit_train_batches'] = cfg.GENERAL.CHECKPOINT_STEPS
        
    # [Autonomous Mode] Dynamic Device Configuration
    if 'devices' not in trainer_cfg:
        trainer_cfg['devices'] = 1
        
    log.info(f"Instantiating trainer <{cfg.trainer._target_}> with {trainer_cfg['devices']} devices")
    
    # Determine Strategy override
    strategy_kwargs = {}
    # Single-GPU Mode: Remove any DDP strategy to prevent multi-process spawning
    if 'strategy' in trainer_cfg:
        trainer_cfg.pop('strategy')

    trainer: Trainer = hydra.utils.instantiate(
        trainer_cfg, 
        callbacks=callbacks, 
        logger=loggers, 
        **strategy_kwargs,
        plugins=(SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get('launcher',None) is not None) else None),
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # [Finetuning] Load weights if specified
    if 'FINETUNE_FROM' in cfg and cfg.FINETUNE_FROM is not None:
        log.info(f"Finetuning from checkpoint: {cfg.FINETUNE_FROM}")
        ckpt = torch.load(cfg.FINETUNE_FROM, map_location='cpu')
        state_dict = ckpt['state_dict']
        
        # Filter state_dict to handle size mismatches and name changes
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        # [Crucial Fix] Enhanced Weight Mapping for Hybrid Architecture
        # Account for '.block' nesting introduced by ViTBlock/Mamba wrappers in AdaptiveNViT
        is_hybrid = hasattr(model, 'nvit_backbone')
        for k, v in state_dict.items():
            # 1. Base mapping: backbone -> nvit_backbone (Only for Hybrid models)
            k_new = k
            if is_hybrid and k.startswith('backbone.'):
                k_new = k.replace('backbone.', 'nvit_backbone.', 1)
            
                # 2. Block-level nesting fix
                # Converts 'nvit_backbone.blocks.0.norm1' to 'nvit_backbone.blocks.0.block.norm1'
                if 'blocks.' in k_new and '.block.' not in k_new:
                    parts = k_new.split('.')
                    try:
                        idx = parts.index('blocks')
                        # If we have blocks.N.item, insert 'block' after N
                        if idx + 1 < len(parts) and parts[idx+1].isdigit():
                            parts.insert(idx + 2, 'block')
                            k_new = '.'.join(parts)
                    except (ValueError, IndexError):
                        pass
            
            # 3. Shape validation and filtering
            if k_new in model_state_dict:
                if v.shape == model_state_dict[k_new].shape:
                    filtered_state_dict[k_new] = v
                else:
                    log.warning(f"Shape mismatch for {k_new}: CKPT {v.shape} vs MODEL {model_state_dict[k_new].shape}. Skipping.")
            else:
                log.debug(f"Skipping {k} (not in model)")
        
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        total_model_keys = len(model.state_dict())
        matched_keys = total_model_keys - len(missing)
        match_rate = matched_keys / total_model_keys if total_model_keys > 0 else 0
        log.info(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}, Match Rate: {match_rate:.2%}")
        if match_rate < 0.20:
            raise RuntimeError(f"Weight mapping failed: {match_rate:.2%} Match Rate is below 20% threshold! Missing: {len(missing)}.")

    # [Fix] Configure ckpt_path for "Perfect Resumption" (断点重训)
    # Priority: 1. Command line explicit path | 2. Auto-detected last.ckpt | 3. Fresh start
    explicit_ckpt = cfg.get('ckpt_path', None)
    if explicit_ckpt is not None and explicit_ckpt != 'null':
        log.info(f"Using explicitly provided checkpoint for resumption: {explicit_ckpt}")
        ckpt_path_to_use = explicit_ckpt
    else:
        # Auto-detect last.ckpt in the output directory
        auto_last = os.path.join(cfg.paths.output_dir, 'checkpoints', 'last.ckpt')
        if os.path.isfile(auto_last):
            log.info(f"✨ Auto-resuming from Master Resumption Node: {auto_last}")
            ckpt_path_to_use = auto_last
        else:
            log.info("No matching 'last.ckpt' found. Starting fresh run.")
            ckpt_path_to_use = None
    
    log.info(f"Trainer Max Steps: {trainer.max_steps}")
    log.info(f"Using ckpt_path: {ckpt_path_to_use}")

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path_to_use)
    log.info("Fitting done")


@hydra.main(version_base="1.2", config_path="../../4D-Humans/hmr2/configs", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    # [Fix] Patch missing GENERAL keys that cause InterpolationError
    # Must be done HERE before @task_wrapper (extras) touches the config
    with open_dict(cfg):
        if 'GENERAL' not in cfg:
            cfg.GENERAL = DictConfig({}) # Use DictConfig instead of CfgNode
        if 'LOG_STEPS' not in cfg.GENERAL:
            cfg.GENERAL.LOG_STEPS = 10
        if 'VAL_STEPS' not in cfg.GENERAL:
            cfg.GENERAL.VAL_STEPS = 100
        if 'CHECKPOINT_STEPS' not in cfg.GENERAL:
            cfg.GENERAL.CHECKPOINT_STEPS = 1000
        if 'CHECKPOINT_SAVE_TOP_K' not in cfg.GENERAL:
            cfg.GENERAL.CHECKPOINT_SAVE_TOP_K = 1
            
        # [Fix] Disable config printing to avoid resolution errors in extras
        if 'extras' in cfg:
            cfg.extras.print_config = False
            
        # [Fix] Override trainer log steps to static value
            
        # [Fix] Override trainer log steps to static value
        if 'trainer' in cfg:
            cfg.trainer.log_every_n_steps = 10
            
    # Create output directory early to avoid issues with tags.log etc.
    if 'paths' in cfg and 'output_dir' in cfg.paths:
        Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

    # [Optimization] Enable Tensor Cores
    torch.set_float32_matmul_precision('medium')

    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
