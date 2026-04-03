#!/home/yangz/.conda/envs/4D-humans/bin/python
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
from pathlib import Path

# Add 4D-Humans to path to find hmr2
sys.path.insert(0, '/home/yangz/4D-Humans')
sys.path.insert(0, '/home/yangz/NViT-master')

import hydra
import torch
import pytorch_lightning as pl
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
from bio_dataset import BioMambaDataset
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from nvit.masking_utils import MaskingPatcher

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

            # [Fix] Add True Proc RSS to differentiate from Page Cache
            proc = psutil.Process(os.getpid())
            rss_gb = proc.memory_info().rss / 1024**3
            
            log.info(f"❤️ [Health] Step:{trainer.global_step} | Host CPU:{cpu_pct}% | Host RAM:{mem.percent}% ({mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB) | Proc RSS:{rss_gb:.1f}GB {diag_str} |{gpu_str}")
            
            if mem.percent > 95:
                log.error("🛑 CRITICAL Host RAM usage (>95%)! Watchdog may kill process soon.")
            
            self.last_log_time = time.time()

class GuidedDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Hardcode verify paths (Production should use cfg)
        self.dataset_file = '/home/yangz/4D-Humans/data/metadata/3dpw_test.npz'
        self.img_dir = '/home/yangz/4D-Humans/data/3DPW'
        
    def setup(self, stage=None):
        # Load Model Config for Dataset preprocessing
        _, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        
        self.train_ds = BioMambaDataset(m_cfg, dataset_file=self.dataset_file, img_dir=self.img_dir, train=True)
        # Use same for val/test in sanity check
        self.val_ds = BioMambaDataset(m_cfg, dataset_file=self.dataset_file, img_dir=self.img_dir, train=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, 
            batch_size=self.cfg.TRAIN.BATCH_SIZE, 
            shuffle=True, 
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            prefetch_factor=getattr(self.cfg.GENERAL, 'PREFETCH_FACTOR', 2),
            persistent_workers=False, # Defense against RAM leak
            pin_memory=False # Defense against SHM/RAM overhead
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, 
            batch_size=self.cfg.TRAIN.BATCH_SIZE, 
            shuffle=False, 
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            persistent_workers=False
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=self.cfg.GENERAL.NUM_WORKERS)

@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    """Save config files to rootdir."""
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'))
    # Dataset cfg is now None or simplified
    # with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
    #    f.write(dataset_cfg.dump())

@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

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

    # Setup model (Using Guided HMR2)
    model = GuidedHMR2Module(cfg)
    
    # [NEW: Trainable Parameter Summary]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model Parameters: Total={total_params:,} | Trainable={trainable_params:,} ({trainable_params/total_params:.1%})")
    
    # [NEW: KTI-Guided Surgical Freezing]
    freeze_depth = cfg.get('FREEZE_DEPTH', 8)
    if freeze_depth > 0:
        log.info(f"Surgically freezing first {freeze_depth} layers of backbone (ViT stage)...")
        if hasattr(model, 'nvit_backbone') and hasattr(model.nvit_backbone, 'surgical_freeze'):
            model.nvit_backbone.surgical_freeze(freeze_depth=freeze_depth)
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'surgical_freeze'):
            model.backbone.surgical_freeze(freeze_depth=freeze_depth)
        else:
            log.warning("Could not find surgical_freeze method on backbone!")

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

    # Setup checkpoint saving
    # Monitor val/loss to save the best model (best.ckpt)
    # save_top_k=1 will keep the single best model based on min val/loss
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'), 
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS, 
        save_last=True,
        monitor='val/loss', # Monitors validation loss for model saving
        mode='min',
        save_top_k=1, 
        filename='best'
    )
    
    # [NEW] Setup Early Stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val/loss',
        min_delta=0.00,
        patience=10, # Stop after 10 validation cycles without improvement
        verbose=True,
        mode='min'
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    health_monitor = SystemHealthMonitor(log_interval=30)
    callbacks = [
        checkpoint_callback, 
        early_stop_callback,
        lr_monitor,
        health_monitor,
    ]
    # [Fix] Force DDP strategy with unused params by overriding Hydra config
    # Convert DictConfig to dict to allow popping
    from pytorch_lightning.strategies import DDPStrategy

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)

    # devices 默认值
    if 'devices' not in trainer_cfg or trainer_cfg['devices'] is None:
        trainer_cfg['devices'] = 1

    devices = int(trainer_cfg['devices'])
    trainer_cfg['accelerator'] = trainer_cfg.get('accelerator', 'auto')
    trainer_cfg['num_nodes'] = int(trainer_cfg.get('num_nodes', 1))

    # WebDataset 通常是 IterableDataset；建议禁用 distributed sampler，靠 wds.split_by_node/worker 来切分
    trainer_cfg['use_distributed_sampler'] = False
    
    # [NEW] Enable Periodic Validation
    trainer_cfg['val_check_interval'] = cfg.GENERAL.CHECKPOINT_STEPS
    trainer_cfg['limit_val_batches'] = 50 # Only run 50 batches for validation to save time

    strategy_kwargs = {}

    if devices <= 1:
        # 单卡：明确不要 strategy，避免多进程
        trainer_cfg.pop('strategy', None)
        log.info("Single-GPU mode: strategy disabled.")
    else:
        # 多卡：启用真正 DDP（NCCL）并允许 unused params
        trainer_cfg.pop('strategy', None)  # 避免 Hydra 配置冲突
        strategy_kwargs["strategy"] = DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=False,
        )
        log.info(f"DDP mode: devices={devices}, find_unused_parameters=True, backend=nccl")

    log.info(f"Instantiating trainer <{cfg.trainer._target_}> with devices={devices}")

    trainer: Trainer = hydra.utils.instantiate(
        trainer_cfg,
        callbacks=callbacks,
        logger=loggers,
        **strategy_kwargs,
        plugins=None,  # 裸机不需要 SLURMEnvironment
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
    finetune_from = cfg.get('FINETUNE_FROM', '/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt')
    if finetune_from is not None:
        log.info(f"Finetuning from checkpoint: {finetune_from}")
        ckpt = torch.load(finetune_from, map_location='cpu')
        state_dict = ckpt['state_dict']
        
        # Filter state_dict to handle size mismatches and name changes
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            # Handle backbone -> nvit_backbone mapping
            k_new = k
            if k.startswith('backbone.'):
                k_new = k.replace('backbone.', 'nvit_backbone.', 1)
            
            # [Fix] Handle AdaptiveNViT's ViTBlock wrapping which adds an extra `.block.`
            if 'blocks.' in k_new:
                parts = k_new.split('.')
                try:
                    b_idx = parts.index('blocks')
                    if b_idx + 1 < len(parts) and parts[b_idx + 1].isdigit():
                        idx = parts[b_idx + 1]
                        test_key = k_new.replace(f'blocks.{idx}.', f'blocks.{idx}.block.', 1)
                        if test_key in model_state_dict:
                            k_new = test_key
                except ValueError:
                    pass
            
            if k_new in model_state_dict:
                if v.shape == model_state_dict[k_new].shape:
                    filtered_state_dict[k_new] = v
                else:
                    log.warning(f"Skipping {k_new} due to size mismatch: {v.shape} vs {model_state_dict[k_new].shape}")
            else:
                log.debug(f"Skipping {k} (not in model)")
        
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        total_model_keys = len(model.state_dict())
        matched_keys = total_model_keys - len(missing)
        match_rate = matched_keys / total_model_keys if total_model_keys > 0 else 0
        log.info(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}, Match Rate: {match_rate:.2%}")
        # Threshold removed to allow Topology Ablations where massive chunks of layers naturally mismatch

    # [Fix] Configure ckpt_path.
    # - If cfg.ckpt_path is explicitly set (e.g. via run_full_ddp.sh --resume), use it.
    # - If not set, auto-detect last.ckpt in checkpoint directory.
    # - If nothing found, start fresh (ckpt_path=None).
    explicit_ckpt = cfg.get('ckpt_path', None)
    if explicit_ckpt is not None and explicit_ckpt != 'null':
        ckpt_path_to_use = explicit_ckpt
    else:
        # Auto-detect last.ckpt
        auto_last = os.path.join(cfg.paths.output_dir, 'checkpoints', 'last.ckpt')
        if os.path.isfile(auto_last):
            log.info(f"Auto-resuming from last checkpoint: {auto_last}")
            ckpt_path_to_use = auto_last
        else:
            log.info("No checkpoint found. Starting fresh run.")
            ckpt_path_to_use = None
    
    log.info(f"Trainer Max Steps: {trainer.max_steps}")
    log.info(f"Using ckpt_path: {ckpt_path_to_use}")

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path_to_use)
    log.info("Fitting done")


@hydra.main(version_base="1.2", config_path="/home/yangz/4D-Humans/hmr2/configs_hydra", config_name="train.yaml")
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
