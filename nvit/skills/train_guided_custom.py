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
from pathlib import Path

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

import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)

log = get_pylogger(__name__)

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
        return torch.utils.data.DataLoader(self.train_ds, batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=self.cfg.GENERAL.NUM_WORKERS)
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=self.cfg.GENERAL.NUM_WORKERS)
    
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
    dataset_cfg = dataset_config()

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

    # Setup Tensorboard logger
    logger = TensorBoardLogger(os.path.join(cfg.paths.output_dir, 'tensorboard'), name='', version='', default_hp_metric=False)
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'), 
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS, 
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint_callback, 
        lr_monitor,
    ]
    # [Fix] Force DDP strategy with unused params by overriding Hydra config
    # Convert DictConfig to dict to allow popping
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    if 'strategy' in trainer_cfg:
        trainer_cfg.pop('strategy')
    
    
    # [Autonomous Mode] Dynamic Device Configuration
    # If devices not set, default to 1. If set to >1, force DDP with unused params.
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
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path='last')
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
            
    print("DEBUG: Config Keys:", cfg.keys())
    if 'MODEL' in cfg:
         print("DEBUG: MODEL found.")
    else:
         print("DEBUG: MODEL MISSING!")

    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
