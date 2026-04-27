"""Small-subset datamodule for pilot / sanity checks (not used in production HMR2 tar training)."""

from __future__ import annotations

import gc

import pyrootutils
import torch
import torch.utils.data
import pytorch_lightning as pl
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from nvit.bio_dataset import BioMambaDataset


class GuidedDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        project_root = pyrootutils.find_root()
        self.dataset_file = str(project_root.parent / "4D-Humans" / "data/metadata/3dpw_test.npz")
        self.img_dir = str(project_root.parent / "4D-Humans" / "data/3DPW")

    def setup(self, stage=None):
        _, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        self.train_ds = BioMambaDataset(m_cfg, dataset_file=self.dataset_file, img_dir=self.img_dir, train=True)
        self.val_ds = BioMambaDataset(m_cfg, dataset_file=self.dataset_file, img_dir=self.img_dir, train=False)
        gc.freeze()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            prefetch_factor=getattr(self.cfg.GENERAL, "PREFETCH_FACTOR", 2),
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.GENERAL.NUM_WORKERS,
        )
