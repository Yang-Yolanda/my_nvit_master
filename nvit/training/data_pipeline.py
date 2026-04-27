"""Data: resolve dataset YAML → Lightning datamodule (iterators for train/val)."""

from __future__ import annotations

from hmr2.configs import dataset_config
from hmr2.datasets import HMR2DataModule


def build_dataset_config(ds_conf_name: str):
    return dataset_config(ds_conf_name)


def build_hmr2_datamodule(cfg, dataset_cfg):
    """Production path: WebDataset / tar mix from `datasets_tar.yaml` + cfg."""
    return HMR2DataModule(cfg, dataset_cfg)
