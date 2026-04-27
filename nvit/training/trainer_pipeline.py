"""Trainer: TensorBoard, Lightning Trainer config normalization, resume path, then .fit()."""

from __future__ import annotations

import os
import signal
from typing import Any, List, Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from hmr2.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def build_tensorboard_loggers(cfg: DictConfig):
    logger = TensorBoardLogger(
        os.path.join(cfg.paths.output_dir, "tensorboard"),
        name="",
        version="",
        default_hp_metric=False,
    )
    return logger, [logger]


def normalize_trainer_config(cfg: DictConfig) -> dict:
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)

    _st = trainer_cfg.get("strategy")
    if isinstance(_st, dict) and set(_st.keys()) <= {"find_unused_parameters"}:
        _fu = bool(_st.get("find_unused_parameters", False))
        trainer_cfg["strategy"] = (
            "ddp_find_unused_parameters_true" if _fu else "ddp_find_unused_parameters_false"
        )
        log.warning("trainer.strategy was a Hydra-merged dict; normalized to %r", trainer_cfg["strategy"])

    trainer_cfg["limit_val_batches"] = 0.0
    trainer_cfg["check_val_every_n_epoch"] = None
    trainer_cfg["num_sanity_val_steps"] = 0

    if "limit_train_batches" not in trainer_cfg and "TRAIN_BATCHES_PER_EPOCH" in cfg.GENERAL:
        trainer_cfg["limit_train_batches"] = cfg.GENERAL.TRAIN_BATCHES_PER_EPOCH

    if "devices" not in trainer_cfg:
        trainer_cfg["devices"] = 1

    return trainer_cfg


def create_lightning_trainer(
    cfg: DictConfig,
    trainer_cfg: dict,
    callbacks: List[pl.Callback],
    loggers: List[Any],
) -> Trainer:
    num_nodes = int(trainer_cfg.get("num_nodes", 1) or 1)
    dev = trainer_cfg["devices"]
    if isinstance(dev, (list, tuple)):
        n_devices = len(dev)
    else:
        try:
            n_devices = int(dev)
        except (TypeError, ValueError):
            n_devices = 1

    log.info(
        f"Instantiating trainer <{cfg.trainer._target_}> with "
        f"num_nodes={num_nodes}, devices={trainer_cfg['devices']} (per-node GPU count={n_devices})"
    )

    if num_nodes == 1 and n_devices == 1 and "strategy" in trainer_cfg:
        trainer_cfg.pop("strategy")

    plugins = SLURMEnvironment(requeue_signal=signal.SIGUSR2) if (cfg.get("launcher", None) is not None) else None

    return hydra.utils.instantiate(
        trainer_cfg,
        callbacks=callbacks,
        logger=loggers,
        plugins=plugins,
    )


def resolve_fit_checkpoint_path(cfg: DictConfig) -> Optional[str]:
    explicit_ckpt = cfg.get("ckpt_path", None)
    if explicit_ckpt is not None and explicit_ckpt != "null":
        log.info(f"Using explicitly provided checkpoint for resumption: {explicit_ckpt}")
        return explicit_ckpt

    auto_last = os.path.join(cfg.paths.output_dir, "checkpoints", "last.ckpt")
    if os.path.isfile(auto_last):
        log.info(f"✨ Auto-resuming from Master Resumption Node: {auto_last}")
        return auto_last

    log.info("No matching 'last.ckpt' found. Starting fresh run.")
    return None
