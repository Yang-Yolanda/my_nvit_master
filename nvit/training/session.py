"""
Training session orchestration (readable flow).

Intended reading order:
  1) Runtime paths & seeds
  2) Data: dataset YAML → datamodule (train/val iterators)
  3) Model: cfg → Lightning module (freeze / mask)
  4) Trainer: loggers + callbacks (checkpoint cadence) → Lightning Trainer
  5) Optional FINETUNE_FROM weight load
  6) fit(): batches → forward → loss → backward → optimizer step (inside Lightning);
     ModelCheckpoint saves when step condition is met; training continues until max_steps / epochs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
from omegaconf import DictConfig, open_dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from nvit.utils.path_utils import get_project_root, sync_cfg_paths_output_to_hydra_run
from nvit.training.config_runtime import (
    ensure_data_root_env,
    patch_general_keys_for_train_loop,
    resolve_datasets_config_path,
    save_model_config_yaml,
)
from nvit.training.constants import DEFAULT_CHECKPOINT_EVERY_N_TRAIN_STEPS
from nvit.training.data_pipeline import build_dataset_config, build_hmr2_datamodule
from nvit.training.model_pipeline import build_guided_lightning_module, load_finetune_weights
from nvit.training.checkpoints import build_checkpoint_callbacks
from nvit.training.trainer_pipeline import (
    build_tensorboard_loggers,
    create_lightning_trainer,
    normalize_trainer_config,
    resolve_fit_checkpoint_path,
)

from hmr2.utils.misc import task_wrapper, log_hyperparameters
from hmr2.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@rank_zero_only
def _log_unified_run_dir(path: str) -> None:
    log.info(f"Unified run directory (checkpoints, tensorboard, configs): {path}")


@task_wrapper
def run_training_session(cfg: DictConfig) -> Optional[Tuple[dict, dict]]:
    # --- 1) Runtime: Hydra output dir, DATA_ROOT, reproducibility ---
    sync_cfg_paths_output_to_hydra_run(cfg)
    _log_unified_run_dir(str(cfg.paths.output_dir))

    ensure_data_root_env()

    seed = cfg.get("seed", 1234)
    pl.seed_everything(seed, workers=True)

    # --- 2) Data: YAML list of shards → HMR2DataModule (PyTorch Lightning iterators) ---
    ds_conf_name = resolve_datasets_config_path(cfg)
    dataset_cfg = build_dataset_config(ds_conf_name)

    save_model_config_yaml(cfg, cfg.paths.output_dir)

    datamodule = build_hmr2_datamodule(cfg, dataset_cfg)

    # --- 2b) GENERAL defaults used by trainer & checkpoint cadence ---
    patch_general_keys_for_train_loop(cfg)

    # --- 3) Model: architecture + optional backbone freeze + optional attention mask ---
    model = build_guided_lightning_module(cfg)

    # --- 4) Trainer: TensorBoard + step checkpoints + LR/health monitors ---
    logger, loggers = build_tensorboard_loggers(cfg)
    callbacks, _ckpt_root = build_checkpoint_callbacks(cfg)

    trainer_cfg = normalize_trainer_config(cfg)
    trainer = create_lightning_trainer(cfg, trainer_cfg, callbacks, loggers)

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

    # --- 5) Optional pretrained / finetune weights (before optimizer state from resume) ---
    load_finetune_weights(cfg, model)

    # --- 6) Resume full training state if requested / last.ckpt exists ---
    ckpt_path_to_use = resolve_fit_checkpoint_path(cfg)

    log.info(f"Trainer Max Steps: {trainer.max_steps}")
    log.info(f"Using ckpt_path: {ckpt_path_to_use}")

    # --- 7) Training loop (Lightning): batch → training_step → backward → step; callbacks save on schedule ---
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path_to_use)
    log.info("Fitting done")


def apply_hydra_main_config_defaults(cfg: DictConfig) -> None:
    """Patches that must run in `main` before @task_wrapper extras touch the config."""
    with open_dict(cfg):
        if "GENERAL" not in cfg:
            cfg.GENERAL = DictConfig({})
        if "LOG_STEPS" not in cfg.GENERAL:
            cfg.GENERAL.LOG_STEPS = 10
        if "VAL_STEPS" not in cfg.GENERAL:
            cfg.GENERAL.VAL_STEPS = 100
        if "TRAIN_BATCHES_PER_EPOCH" not in cfg.GENERAL and "CHECKPOINT_STEPS" in cfg.GENERAL:
            cfg.GENERAL.TRAIN_BATCHES_PER_EPOCH = cfg.GENERAL.CHECKPOINT_STEPS
        if "TRAIN_BATCHES_PER_EPOCH" not in cfg.GENERAL:
            cfg.GENERAL.TRAIN_BATCHES_PER_EPOCH = 3000
        if "CHECKPOINT_EVERY_N_TRAIN_STEPS" not in cfg.GENERAL:
            cfg.GENERAL.CHECKPOINT_EVERY_N_TRAIN_STEPS = DEFAULT_CHECKPOINT_EVERY_N_TRAIN_STEPS
        if "CHECKPOINT_SAVE_TOP_K" not in cfg.GENERAL:
            cfg.GENERAL.CHECKPOINT_SAVE_TOP_K = -1

        if "extras" in cfg:
            cfg.extras.print_config = False

        if "trainer" in cfg:
            cfg.trainer.log_every_n_steps = 10


def prepare_hydra_output_dir(cfg: DictConfig) -> None:
    sync_cfg_paths_output_to_hydra_run(cfg)
    if "paths" in cfg and "output_dir" in cfg.paths:
        Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
