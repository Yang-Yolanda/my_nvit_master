"""Config I/O and environment defaults used before model/data construction."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf
from yacs.config import CfgNode

from nvit.utils.path_utils import get_project_root
from hmr2.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def ensure_data_root_env() -> None:
    """dataset_tar URLs use ${DATA_ROOT}/...; if unset, paths stay literal and WebDataset fails loudly or appears hung."""
    if not os.environ.get("DATA_ROOT"):
        data_root = Path(get_project_root()) / "hmr2_training_data"
        os.environ["DATA_ROOT"] = str(data_root)
        log.info(f"DATA_ROOT was unset; defaulting to {os.environ['DATA_ROOT']}")


def resolve_datasets_config_path(cfg) -> str:
    _nvit_ds = Path(get_project_root()) / "scripts" / "datasets_tar.yaml"
    _default_ds = str(_nvit_ds) if _nvit_ds.is_file() else "datasets_tar.yaml"
    return cfg.get("DATASETS_CONFIG_FILE", _default_ds)


def save_model_config_yaml(model_cfg, rootdir: str) -> None:
    """Write merged Hydra cfg to model_config.yaml under the run output directory."""
    try:
        Path(rootdir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, "model_config.yaml"))
    except OSError as e:
        log.warning(f"Failed to save configs to {rootdir} (possibly OSS mount conflict): {e}")


def patch_general_keys_for_train_loop(cfg) -> None:
    """Fill GENERAL keys that otherwise break interpolation or trainer defaults."""
    if "GENERAL" not in cfg:
        cfg.GENERAL = CfgNode()
    if "LOG_STEPS" not in cfg.GENERAL:
        cfg.GENERAL.LOG_STEPS = 10
    if "TRAIN_BATCHES_PER_EPOCH" not in cfg.GENERAL and "CHECKPOINT_STEPS" in cfg.GENERAL:
        cfg.GENERAL.TRAIN_BATCHES_PER_EPOCH = cfg.GENERAL.CHECKPOINT_STEPS
    if "TRAIN_BATCHES_PER_EPOCH" not in cfg.GENERAL:
        cfg.GENERAL.TRAIN_BATCHES_PER_EPOCH = 3000

    from nvit.training.constants import DEFAULT_CHECKPOINT_EVERY_N_TRAIN_STEPS

    if "CHECKPOINT_EVERY_N_TRAIN_STEPS" not in cfg.GENERAL:
        cfg.GENERAL.CHECKPOINT_EVERY_N_TRAIN_STEPS = DEFAULT_CHECKPOINT_EVERY_N_TRAIN_STEPS
    if "CHECKPOINT_SAVE_TOP_K" not in cfg.GENERAL:
        cfg.GENERAL.CHECKPOINT_SAVE_TOP_K = -1

    cfg.trainer.log_every_n_steps = cfg.GENERAL.LOG_STEPS
