"""Checkpoint callbacks: step-based weights + full training state + LR monitor + health."""

from __future__ import annotations

import os

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from nvit.training.constants import DEFAULT_CHECKPOINT_EVERY_N_TRAIN_STEPS
from nvit.training.health_monitor import SystemHealthMonitor
from hmr2.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class UniqueModelCheckpoint(ModelCheckpoint):
    """Allow multiple ModelCheckpoint instances without state_key collisions."""

    def __init__(self, *args, **kwargs):
        self._state_key = kwargs.pop("state_key", None)
        super().__init__(*args, **kwargs)

    @property
    def state_key(self) -> str:
        return self._state_key if self._state_key else super().state_key


def build_checkpoint_callbacks(cfg):
    """
    Save step_{step}.ckpt (weights) and last.ckpt (full state) every GENERAL.CHECKPOINT_EVERY_N_TRAIN_STEPS.
    """
    gen = cfg.GENERAL
    ckpt_every = int(gen.get("CHECKPOINT_EVERY_N_TRAIN_STEPS", DEFAULT_CHECKPOINT_EVERY_N_TRAIN_STEPS))
    save_top_k_weights = int(gen.get("CHECKPOINT_SAVE_TOP_K", -1))

    ckpt_root = os.path.join(str(cfg.paths.output_dir), "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    weights_step_callback = UniqueModelCheckpoint(
        state_key="weights_step",
        dirpath=ckpt_root,
        save_weights_only=True,
        save_top_k=save_top_k_weights,
        every_n_train_steps=ckpt_every,
        every_n_epochs=None,
        filename="step_{step}",
        monitor=None,
        save_on_train_epoch_end=False,
    )

    last_state_callback = UniqueModelCheckpoint(
        state_key="last",
        dirpath=ckpt_root,
        save_weights_only=False,
        save_last=True,
        save_top_k=0,
        every_n_train_steps=ckpt_every,
        every_n_epochs=None,
        monitor=None,
        save_on_train_epoch_end=False,
    )

    log.info(
        f"Checkpoints: step_{{step}}.ckpt (weights only) every {ckpt_every} steps (top_k={save_top_k_weights}); "
        f"last.ckpt (full state) every {ckpt_every} steps."
    )
    log.info(f"Checkpoint directory (look here, not under log_dir root): {ckpt_root}")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    health_monitor = SystemHealthMonitor(log_interval=30)

    return [weights_step_callback, last_state_callback, lr_monitor, health_monitor], ckpt_root
