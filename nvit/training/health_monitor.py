"""Optional host/GPU health logging during training (rank 0)."""

from __future__ import annotations

import time

import psutil
import pytorch_lightning as pl
import torch
from hmr2.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SystemHealthMonitor(pl.Callback):
    def __init__(self, log_interval: int = 30):
        super().__init__()
        self.log_interval = log_interval
        self.last_log_time = time.time()

    @pl.utilities.rank_zero.rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if time.time() - self.last_log_time < self.log_interval:
            return

        cpu_pct = psutil.cpu_percent()
        mem = psutil.virtual_memory()

        gpu_str = ""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    used = total - free
                    util = used / total * 100
                    gpu_str += f" GPU{i}:{util:.1f}%({used/1024**3:.1f}GB)"
                except Exception:
                    pass

        diag_str = ""
        if mem.percent > 90:
            try:
                dl = trainer.train_dataloader
                while hasattr(dl, "loader"):
                    dl = dl.loader
                diag_str = (
                    f" | ⚠️ [OOM-Risk] Workers:{getattr(dl, 'num_workers', 'N/A')} "
                    f"B:{getattr(dl, 'batch_size', 'N/A')}"
                )
            except Exception:
                pass

        log.info(
            f"❤️ [Health] Step:{trainer.global_step} | Host CPU:{cpu_pct}% | "
            f"Host RAM:{mem.percent}% ({mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB){diag_str} |{gpu_str}"
        )

        if mem.percent > 95:
            log.error("🛑 CRITICAL Host RAM usage (>95%)! Watchdog may kill process soon.")

        self.last_log_time = time.time()
