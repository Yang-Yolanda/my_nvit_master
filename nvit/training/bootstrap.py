"""Process-level setup: safe printing on shared FS, Torch/CUDA/DataLoader tuning."""

from __future__ import annotations

import builtins
import os
import sys


def install_safe_print_for_shared_fs() -> None:
    """CPFS/NFS + concurrent jobs can make stdout raise OSError 22; smplx uses print() in SMPL __init__."""
    _orig = builtins.print

    def _safe_print(*args, **kwargs):
        try:
            return _orig(*args, **kwargs)
        except OSError:
            try:
                kw = dict(kwargs)
                if kw.get("file") in (None, sys.stdout):
                    kw["file"] = sys.stderr
                return _orig(*args, **kw)
            except OSError:
                return None

    builtins.print = _safe_print


def configure_training_runtime_environment() -> None:
    """Torch backends, cudnn benchmark, multiprocessing sharing, OpenCV/OpenMP thread limits."""
    import cv2
    import torch
    import torch.multiprocessing

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy("file_system")
    cv2.setNumThreads(0)
    os.environ["OMP_NUM_THREADS"] = "1"
