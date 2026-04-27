"""
SMPLer → CH5 / HMR2 Evaluator bridge.

Reuses the same `create_dataset` / crop pipeline as HMR2 by exposing `.cfg` compatible
with `hmr2.datasets.create_dataset`, and returns `forward(batch)` dict keys expected by
`hmr2.utils.pose_utils.Evaluator`: `pred_keypoints_3d`, `pred_keypoints_2d`.

SMPL parameters are taken from SMPLer's final stage; 3D joints are produced with the
same `hmr2.models.SMPL` wrapper as HMR2 so joint indexing matches `KEYPOINT_LIST`.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmr2.utils.geometry import perspective_projection

logger = logging.getLogger(__name__)


def _patch_smpler_config(smpler_root: Path) -> None:
    """Rewrite SMPLer `config` paths to absolute locations under `smpler_root`."""
    smpler_root = smpler_root.resolve()
    import importlib

    cfg_mod = importlib.import_module("config")
    cfg_mod.smpl_mean_params_path = str(smpler_root / cfg_mod.smpl_mean_params_path)
    cfg_mod.smpl_neutral = str(smpler_root / cfg_mod.smpl_neutral)
    cfg_mod.JOINT_REGRESSOR_H36M_correct = str(smpler_root / cfg_mod.JOINT_REGRESSOR_H36M_correct)
    cfg_mod.JOINT_REGRESSOR_3DPW = str(smpler_root / cfg_mod.JOINT_REGRESSOR_3DPW)
    hd = cfg_mod.hrnet_dict
    patched = {}
    for k, v in hd.items():
        patched[k] = (str(smpler_root / v[0]), str(smpler_root / v[1]), v[2])
    cfg_mod.hrnet_dict = patched


def _import_smpler_modules(smpler_root: Path) -> Tuple[Any, Any]:
    """Load SMPLer's `models.*` without colliding with other `models` bindings (e.g. timm/nvit)."""
    root = smpler_root.resolve()
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))

    # SMPLer uses top-level names `config`, `utils`, `models` — clear stale imports.
    if "config" in sys.modules:
        del sys.modules["config"]
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]

    _patch_smpler_config(smpler_root)

    # Register SMPLer's local packages (directories under repo root)
    models_dir = root / "models"
    _pkg = types.ModuleType("models")
    _pkg.__path__ = [str(models_dir)]  # type: ignore[attr-defined]
    sys.modules["models"] = _pkg

    _utils_pkg = types.ModuleType("utils")
    _utils_pkg.__path__ = [str(root / "utils")]  # type: ignore[attr-defined]
    sys.modules["utils"] = _utils_pkg

    smpler_mod = importlib.import_module("models.smpler")
    basics_mod = importlib.import_module("models.transformer_basics")
    return smpler_mod.SMPLer, basics_mod.TranformerConfig


def _load_smpler_network(
    smpler_root: Path,
    checkpoint_path: Path,
    hrnet_type: str,
    data_mode: str,
    num_transformers: int = 3,
    device: torch.device | None = None,
) -> nn.Module:
    SMPLer, TranformerConfig = _import_smpler_modules(smpler_root)
    import config as project_cfg  # patched paths  # noqa: E402

    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"SMPLer checkpoint not found: {ckpt}")

    args = Namespace(
        data_mode=data_mode,
        model_type="smpler",
        hrnet_type=hrnet_type,
        num_transformers=num_transformers,
    )
    trans_cfg = TranformerConfig()
    trans_cfg.raw_feat_dim = project_cfg.hrnet_dict[hrnet_type][2]

    model = SMPLer(args, trans_cfg)
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def _load_reference_smpl_and_cfg(
    hmr2_cfg_ckpt: str,
    device: torch.device,
):
    """Use any NViT/HMR2 ckpt that loads with `load_model_from_ckpt` to obtain `cfg` + `smpl`."""
    from nvit.utils.model_io import load_model_from_ckpt

    ref = load_model_from_ckpt(hmr2_cfg_ckpt, device=device)
    if not hasattr(ref, "smpl"):
        raise RuntimeError("Reference checkpoint must expose `.smpl` (HMR2 / GuidedHMR2).")
    return ref.cfg, ref.smpl


class SMPLerCH5Wrapper(nn.Module):
    """
    Wraps SMPLer inference + HMR2 SMPL decoding for CH5 evaluation.

    `forward` returns the same keys as `GuidedHMR2Module` / `HMR2.forward_step` for 3D/2D kpts.
    """

    def __init__(
        self,
        smpler: nn.Module,
        cfg,
        smpl: nn.Module,
        input_size: int = 224,
    ):
        super().__init__()
        self.smpler = smpler
        self.cfg = cfg
        self.smpl = smpl
        self.input_size = input_size

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        img = batch["img"]
        if img.shape[-1] != self.input_size or img.shape[-2] != self.input_size:
            img = F.interpolate(img, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

        pred_dicts = self.smpler(img)
        last = pred_dicts[-1]
        theta = last["theta"]
        beta = last["beta"]
        pred_cam = last["cam"]

        batch_size = theta.shape[0]
        device = theta.device
        dtype = theta.dtype

        global_orient = theta[:, :1]
        body_pose = theta[:, 1:]
        pred_smpl_params = {
            "global_orient": global_orient.reshape(batch_size, -1, 3, 3),
            "body_pose": body_pose.reshape(batch_size, -1, 3, 3),
            "betas": beta.reshape(batch_size, -1),
        }

        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )

        smpl_out = self.smpl(**{k: v.float() for k, v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_out.joints.reshape(batch_size, -1, 3)
        pred_vertices = smpl_out.vertices.reshape(batch_size, -1, 3)

        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(
            pred_keypoints_3d,
            translation=pred_cam_t,
            focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE,
        )
        pred_keypoints_2d = pred_keypoints_2d.reshape(batch_size, -1, 2)

        return {
            "pred_cam": pred_cam,
            "pred_smpl_params": pred_smpl_params,
            "pred_keypoints_3d": pred_keypoints_3d,
            "pred_vertices": pred_vertices,
            "pred_keypoints_2d": pred_keypoints_2d,
            "pred_cam_t": pred_cam_t,
        }


def build_smpler_ch5_wrapper(
    *,
    smpler_root: Optional[Path] = None,
    smpler_ckpt: Optional[Path] = None,
    hrnet_type: str = "w32",
    data_mode: str = "h36m",
    hmr2_cfg_ckpt: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> SMPLerCH5Wrapper:
    """
    Build SMPLer + HMR2-SMPL wrapper.

    Environment:
      SMPLER_ROOT — SMPLer repo root (default: /home/yangz/external_baselines/SMPLer)
      HMR2_CFG_REFERENCE_CKPT — any loadable NViT/HMR2 ckpt for `cfg` + `smpl` (required if hmr2_cfg_ckpt is None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(smpler_root or os.environ.get("SMPLER_ROOT", "/home/yangz/external_baselines/SMPLer"))
    ck = Path(smpler_ckpt or os.environ.get("SMPLER_CKPT", ""))
    if not ck.is_file():
        raise FileNotFoundError(
            f"Set smpler_ckpt or SMPLER_CKPT to SMPLer weights (e.g. pretrained/SMPLer_h36m.pt). Got: {ck}"
        )

    ref_ckpt = hmr2_cfg_ckpt or os.environ.get("HMR2_CFG_REFERENCE_CKPT")
    if not ref_ckpt or not os.path.isfile(ref_ckpt):
        raise FileNotFoundError(
            "Need HMR2_CFG_REFERENCE_CKPT or --hmr2_cfg_ckpt pointing to a NViT/HMR2 checkpoint "
            "for SMPL head config (joint order / projection)."
        )

    smpler = _load_smpler_network(root, ck, hrnet_type=hrnet_type, data_mode=data_mode, device=device)
    cfg, smpl = _load_reference_smpl_and_cfg(ref_ckpt, device=device)
    smpl = smpl.to(device)
    return SMPLerCH5Wrapper(smpler, cfg, smpl, input_size=224)
