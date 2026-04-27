#!/usr/bin/env python3
"""
Evaluate SMPLer via SMPLerCH5Wrapper using the same CH5 pipeline as `standard_eval.py`.

Example:
  export SMPLER_ROOT=/home/yangz/external_baselines/SMPLer
  export PYTHONPATH=/cpfs_infra/shared/yangz/NViT-master:/cpfs_infra/shared/yangz/4D-Humans:$PYTHONPATH
  python nvit/eval_smpler_ch5.py \\
    --ckpt /path/to/nvit_or_hmr2.ckpt \\
    --smpler_ckpt $SMPLER_ROOT/pretrained/SMPLer_3dpw.pt \\
    --dataset 3DPW-TEST \\
    --data_mode 3dpw
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Project imports (same pattern as standard_eval)
nvit_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(nvit_dir.parent))

from nvit.skills.evaluate_model.standard_eval import EvaluatorSkill  # noqa: E402
from nvit.external_baselines.smpler_adapter import build_smpler_ch5_wrapper  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CH5 eval for SMPLer (SMPLerCH5Wrapper)")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Reference NViT/HMR2 checkpoint (for cfg + hmr2 SMPL joint order only).",
    )
    parser.add_argument("--smpler_ckpt", type=str, required=True, help="Path to SMPLer_*.pt")
    parser.add_argument(
        "--smpler_root",
        type=str,
        default=os.environ.get("SMPLER_ROOT", "/home/yangz/external_baselines/SMPLer"),
    )
    parser.add_argument("--hrnet_type", type=str, default="w32", choices=["w32", "w48"])
    parser.add_argument(
        "--data_mode",
        type=str,
        default="h36m",
        choices=["h36m", "3dpw"],
        help="Must match the SMPLer checkpoint (see SMPLer README: h36m vs 3dpw weights).",
    )
    parser.add_argument("--dataset", type=str, default="3DPW-TEST")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--skip_errors", action="store_true", default=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get(
            "HMR2_EVAL_DATA_DIR",
            "/cpfs_infra/shared/yangz/4D-Humans/hmr2_evaluation_data",
        ),
    )
    parser.add_argument("--use_mean_alignment", action="store_true")
    args = parser.parse_args()

    os.environ["SMPLER_ROOT"] = args.smpler_root

    class SMPLerEvalSkill(EvaluatorSkill):
        def load_model(self, ckpt_path: str, pth_ref_ckpt=None):
            del pth_ref_ckpt  # pruned .pth + ref ckpt: not used for SMPLer wrapper
            return build_smpler_ch5_wrapper(
                smpler_root=Path(args.smpler_root),
                smpler_ckpt=Path(args.smpler_ckpt),
                hrnet_type=args.hrnet_type,
                data_mode=args.data_mode,
                hmr2_cfg_ckpt=ckpt_path,
                device=self.device,
            )

    skill = SMPLerEvalSkill(gpu=args.gpu)
    ns = argparse.Namespace(
        ckpt=args.ckpt,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_batches=args.limit_batches,
        skip_errors=args.skip_errors,
        output=args.output,
        data_dir=args.data_dir,
        use_mean_alignment=args.use_mean_alignment,
    )
    skill.run_eval(ns)


if __name__ == "__main__":
    main()
