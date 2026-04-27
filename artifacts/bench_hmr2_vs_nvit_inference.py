#!/usr/bin/env python3
"""
对比两个 HMR2 系 checkpoint 的纯前向推理延迟（同一环境、同一输入尺寸）。

说明:
- METRO / Mesh Graphormer 不在 4D-humans 栈里，本脚本不覆盖；若要比其速度，需在各自仓库单独跑同类 micro-benchmark。
- 使用与 `nvit.utils.model_io.load_model_from_ckpt` 相同的加载逻辑；batch 为 {"img": (B,3,256,256)}。

例:
  cd /cpfs_infra/shared/yangz/NViT-master
  python artifacts/bench_hmr2_vs_nvit_inference.py \\
    --ckpt-a /cpfs_infra/shared/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt \\
    --ckpt-b /mnt/yangz/nvit_output/ch6/train/runs/.../checkpoints/step_step=492000.ckpt \\
    --label-a HMR2_multirun --label-b NViT_ch6_best
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "nvit") not in sys.path:
    sys.path.insert(0, str(_REPO))

from nvit.utils.model_io import load_model_from_ckpt  # noqa: E402


def _load_for_bench(ckpt: Path, device: torch.device):
    ref = os.environ.get("HMR2_PTH_REF_CKPT") or os.environ.get("NVIT_PTH_REF_CKPT")
    if ref and str(ckpt).lower().endswith((".pth", ".pt")):
        from nvit.utils.hmr2_pruned_pth import load_model_hmr2_pth_or_ckpt

        return load_model_hmr2_pth_or_ckpt(
            str(ckpt), str(device), ref
        )
    return load_model_from_ckpt(str(ckpt), device=str(device))


def _bench_one(
    name: str,
    ckpt: Path,
    device: torch.device,
    batch: int,
    warmup: int,
    iters: int,
    use_amp: bool,
) -> dict:
    model = _load_for_bench(ckpt, device)
    model.eval()
    img = torch.randn(batch, 3, 256, 256, device=device, dtype=torch.float32)
    batch_d = {"img": img}

    @torch.inference_mode()
    def _run() -> None:
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(batch_d)
        else:
            _ = model(batch_d)

    for _ in range(warmup):
        _run()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_s = t1 - t0
    per_step_ms = 1000.0 * total_s / iters
    per_img_ms = per_step_ms / batch
    return {
        "name": name,
        "ckpt": str(ckpt),
        "batch": batch,
        "iters": iters,
        "ms_per_step": per_step_ms,
        "ms_per_image": per_img_ms,
        "images_per_s": 1000.0 / per_img_ms if per_img_ms > 0 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", type=Path, required=True)
    ap.add_argument("--ckpt-b", type=Path, required=True)
    ap.add_argument("--label-a", type=str, default="A")
    ap.add_argument("--label-b", type=str, default="B")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--amp", action="store_true", help="torch.autocast fp16 (CUDA only)")
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    for p in (args.ckpt_a, args.ckpt_b):
        if not p.is_file():
            raise SystemExit(f"checkpoint 不存在: {p}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ra = _bench_one(
        args.label_a, args.ckpt_a, device, args.batch, args.warmup, args.iters, args.amp
    )
    rb = _bench_one(
        args.label_b, args.ckpt_b, device, args.batch, args.warmup, args.iters, args.amp
    )

    lines = [
        f"device={device} batch={args.batch} warmup={args.warmup} iters={args.iters} amp={args.amp}",
        f"[{ra['name']}] ms/step={ra['ms_per_step']:.3f}  ms/img={ra['ms_per_image']:.3f}  img/s={ra['images_per_s']:.2f}",
        f"    {ra['ckpt']}",
        f"[{rb['name']}] ms/step={rb['ms_per_step']:.3f}  ms/img={rb['ms_per_image']:.3f}  img/s={rb['images_per_s']:.2f}",
        f"    {rb['ckpt']}",
    ]
    text = "\n".join(lines) + "\n"
    print(text)

    if args.out_csv:
        import csv

        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "ckpt",
                    "batch",
                    "iters",
                    "ms_per_step",
                    "ms_per_image",
                    "images_per_s",
                ],
            )
            w.writeheader()
            w.writerow({k: ra[k] for k in w.fieldnames if k in ra})
            w.writerow({k: rb[k] for k in w.fieldnames if k in rb})
        print(f"[ok] {args.out_csv}")


if __name__ == "__main__":
    main()
