#!/usr/bin/env python3
"""对多个 Lightning checkpoint 依次做同一套前向测速，输出与 bench_hmr2_vs_nvit 相同列的 CSV（多行）。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
_ART = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_ART) not in sys.path:
    sys.path.insert(0, str(_ART))
from bench_hmr2_vs_nvit_inference import _bench_one  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="多模型推理测速。每项: --entry NAME=CKPT（可重复）"
    )
    p.add_argument(
        "--entry",
        action="append",
        default=[],
        metavar="NAME=CKPT",
    )
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out-csv", type=Path, required=True)
    args = p.parse_args()
    if not args.entry:
        raise SystemExit("请至少给一项 --entry NAME=CKPT")

    pairs: list[tuple[str, Path]] = []
    for s in args.entry:
        if "=" not in s:
            raise SystemExit(f"非法 --entry（需 NAME=路径）: {s}")
        name, path = s.split("=", 1)
        ck = Path(path.strip())
        if not ck.is_file():
            raise SystemExit(f"checkpoint 不存在: {ck}")
        pairs.append((name.strip(), ck))

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    rows: list[dict] = []
    for name, ckpt in pairs:
        r = _bench_one(
            name, ckpt, device, args.batch, args.warmup, args.iters, args.amp
        )
        rows.append(r)
        print(
            f"[{r['name']}] ms/img={r['ms_per_image']:.3f}  "
            f"img/s={r['images_per_s']:.2f}  {ckpt}"
        )

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
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"[ok] {args.out_csv}")


if __name__ == "__main__":
    main()
