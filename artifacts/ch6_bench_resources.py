#!/usr/bin/env python3
"""
Chapter 6 资源表 bench：在统一协议下测 Params / FLOPs(可选) / Latency / 峰值显存。

依赖: 需 GPU、torch；FLOPs 需 `thop`（与 nvit/efficiency_profiler 一致；未安装则 FLOPs 列为 —）。

例:
  export PYTHONPATH=/cpfs/.../NViT-master
  python3 artifacts/ch6_bench_resources.py --out /path/ch6_resource_table.csv

可选环境变量（缺省则只测存在的 ckpt）:
  CH6_BENCH_HMR2_CKPT    标准 HMR2 / 32L baseline ckpt
  CH6_BENCH_TRUNC_CKPT   浅层/截断 ViT ckpt（0–11 等）
  CH6_BENCH_NVIT_CKPT    默认: ch6 360k hybrid
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# repo root
ART = Path(__file__).resolve().parent
REPO = ART.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("PYTHONPATH", str(REPO))

import torch  # noqa: E402

DEFAULT_NVIT = (
    REPO
    / "output/ch6/train/runs/2026-04-17_13-28-24/checkpoints/step_step=360000.ckpt"
)
WARMUP = 50
REPEAT = 200
# 与 README 中论文/表格约定一致: RGB 高×宽
H, W = 256, 192
B = 1


def _thop_gflops(model, batch: dict) -> float | None:
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(batch,), verbose=False)
        gflops = float(macs * 2) / 1e9
        return gflops
    except Exception as e:  # noqa: BLE001
        print(f"WARN thop: {e}", file=sys.stderr)
        return None


@torch.inference_mode()
def bench_one(
    name: str,
    ckpt: Path,
    device: str,
) -> dict:
    from nvit.utils.model_io import load_model_from_ckpt

    if not ckpt.is_file():
        return {
            "architecture": name,
            "params_m": "—",
            "flops_g": "—",
            "latency_ms": "—",
            "vram_mb": "—",
            "note": f"missing checkpoint: {ckpt}",
        }

    print(f"Loading {name} <- {ckpt} ...", flush=True)
    model = load_model_from_ckpt(str(ckpt), device=device)
    model.eval()

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，无法按协议 bench。")

    dev = torch.device(device)
    if device.startswith("cuda"):
        torch.cuda.set_device(dev.index if dev.index is not None else 0)

    batch = {
        "img": torch.randn(B, 3, H, W, device=dev, dtype=torch.float32),
    }

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flops = _thop_gflops(model, batch)
    flop_s = f"{flops:.2f}" if flops is not None else "—"

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            _ = model(batch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            _ = model(batch)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / REPEAT * 1e3
        vram_mb = torch.cuda.max_memory_allocated(dev) / (1024.0**2)
    else:
        latency_ms = 0.0
        for _ in range(WARMUP):
            _ = model(batch)
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            _ = model(batch)
        latency_ms = (time.perf_counter() - t0) / REPEAT * 1e3
        vram_mb = 0.0

    meta = (
        f"FP32, forward only, {H}x{W} RGB, B={B}, "
        f"warmup={WARMUP} repeat={REPEAT}, cuda.sync, "
        f"thop=GFLOP(2*MAC) or —"
    )
    return {
        "architecture": name,
        "params_m": f"{params_m:.3f}",
        "flops_g": flop_s,
        "latency_ms": f"{latency_ms:.3f}",
        "vram_mb": f"{vram_mb:.1f}",
        "note": meta,
    }


def run_bench(
    out_csv: Path,
    hmr2_ckpt: Path | None,
    trunc_ckpt: Path | None,
    nvit_ckpt: Path | None,
    device: str = "cuda:0",
) -> list[dict]:
    hmr2 = hmr2_ckpt or (Path(os.environ["CH6_BENCH_HMR2_CKPT"]) if os.environ.get("CH6_BENCH_HMR2_CKPT") else None)
    trunc = trunc_ckpt or (Path(os.environ["CH6_BENCH_TRUNC_CKPT"]) if os.environ.get("CH6_BENCH_TRUNC_CKPT") else None)
    nvit = nvit_ckpt
    if nvit is None and os.environ.get("CH6_BENCH_NVIT_CKPT"):
        nvit = Path(os.environ["CH6_BENCH_NVIT_CKPT"])
    if nvit is None:
        nvit = DEFAULT_NVIT

    specs: list[tuple[str, Path | None]] = [
        ("HMR2 ViT-32 (baseline)", hmr2),
        ("Trunc. ViT 0–11 (or shallow ckpt)", trunc),
        ("KTI-guided hybrid (NViT ch6)", nvit),
    ]
    rows: list[dict] = []
    for label, ck in specs:
        if not ck or not Path(ck).is_file():
            print(f"SKIP (no valid ckpt): {label} -> {ck}", file=sys.stderr)
            rows.append(
                {
                    "architecture": label,
                    "params_m": "—",
                    "flops_g": "—",
                    "latency_ms": "—",
                    "vram_mb": "—",
                    "note": f"set --ckpt-* or CH6_BENCH_HMR2_CKPT / _TRUNC_CKPT / _NVIT_CKPT; was {ck}",
                }
            )
            continue
        try:
            rows.append(bench_one(label, Path(ck), device))
        except Exception as e:  # noqa: BLE001
            print(f"ERR {label}: {e}", file=sys.stderr)
            rows.append(
                {
                    "architecture": label,
                    "params_m": "—",
                    "flops_g": "—",
                    "latency_ms": "—",
                    "vram_mb": "—",
                    "note": str(e)[:200],
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    import csv

    fields = [
        "architecture",
        "params_m",
        "flops_g",
        "latency_ms",
        "vram_mb",
        "note",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "input_hw": [H, W],
        "batch": B,
        "dtype": "float32",
        "warmup": WARMUP,
        "repeat": REPEAT,
        "timing": "time.perf_counter, cuda.synchronize, forward only, no dataloader",
        "flops_tool": "thop (2*MAC as GFLOPs) if installed",
    }
    (out_csv.parent / (out_csv.stem + "_bench_meta.json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/cpfs_infra/shared/yangz/0228/图表/chapter06/ch6_resource_table.csv"
        ),
    )
    ap.add_argument(
        "--ckpt-hmr2", type=Path, default=None, help="HMR2 / 32L; 可略过以仅测 NViT"
    )
    ap.add_argument("--ckpt-trunc", type=Path, default=None, help="截断 shallow ckpt")
    ap.add_argument(
        "--ckpt-nvit",
        type=Path,
        default=DEFAULT_NVIT,
        help="NViT hybrid（默认 ch6 360k）",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    if not args.ckpt_nvit.is_file() and "CH6_BENCH_NVIT_CKPT" not in os.environ:
        print("WARN: default NViT ckpt not found; set --ckpt-nvit or CH6_BENCH_NVIT_CKPT", file=sys.stderr)

    run_bench(
        out_csv=args.out,
        hmr2_ckpt=args.ckpt_hmr2,
        trunc_ckpt=args.ckpt_trunc,
        nvit_ckpt=args.ckpt_nvit,
        device=args.device,
    )
    print("Wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
