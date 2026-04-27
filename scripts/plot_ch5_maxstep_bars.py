#!/usr/bin/env python3
"""
从 metrics_master.csv 读取 CH5 消融结果，按「每组最大训练 step」取一行，画 MPJPE / PA-MPJPE 柱状图（不跑评测）。

experiment 形如 ch5/M2_Pos24/step_90000；同组同数据集多行时取 step 最大者；step 相同时优先无 limit_batches 的全量行。
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_experiment(exp: str) -> tuple[str | None, int]:
    m = re.match(r"ch5/([^/]+)/step_(\d+)\s*$", str(exp).strip())
    if not m:
        return None, -1
    return m.group(1), int(m.group(2))


def _group_order_key(name: str) -> tuple[int, str]:
    m = re.match(r"^M(\d+)_", name)
    return (int(m.group(1)), name) if m else (9999, name)


def _parse_limit_batches(val: str) -> int | None:
    s = (val or "").strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def load_ch5_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r.get("chapter") or "").strip() != "ch5":
                continue
            if (r.get("family") or "").strip() != "NViT":
                continue
            g, step = _parse_experiment(r.get("experiment") or "")
            if g is None or step < 0:
                continue
            r["_group"] = g
            r["_step"] = step
            r["_limit"] = _parse_limit_batches(r.get("limit_batches") or "")
            r["_ts"] = _parse_ts(r.get("timestamp_utc") or "")
            rows.append(r)
    return rows


def pick_best_per_group_dataset(
    rows: list[dict[str, Any]],
    *,
    eval_mode: str,
    datasets_filter: set[str] | None,
) -> list[dict[str, Any]]:
    """eval_mode: full | smoke | any (prefer full when choosing among max-step ties)."""
    filtered: list[dict[str, Any]] = []
    for r in rows:
        ds = (r.get("dataset") or "").strip()
        if datasets_filter is not None and ds not in datasets_filter:
            continue
        lim = r["_limit"]
        if eval_mode == "full" and lim is not None:
            continue
        if eval_mode == "smoke" and lim is None:
            continue
        filtered.append(r)

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in filtered:
        buckets[(r["_group"], (r.get("dataset") or "").strip())].append(r)

    out: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        max_step = max(x["_step"] for x in bucket)
        at_max = [x for x in bucket if x["_step"] == max_step]
        fulls = [x for x in at_max if x["_limit"] is None]
        if eval_mode == "any" and fulls:
            chosen = fulls
        elif eval_mode == "any" and not fulls:
            chosen = at_max
        else:
            chosen = at_max
        best = max(chosen, key=lambda x: x["_ts"])
        out.append(best)
    return out


def _float_metric(row: dict[str, Any], col: str) -> float | None:
    v = row.get(col)
    if v is None or str(v).strip() == "" or str(v).lower() == "nan":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def plot_dataset(
    picked: list[dict[str, Any]],
    dataset: str,
    out_path: Path,
    title_suffix: str,
) -> None:
    import matplotlib.pyplot as plt

    sub = [r for r in picked if (r.get("dataset") or "").strip() == dataset]
    if not sub:
        raise SystemExit(f"No rows for dataset={dataset!r}")

    groups = sorted({r["_group"] for r in sub}, key=_group_order_key)
    mpjpe = []
    pa = []
    for g in groups:
        row = next(r for r in sub if r["_group"] == g)
        mpjpe.append(_float_metric(row, "MPJPE_mm"))
        pa.append(_float_metric(row, "PA_MPJPE_mm"))

    max_step = max((r["_step"] for r in sub), default=-1)
    step_note = f", max step={max_step}" if max_step >= 0 else ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, vals, ylab, t in (
        (axes[0], mpjpe, "MPJPE (mm)", "MPJPE"),
        (axes[1], pa, "PA-MPJPE (mm)", "PA-MPJPE"),
    ):
        ax.bar(
            [str(g) for g in groups],
            [v if v is not None else float("nan") for v in vals],
            color="#4C72B0",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_ylabel(ylab)
        ax.set_xlabel("CH5 group")
        ax.set_title(f"{t} — {dataset}{title_suffix}{step_note}")
        ax.tick_params(axis="x", rotation=25)
        for i, v in enumerate(vals):
            if v is None:
                continue
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Plot CH5 max-step ablation bars from metrics_master.csv")
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=root / "artifacts/eval_unified/metrics_master.csv",
        help="Path to metrics_master.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "artifacts/eval_unified/plots",
        help="Directory for PNG output",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default="ALL",
        help='Comma-separated dataset names, or ALL for all datasets present after filtering (e.g. "3DPW-TEST,3DPW-OCC-TEST")',
    )
    ap.add_argument(
        "--eval-mode",
        choices=("full", "smoke", "any"),
        default="full",
        help="full: only rows with empty limit_batches; smoke: only limited batches; any: both, prefer full on tie",
    )
    ap.add_argument(
        "--basename",
        type=str,
        default="ch5_maxstep_bars",
        help="Output files: {basename}_{dataset}.png (slashes replaced)",
    )
    args = ap.parse_args()

    if not args.metrics_csv.is_file():
        raise SystemExit(f"Missing CSV: {args.metrics_csv}")

    rows = load_ch5_rows(args.metrics_csv)
    if not rows:
        raise SystemExit("No NViT ch5 rows found in CSV.")

    ds_filter: set[str] | None = None
    if args.datasets.strip().upper() != "ALL":
        ds_filter = {s.strip() for s in args.datasets.split(",") if s.strip()}

    picked = pick_best_per_group_dataset(rows, eval_mode=args.eval_mode, datasets_filter=ds_filter)
    if not picked:
        raise SystemExit(
            "No rows after filtering. Try --eval-mode any or --eval-mode smoke, "
            "or broaden --datasets."
        )

    datasets = sorted({(r.get("dataset") or "").strip() for r in picked})
    if args.datasets.strip().upper() != "ALL" and ds_filter is not None:
        datasets = sorted(ds_filter & set(datasets))

    suffix = f" ({args.eval_mode} eval)"
    for ds in datasets:
        safe = ds.replace("/", "_")
        out = args.out_dir / f"{args.basename}_{safe}.png"
        plot_dataset(picked, ds, out, suffix)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
