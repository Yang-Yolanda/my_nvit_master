#!/usr/bin/env python3
"""
Plot Ch6 per-dataset metrics with baselines.

Default behavior:
- NViT: auto-pick composite-best checkpoint from metrics_master.csv.
- METRO / Mesh Graphormer: use literature values from ch6_best_vs_baselines.py.

Local verification behavior:
- If --metro-json / --meshgraphormer-json is provided, overwrite that method with local eval JSON.
- If --hmr2-baseline-json is provided, add a fourth curve (your 4D-Humans HMR2 ckpt eval, same json schema as standard_eval).
- If --local-baseline-csv is provided, overwrite matching method+dataset cells.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_ART = _REPO / "artifacts"
if str(_ART) not in sys.path:
    sys.path.insert(0, str(_ART))

import ch6_best_vs_baselines as c6  # noqa: E402

DEFAULT_DATASETS = [
    "3DPW-TEST",
    "H36M-VAL-P2",
    "COCO-VAL",
    "POSETRACK-VAL",
    "LSP-EXTENDED",
]


def _f(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return None if math.isnan(v) else v
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def _canon_method(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    if "metro" in s:
        return "METRO"
    if "mesh" in s and "graph" in s:
        return "Mesh Graphormer"
    return name.strip()


def _init_table(datasets: list[str]) -> dict[str, dict[str, dict[str, float | None]]]:
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for b in c6.BASELINES:
        m = str(b["method"])
        out[m] = {d: {"MPJPE_mm": None, "PA_MPJPE_mm": None, "KPL2": None} for d in datasets}
        out[m]["3DPW-TEST"]["MPJPE_mm"] = _f(b.get("dpw_mpjpe"))
        out[m]["3DPW-TEST"]["PA_MPJPE_mm"] = _f(b.get("dpw_pa_mpjpe"))
        out[m]["H36M-VAL-P2"]["MPJPE_mm"] = _f(b.get("h36m_mpjpe"))
        out[m]["H36M-VAL-P2"]["PA_MPJPE_mm"] = _f(b.get("h36m_pa_mpjpe"))
    return out


def _load_best_nvit_json(metrics_csv: Path, rank_metric_3d: str) -> tuple[Path, str]:
    records = c6.load_ch6_records_from_master(metrics_csv, c6.RANK_DATASETS)
    if not records:
        raise SystemExit(f"no valid ch6 records in {metrics_csv}")
    best = c6.compute_composite_best(records, rank_metric_3d)
    if not best:
        raise SystemExit("cannot compute composite best")
    j = Path(str(best.get("json_path") or ""))
    if not j.is_file():
        raise SystemExit(f"best json missing: {j}")
    exp = str(best.get("experiment") or j.stem)
    return j, exp


def _apply_eval_json(
    table: dict[str, dict[str, dict[str, float | None]]],
    *,
    method: str,
    eval_json: Path,
    datasets: list[str],
    use_canon: bool = True,
) -> str:
    """将 standard_eval 风格 json 写入 table。use_canon=False 时保留图例名（如本机 HMR2 baseline）。"""
    with eval_json.open(encoding="utf-8") as f:
        obj = json.load(f)
    results = obj.get("results") or {}
    m = _canon_method(method) if use_canon else str(method)
    if m not in table:
        table[m] = {d: {"MPJPE_mm": None, "PA_MPJPE_mm": None, "KPL2": None} for d in datasets}
    for d in datasets:
        rr = results.get(d) or {}
        table[m][d]["MPJPE_mm"] = _f(rr.get("mode_mpjpe"))
        table[m][d]["PA_MPJPE_mm"] = _f(rr.get("mode_re"))
        table[m][d]["KPL2"] = _f(rr.get("mode_kpl2"))
    return m


def _apply_local_csv(
    table: dict[str, dict[str, dict[str, float | None]]],
    *,
    local_csv: Path,
    datasets: list[str],
) -> None:
    with local_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            method = _canon_method(str(row.get("Method", "")).strip())
            ds = str(row.get("Dataset", "")).strip()
            if not method or not ds:
                continue
            if ds not in datasets:
                continue
            if method not in table:
                table[method] = {d: {"MPJPE_mm": None, "PA_MPJPE_mm": None, "KPL2": None} for d in datasets}
            table[method][ds]["MPJPE_mm"] = _f(row.get("MPJPE_mm"))
            table[method][ds]["PA_MPJPE_mm"] = _f(row.get("PA_MPJPE_mm"))
            table[method][ds]["KPL2"] = _f(row.get("KPL2"))


def _write_csv(
    out_csv: Path,
    table: dict[str, dict[str, dict[str, float | None]]],
    methods: list[str],
    datasets: list[str],
) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Method", "Dataset", "MPJPE_mm", "PA_MPJPE_mm", "KPL2"])
        w.writeheader()
        for m in methods:
            for d in datasets:
                row = table[m][d]
                w.writerow(
                    {
                        "Method": m,
                        "Dataset": d,
                        "MPJPE_mm": "" if row["MPJPE_mm"] is None else f'{row["MPJPE_mm"]:.6f}',
                        "PA_MPJPE_mm": "" if row["PA_MPJPE_mm"] is None else f'{row["PA_MPJPE_mm"]:.6f}',
                        "KPL2": "" if row["KPL2"] is None else f'{row["KPL2"]:.6f}',
                    }
                )


def _grouped_bar(
    ax,
    *,
    datasets: list[str],
    methods: list[str],
    table: dict[str, dict[str, dict[str, float | None]]],
    key: str,
    ylabel: str,
    title: str,
    higher_better: bool,
) -> None:
    x = np.arange(len(datasets))
    n = max(1, len(methods))
    width = min(0.36, 0.8 / n)
    offsets = (np.arange(n) - (n - 1) / 2.0) * width

    for i, m in enumerate(methods):
        vals = np.array([np.nan if table[m][d][key] is None else float(table[m][d][key]) for d in datasets], dtype=float)
        ok = np.isfinite(vals)
        if ok.any():
            bars = ax.bar(x[ok] + offsets[i], vals[ok], width=width, label=m)
            for b in bars:
                y = float(b.get_height())
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    y,
                    f"{y:.3f}" if key == "KPL2" else f"{y:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    arrow = "↑" if higher_better else "↓"
    ax.set_title(f"{title} ({arrow})")
    ax.grid(axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Ch6 per-dataset metrics with baseline overlays.")
    ap.add_argument("--metrics-csv", type=Path, default=_REPO / "artifacts" / "eval_unified" / "metrics_master.csv")
    ap.add_argument("--rank-metric-3d", choices=("mode_re", "mode_mpjpe"), default="mode_re")
    ap.add_argument("--nvit-json", type=Path, default=None, help="optional override for NViT eval json")
    ap.add_argument("--nvit-label", type=str, default=None, help="label for NViT curve")
    ap.add_argument("--metro-json", type=Path, default=None, help="local METRO eval json (same schema as standard_eval output)")
    ap.add_argument("--meshgraphormer-json", type=Path, default=None, help="local Mesh Graphormer eval json")
    ap.add_argument(
        "--hmr2-baseline-json",
        type=Path,
        default=None,
        help="本机 HMR2/HMR2 系 checkpoint 的 standard_eval 输出 json（与 NViT 同管线评测后再对比）",
    )
    ap.add_argument(
        "--hmr2-baseline-label",
        type=str,
        default="HMR2 (4DH finetune)",
        help="图上与 CSV 中的方法名；勿与 METRO 混用",
    )
    ap.add_argument("--local-baseline-csv", type=Path, default=None, help="optional overrides: Method,Dataset,MPJPE_mm,PA_MPJPE_mm,KPL2")
    ap.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS), help="comma-separated dataset order")
    ap.add_argument("--output-dir", type=Path, default=_REPO / "outputs" / "eval_global" / "Ch6A")
    ap.add_argument("--out-stem", type=str, default="ch6_dataset_effects_with_baselines")
    args = ap.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    table = _init_table(datasets)

    if args.nvit_json:
        nvit_json = args.nvit_json
        if not nvit_json.is_file():
            raise SystemExit(f"nvit json not found: {nvit_json}")
        nvit_exp = args.nvit_label or f"NViT ({nvit_json.stem})"
    else:
        nvit_json, exp = _load_best_nvit_json(args.metrics_csv, args.rank_metric_3d)
        nvit_exp = args.nvit_label or f"NViT best ({exp})"
    _apply_eval_json(
        table, method=nvit_exp, eval_json=nvit_json, datasets=datasets, use_canon=False
    )

    if args.metro_json:
        if not args.metro_json.is_file():
            raise SystemExit(f"metro json not found: {args.metro_json}")
        _apply_eval_json(
            table, method="METRO", eval_json=args.metro_json, datasets=datasets, use_canon=True
        )
    if args.meshgraphormer_json:
        if not args.meshgraphormer_json.is_file():
            raise SystemExit(f"meshgraphormer json not found: {args.meshgraphormer_json}")
        _apply_eval_json(
            table,
            method="Mesh Graphormer",
            eval_json=args.meshgraphormer_json,
            datasets=datasets,
            use_canon=True,
        )
    hmr2_key: str | None = None
    if args.hmr2_baseline_json:
        if not args.hmr2_baseline_json.is_file():
            raise SystemExit(f"hmr2-baseline json not found: {args.hmr2_baseline_json}")
        hmr2_key = _apply_eval_json(
            table,
            method=args.hmr2_baseline_label,
            eval_json=args.hmr2_baseline_json,
            datasets=datasets,
            use_canon=False,
        )
    if args.local_baseline_csv:
        if not args.local_baseline_csv.is_file():
            raise SystemExit(f"local baseline csv not found: {args.local_baseline_csv}")
        _apply_local_csv(table, local_csv=args.local_baseline_csv, datasets=datasets)

    ordered: list[str] = []
    for m in ("METRO", "Mesh Graphormer"):
        if m in table:
            ordered.append(m)
    if hmr2_key and hmr2_key in table:
        ordered.append(hmr2_key)
    if nvit_exp in table and nvit_exp not in ordered:
        ordered.append(nvit_exp)
    for m in sorted(table.keys()):
        if m not in ordered:
            ordered.append(m)
    methods = ordered

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / f"{args.out_stem}.csv"
    out_png = args.output_dir / f"{args.out_stem}.png"
    _write_csv(out_csv, table, methods, datasets)

    fig, axes = plt.subplots(1, 3, figsize=(4.0 + 3.0 * max(1, len(methods)), 4.6))
    _grouped_bar(
        axes[0],
        datasets=datasets,
        methods=methods,
        table=table,
        key="MPJPE_mm",
        ylabel="Error (mm)",
        title="MPJPE",
        higher_better=False,
    )
    _grouped_bar(
        axes[1],
        datasets=datasets,
        methods=methods,
        table=table,
        key="PA_MPJPE_mm",
        ylabel="Error (mm)",
        title="PA-MPJPE",
        higher_better=False,
    )
    _grouped_bar(
        axes[2],
        datasets=datasets,
        methods=methods,
        table=table,
        key="KPL2",
        ylabel="Score",
        title="KPL2",
        higher_better=True,
    )
    fig.suptitle("Ch6 per-dataset effects with baselines")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] csv: {out_csv}")
    print(f"[ok] png: {out_png}")
    print(f"[nvit] {nvit_json}")
    if args.metro_json:
        print(f"[local metro] {args.metro_json}")
    if args.meshgraphormer_json:
        print(f"[local mesh graphormer] {args.meshgraphormer_json}")
    if args.local_baseline_csv:
        print(f"[local baseline csv] {args.local_baseline_csv}")
    if args.hmr2_baseline_json:
        print(f"[hmr2 baseline] {args.hmr2_baseline_json} label={args.hmr2_baseline_label!r}")


if __name__ == "__main__":
    main()
