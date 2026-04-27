#!/usr/bin/env python3
"""
基线（METRO / Mesh Graphormer 文献数）、可选本机 HMR2 评测 json、与 Ch6 composite best NViT 的对比表 + 小图。

- 本机 HMR2：需先用与 NViT 相同的 `standard_eval` 管线对 checkpoint 产出 json，再传 `--hmr2-json`。
- composite best 的选取与 `artifacts/ch6_best_vs_baselines.py` / `metrics_master.csv` 一致。

输出（默认）:
  outputs/eval_global/Ch6A/ch6_baseline_vs_nvit_best.{csv,md,png}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_ART = _REPO / "artifacts"
if str(_ART) not in sys.path:
    sys.path.insert(0, str(_ART))

import ch6_best_vs_baselines as c6  # noqa: E402

DEFAULT_NVIT_PARAMS_M = float(os.environ.get("NVIT_CH6_PARAMS_M", str(c6.DEFAULT_NVIT_PARAMS_M)))


def _row_from_baseline(b: dict) -> dict:
    return {
        "Method": b["method"],
        "Params_M": b["params_m"],
        "3DPW_MPJPE": b.get("dpw_mpjpe"),
        "3DPW_PA_MPJPE": b.get("dpw_pa_mpjpe"),
        "H36M_MPJPE": b.get("h36m_mpjpe"),
        "H36M_PA_MPJPE": b.get("h36m_pa_mpjpe"),
        "3DPW_MPVE": b.get("dpw_mpve"),
        "Source": "文献/论文表",
    }


def _row_nvit(best: dict, params_m: float) -> dict:
    res = best.get("results") or {}
    d3 = res.get("3DPW-TEST") or {}
    h36 = res.get("H36M-VAL-P2") or {}
    exp = str(best.get("experiment") or "")
    ck = str(best.get("checkpoint") or "")
    j = str(best.get("json_path") or "")
    return {
        "Method": f"NViT (Ch6, composite-best)",
        "Params_M": params_m,
        "3DPW_MPJPE": d3.get("mode_mpjpe"),
        "3DPW_PA_MPJPE": d3.get("mode_re"),
        "H36M_MPJPE": h36.get("mode_mpjpe"),
        "H36M_PA_MPJPE": h36.get("mode_re"),
        "3DPW_MPVE": "—",
        "Source": f"eval: {j}; exp={exp}; ckpt={ck}",
    }


def _row_hmr2_from_json(
    jpath: Path, label: str, params_m: float, ckpt_note: str = ""
) -> dict:
    with jpath.open(encoding="utf-8") as f:
        j = json.load(f)
    r = j.get("results") or {}
    d3 = r.get("3DPW-TEST") or {}
    h36 = r.get("H36M-VAL-P2") or {}
    src = f"eval json: {jpath}"
    if ckpt_note:
        src += f" | ckpt: {ckpt_note}"
    return {
        "Method": label,
        "Params_M": params_m,
        "3DPW_MPJPE": d3.get("mode_mpjpe"),
        "3DPW_PA_MPJPE": d3.get("mode_re"),
        "H36M_MPJPE": h36.get("mode_mpjpe"),
        "H36M_PA_MPJPE": h36.get("mode_re"),
        "3DPW_MPVE": "—",
        "Source": src,
    }


def _md_table(rows: list[dict]) -> str:
    cols = [
        "Method",
        "Params_M",
        "3DPW_MPJPE",
        "3DPW_PA_MPJPE",
        "H36M_MPJPE",
        "H36M_PA_MPJPE",
    ]
    lines: list[str] = []
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        line = (
            f"| {r['Method']} | {float(r['Params_M']):.3f} | "
            f"{_fmt(r['3DPW_MPJPE'])} | {_fmt(r['3DPW_PA_MPJPE'])} | "
            f"{_fmt(r['H36M_MPJPE'])} | {_fmt(r['H36M_PA_MPJPE'])} |"
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def _fmt(x) -> str:
    if x is None or x == "—":
        return "—"
    if isinstance(x, (int, float)):
        return f"{float(x):.2f}"
    return str(x)


def _plot_bars(rows: list[dict], out_png: Path) -> None:
    methods: list[str] = []
    for r in rows:
        m = str(r["Method"])
        if m.startswith("NViT"):
            methods.append("NViT (best)")
        else:
            methods.append(m)
    # 4 个子图: 2x2
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.2))
    metrics = [
        ("3DPW MPJPE ↓ (mm)", "3DPW_MPJPE", axs[0, 0]),
        ("3DPW PA-MPJPE ↓ (mm)", "3DPW_PA_MPJPE", axs[0, 1]),
        ("H36M MPJPE ↓ (mm)", "H36M_MPJPE", axs[1, 0]),
        ("H36M PA-MPJPE ↓ (mm)", "H36M_PA_MPJPE", axs[1, 1]),
    ]
    n_m = max(1, len(methods))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_m)]
    x = np.arange(len(methods))
    w = 0.55
    for title, key, ax in metrics:
        vals: list[float] = []
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
            else:
                vals.append(float("nan"))
        b = ax.bar(x, vals, width=w, color=colors, edgecolor="0.2", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=10, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for p in b:
            h = float(p.get_height())
            if h == h:
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    h,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    fig.suptitle("Baseline vs NViT Ch6 composite-best (2D-3D)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", type=Path, default=_REPO / "artifacts" / "eval_unified" / "metrics_master.csv")
    ap.add_argument("--rank-metric-3d", choices=("mode_re", "mode_mpjpe"), default="mode_re")
    ap.add_argument(
        "--nvit-params-m",
        type=float,
        default=DEFAULT_NVIT_PARAMS_M,
        help="Params (M) 用于 NViT 行。默认 208.128 或环境 NVIT_CH6_PARAMS_M",
    )
    ap.add_argument(
        "--hmr2-json",
        type=Path,
        default=None,
        help="本机 HMR2 系模型 standard_eval 的 json，与 Ch6/文献对比（需先对同一 ckpt 跑评测）",
    )
    ap.add_argument("--hmr2-label", type=str, default="HMR2 (4DH multirun)")
    ap.add_argument(
        "--hmr2-params-m",
        type=float,
        default=213.0,
        help="HMR2 行 Params (M) 粗估值，可改为你用 thop/统计 的真实值",
    )
    ap.add_argument(
        "--hmr2-ckpt",
        type=str,
        default="",
        help="仅写入 md/csv 的说明串，方便溯源",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "outputs" / "eval_global" / "Ch6A",
    )
    ap.add_argument("--out-stem", type=str, default="ch6_baseline_vs_nvit_best")
    args = ap.parse_args()

    if not args.metrics_csv.is_file():
        raise SystemExit(f"找不到 {args.metrics_csv}")

    records = c6.load_ch6_records_from_master(args.metrics_csv, c6.RANK_DATASETS)
    best = c6.compute_composite_best(records, args.rank_metric_3d)
    if not best:
        raise SystemExit("无法从 metrics 得到 composite best")

    rows: list[dict] = []
    for b in c6.BASELINES:
        rows.append(_row_from_baseline(b))
    if args.hmr2_json:
        if not args.hmr2_json.is_file():
            raise SystemExit(f"找不到 --hmr2-json: {args.hmr2_json}")
        rows.append(
            _row_hmr2_from_json(
                args.hmr2_json,
                args.hmr2_label,
                float(args.hmr2_params_m),
                ckpt_note=(args.hmr2_ckpt or str(args.hmr2_json)),
            )
        )
    rows.append(_row_nvit(best, args.nvit_params_m))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_stem
    out_csv = args.output_dir / f"{stem}.csv"
    out_md = args.output_dir / f"{stem}.md"
    out_png = args.output_dir / f"{stem}.png"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Method",
                "Params_M",
                "3DPW_MPJPE",
                "3DPW_PA_MPJPE",
                "H36M_MPJPE",
                "H36M_PA_MPJPE",
                "3DPW_MPVE",
                "Source",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "Method": r["Method"],
                    "Params_M": f'{float(r["Params_M"]):.6f}',
                    "3DPW_MPJPE": _fmt(r.get("3DPW_MPJPE")) if r.get("3DPW_MPJPE") is not None else "",
                    "3DPW_PA_MPJPE": _fmt(r.get("3DPW_PA_MPJPE")) if r.get("3DPW_PA_MPJPE") is not None else "",
                    "H36M_MPJPE": _fmt(r.get("H36M_MPJPE")) if r.get("H36M_MPJPE") is not None else "",
                    "H36M_PA_MPJPE": _fmt(r.get("H36M_PA_MPJPE")) if r.get("H36M_PA_MPJPE") is not None else "",
                    "3DPW_MPVE": str(r.get("3DPW_MPVE", "")),
                    "Source": r.get("Source", ""),
                }
            )

    md = (
        "# Ch6: 文献 / 本机 HMR2(可选) / NViT (composite best)\n\n"
        f"- `rank_metric_3d`: **{args.rank_metric_3d}**\n"
        f"- NViT checkpoint: `{best.get('checkpoint', '')}`\n"
        f"- NViT json: `{best.get('json_path', '')}`\n"
    )
    if args.hmr2_json:
        md += f"- HMR2 eval json: `{args.hmr2_json}`\n"
        if args.hmr2_ckpt:
            md += f"- HMR2 ckpt (note): `{args.hmr2_ckpt}`\n"
    md += "\n" + _md_table(
        [
            {k: r[k] for k in ["Method", "Params_M", "3DPW_MPJPE", "3DPW_PA_MPJPE", "H36M_MPJPE", "H36M_PA_MPJPE"]}
            for r in rows
        ]
    )
    out_md.write_text(md, encoding="utf-8")
    _plot_bars(rows, out_png)

    print(out_csv)
    print(out_md)
    print(out_png)


if __name__ == "__main__":
    main()
