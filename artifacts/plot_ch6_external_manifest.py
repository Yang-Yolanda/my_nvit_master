#!/usr/bin/env python3
"""
根据 manifest（Ch6 best + 若干本机标准评测 json）写对比表/图。

manifest 由 run_ch6_external_mnt_nvit.sh 生成，示例:
{
  "ch6": {
    "label": "NViT (Ch6 best, step 492k)",
    "eval_json": "artifacts/.../ch6_...json",
    "checkpoint": "/mnt/.../step_492000.ckpt",
    "params_m": 208.128
  },
  "rows": [
    {
      "kind": "external",
      "label": "ablation hmr2_mid",
      "eval_json": "artifacts/.../external_xxx.json",
      "params_m": 0,
      "checkpoint": "/.../a.ckpt",
      "status": "ok"
    },
    { "kind": "external", "label": "hmr2_mid_heavy", "status": "skipped_pth", "note": "..." }
  ],
  "output_stem": "ch6_external_vs_ch6best"
}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _row_from_eval_json(
    jpath: Path, label: str, params_m: float, ckpt: str, source: str
) -> dict:
    with jpath.open(encoding="utf-8") as f:
        j = json.load(f)
    r = j.get("results") or {}
    d3 = r.get("3DPW-TEST") or {}
    h36 = r.get("H36M-VAL-P2") or {}
    src = source
    if ckpt:
        src += f" | ckpt: {ckpt}"
    return {
        "Method": label,
        "Params_M": float(params_m),
        "3DPW_MPJPE": d3.get("mode_mpjpe"),
        "3DPW_PA_MPJPE": d3.get("mode_re"),
        "H36M_MPJPE": h36.get("mode_mpjpe"),
        "H36M_PA_MPJPE": h36.get("mode_re"),
        "3DPW_MPVE": "—",
        "Source": src,
    }


def _row_ch6_nvit(
    ch6: dict, eval_path: Path, ch6_lbl: str, params_m: float, src_extra: str
) -> dict:
    with eval_path.open(encoding="utf-8") as f:
        j = json.load(f)
    r = j.get("results") or {}
    d3 = r.get("3DPW-TEST") or {}
    h36 = r.get("H36M-VAL-P2") or {}
    ck = str(ch6.get("checkpoint") or "")
    return {
        "Method": ch6_lbl,
        "Params_M": float(params_m),
        "3DPW_MPJPE": d3.get("mode_mpjpe"),
        "3DPW_PA_MPJPE": d3.get("mode_re"),
        "H36M_MPJPE": h36.get("mode_mpjpe"),
        "H36M_PA_MPJPE": h36.get("mode_re"),
        "3DPW_MPVE": "—",
        "Source": f"eval: {eval_path} | {src_extra} | ckpt={ck}",
    }


def _row_skipped(label: str, params_m: float, note: str) -> dict:
    return {
        "Method": f"{label} (PTH, skip)",
        "Params_M": float(params_m),
        "3DPW_MPJPE": None,
        "3DPW_PA_MPJPE": None,
        "H36M_MPJPE": None,
        "H36M_PA_MPJPE": None,
        "3DPW_MPVE": "—",
        "Source": note,
    }


def _fmt(x) -> str:
    if x is None or x == "—":
        return "—"
    if isinstance(x, (int, float)) and (x == x):
        return f"{float(x):.2f}"
    return str(x)


def _md_table(rows: list[dict], cols: list[str]) -> str:
    lines: list[str] = []
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        line = "| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _row_order_key(r: dict) -> tuple[int, str]:
    m = str(r.get("Method", "")).lower()
    s = str(r.get("Source", "")).lower()
    if "baseline" in m or "epoch=35-step=1000000" in m or "epoch=35-step=1000000" in s:
        return (0, m)
    if "mid_heavy" in m or "mid_heavy" in s:
        return (1, m)
    if "nvit" in m:
        return (2, m)
    return (9, m)


def _plot(rows: list[dict], out_png: Path) -> None:
    rows = sorted(rows, key=_row_order_key)

    methods: list[str] = []
    for r in rows:
        m = str(r["Method"])
        methods.append(m if len(m) < 32 else m[:29] + "...")
    n_m = max(1, len(methods))
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.8))
    metrics = [
        ("3DPW MPJPE ↓ (mm)", "3DPW_MPJPE", axs[0, 0]),
        ("3DPW PA-MPJPE ↓ (mm)", "3DPW_PA_MPJPE", axs[0, 1]),
        ("H36M MPJPE ↓ (mm)", "H36M_MPJPE", axs[1, 0]),
        ("H36M PA-MPJPE ↓ (mm)", "H36M_PA_MPJPE", axs[1, 1]),
    ]
    # 与 chapter06「各项指标对比图」保持一致：baseline(蓝) / heavy(橙) / nvit(绿)
    def _color_for_row(r: dict) -> str:
        m = str(r.get("Method", "")).lower()
        s = str(r.get("Source", "")).lower()
        if "baseline" in m or "epoch=35-step=1000000" in m or "epoch=35-step=1000000" in s:
            return "#1f77b4"
        if "mid_heavy" in m or "mid_heavy" in s:
            return "#ff7f0e"
        if "nvit" in m:
            return "#2ca02c"
        return "#9467bd"

    colors = [_color_for_row(r) for r in rows]
    x = np.arange(len(methods))
    w = 0.55
    for title, key, ax in metrics:
        vals: list[float] = []
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float)) and v == v:
                vals.append(float(v))
            else:
                vals.append(float("nan"))
        b = ax.bar(x, vals, width=w, color=colors, edgecolor="0.2", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=12, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for p in b:
            h = float(p.get_height())
            if h == h and not (h != h):
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    h,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
    # 按论文图风格：不使用 figure 级大标题，仅保留子图标题
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "outputs" / "eval_global" / "Ch6A",
    )
    args = ap.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"缺少 manifest: {args.manifest}")
    m = json.loads(args.manifest.read_text(encoding="utf-8"))
    ch6 = m.get("ch6") or {}
    stem = str(m.get("output_stem") or "ch6_external_vs_ch6best")
    ch6_eval = Path(ch6["eval_json"])
    if not ch6_eval.is_file():
        raise SystemExit(f"ch6 eval_json 不存在: {ch6_eval}")

    all_rows: list[dict] = [
        _row_ch6_nvit(
            ch6,
            ch6_eval,
            str(ch6.get("row_label") or "NViT (Ch6 best)"),
            float(ch6.get("params_m") or 0),
            "composite",
        )
    ]

    for r in m.get("rows") or []:
        st = str(r.get("status") or "ok")
        if st == "skipped_pth":
            all_rows.append(
                _row_skipped(
                    str(r.get("label") or "ext"),
                    float(r.get("params_m") or 0),
                    str(
                        r.get("note")
                        or "PTH: 需与剪枝后结构一致的 Lightning/建网再 standard_eval"
                    ),
                )
            )
            continue
        ej = r.get("eval_json")
        if not ej:
            all_rows.append(
                _row_skipped(
                    str(r.get("label") or "ext"),
                    float(r.get("params_m") or 0),
                    str(r.get("note") or "无 eval json"),
                )
            )
            continue
        p = Path(ej)
        if not p.is_file():
            all_rows.append(
                _row_skipped(
                    f"{r.get('label', 'ext')}",
                    float(r.get("params_m") or 0),
                    f"eval json 缺失: {p}",
                )
            )
            continue
        all_rows.append(
            _row_from_eval_json(
                p,
                str(r.get("label") or p.stem),
                float(r.get("params_m") or 0),
                str(r.get("checkpoint") or ""),
                f"来源: {r.get('source','external')}",
            )
        )

    all_rows = sorted(all_rows, key=_row_order_key)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{stem}.csv"
    out_md = out_dir / f"{stem}.md"
    out_png = out_dir / f"{stem}.png"

    cols = [
        "Method",
        "Params_M",
        "3DPW_MPJPE",
        "3DPW_PA_MPJPE",
        "H36M_MPJPE",
        "H36M_PA_MPJPE",
        "3DPW_MPVE",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=cols + ["Source"],
        )
        w.writeheader()
        for r in all_rows:
            row: dict = {k: _fmt(r.get(k)) for k in cols}
            row["Source"] = str(r.get("Source", ""))
            w.writerow(row)

    md = (
        f"# {stem}\n\n- manifest: `{args.manifest}`\n\n"
        + _md_table(all_rows, cols)
        + "\n## Source\n\n"
    )
    for r in all_rows:
        md += f"- **{r['Method']}**: {r.get('Source','')}\n"
    out_md.write_text(md, encoding="utf-8")
    _plot(all_rows, out_png)

    print(str(out_csv))
    print(str(out_md))
    print(str(out_png))


if __name__ == "__main__":
    main()
