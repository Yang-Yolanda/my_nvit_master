#!/usr/bin/env python3
"""
第 6 章图表/表格一键生成：读 metrics_master、eval json、（可选）layer_metrics JSON。
落盘: <workspace>/0228/图表/chapter06/

运行:
  python3 artifacts/generate_chapter06_artifacts.py
  LAYER_METRICS_JSON=/path/to/layer_metrics_Control.json python3 ...
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

# Matplotlib 非交互
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- 与 ch6_best_vs_baselines 一致 ----
ART = Path(__file__).resolve().parent
REPO = ART.parent
if str(ART) not in sys.path:
    sys.path.insert(0, str(ART))
import ch6_best_vs_baselines as c6  # noqa: E402

RANK_3D = "mode_re"
RANK_DATASETS = c6.RANK_DATASETS
OUTDIR = Path("/cpfs_infra/shared/yangz/0228/图表/chapter06")
RUN_ID = "ch6_2026-04-17_13-28-24"
MASTER = REPO / "artifacts" / "eval_unified" / "metrics_master.csv"
EXAMPLE_JSON = (
    REPO
    / "artifacts/eval_unified/json/nvit/ch6_2026-04-17_13-28-24_step_360000.json"
)
# 可选: 3DPW 取向 checkpoint 的 eval json（与 EXAMPLE_JSON 的 composite 区分；论文「双行 NViT」时设置）
# 例: ch6_2026-04-17_13-28-24_step_274000.json（以实际产物为准）
NVIT_3DPW_ORIENTED_JSON = os.environ.get("CH6_NVIT_3DPW_JSON")
DEFAULT_NVIT_PARAMS = float(os.environ.get("NVIT_CH6_PARAMS_M", "208.128"))
DEFAULT_LAYER_JSON = (
    REPO
    / "outputs/eval_global/Ch6A/ch6_paper_layerwise/layer_metrics_Control.json"
)
# ch6 主 run 最优（composite 对应 step=492000）的 layer 诊断，用于「各项指标对比图」NViT 柱
CH6_BEST_STEP492K_LAYER = (
    REPO
    / "outputs/eval_global/Ch6A/ch6_main_step492000/layer_metrics_Control.json"
)

DPI = 300


def load_run_records() -> list[dict]:
    all_recs = c6.load_ch6_records_from_master(MASTER, RANK_DATASETS)
    out = [r for r in all_recs if RUN_ID in (r.get("json_path") or "")]
    out.sort(key=lambda r: (r.get("step") is None, r.get("step") or -1))
    return out


def savefig_stem(path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path_no_ext) + ".png", dpi=DPI, bbox_inches="tight")
    plt.savefig(str(path_no_ext) + ".pdf", bbox_inches="tight")
    plt.close()


def write_md(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def booktabs_table(
    headers: list[str], rows: list[list[str]], caption: str, label: str
) -> str:
    n = len(headers)
    spec = "l" + "r" * (n - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ---------- Prompt 1 + filtered CSV ----------
def build_step_table(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for r in records:
        st = r.get("step")
        if st is None:
            continue
        res = r.get("results") or {}
        d3 = res.get("3DPW-TEST") or {}
        h36 = res.get("H36M-VAL-P2") or {}
        rows.append(
            {
                "step": st,
                "3dpw_mpjpe": d3.get("mode_mpjpe"),
                "3dpw_pa": d3.get("mode_re"),
                "h36m_mpjpe": h36.get("mode_mpjpe"),
                "h36m_pa": h36.get("mode_re"),
            }
        )
    rows.sort(key=lambda x: x["step"])
    return rows


def plot_step_dynamics(rows: list[dict]) -> None:
    if not rows:
        return
    steps = np.array([r["step"] for r in rows], dtype=float)
    s0, s1 = float(steps.min()), float(steps.max())
    mark_steps = [10_000, 100_000, 274_000, 360_000, 428_000, 438_000, 492_000, 504_000]
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    titles = [
        ("3DPW MPJPE (mm) ↓", "3dpw_mpjpe"),
        ("3DPW PA-MPJPE (mm) ↓", "3dpw_pa"),
        ("H36M MPJPE (mm) ↓", "h36m_mpjpe"),
        ("H36M PA-MPJPE (mm) ↓", "h36m_pa"),
    ]
    for ax, (title, key) in zip(axs.flat, titles):
        y = [r[key] for r in rows]
        ax.set_facecolor("white")
        # 背景区段
        ax.axvspan(max(s0, 2000), min(s1, 20_000), color="#cfe8ff", alpha=0.35, zorder=0)
        ax.axvspan(20_000, min(s1, 150_000), color="#ffe8c8", alpha=0.28, zorder=0)
        if s1 >= 300_000:
            ax.axvspan(300_000, s1, color="#e6e6e6", alpha=0.45, zorder=0)
        ax.plot(steps, y, color="#1f77b4", lw=1.4, zorder=2)
        for ms in mark_steps:
            if s0 - 1 <= ms <= s1 + 1:
                ax.axvline(ms, color="#888", ls="--", lw=0.9, zorder=1)
                j = int(np.abs(steps - ms).argmin())
                if abs(steps[j] - ms) <= 8000:  # 近邻处打点
                    ax.plot(steps[j], y[j], "o", color="red", ms=3.5, zorder=3)
                    ax.annotate(
                        f"{int(steps[j])//1000}k" if steps[j] >= 1000 else f"{int(steps[j])}",
                        (steps[j], y[j]),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=6,
                        color="#333",
                    )
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25, ls=":", lw=0.7)
    for ax in axs.flat:
        ax.set_xlim(s0, s1)
    axs[1, 0].set_xlabel("Global training step")
    axs[1, 1].set_xlabel("Global training step")
    fig.suptitle(
        "Training-step dynamics (mm; lower is better) — run %s" % RUN_ID, fontsize=11, y=1.02
    )
    fig.tight_layout()
    savefig_stem(OUTDIR / "ch6_step_dynamics")
    # legend 说明
    with open(OUTDIR / "ch6_step_legend.txt", "w", encoding="utf-8") as f:
        f.write(
            "Shaded: early 2k–20k, mid 20k–150k, late ≥300k (within plotted [min step, max step]).\n"
        )


def plot_pareto(rows: list[dict]) -> None:
    if not rows:
        return
    xs = [r["h36m_pa"] for r in rows]
    ys = [r["3dpw_pa"] for r in rows]
    stp = [r["step"] for r in rows]
    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(xs, ys, c="#7f7f7f", s=12, alpha=0.5, zorder=1, label="checkpoints")
    highlight = {274_000, 360_000, 492_000}
    for t in highlight:
        j = min(range(len(stp)), key=lambda i: abs((stp[i] or 0) - t))
        if stp[j] is None or abs(stp[j] - t) > 16_000:
            continue
        plt.scatter(
            [xs[j]], [ys[j]], c="red", s=55, zorder=3, edgecolors="k", linewidths=0.4
        )
        plt.annotate(
            f"{t//1000}k (step≈{stp[j]//1000}k)",
            (xs[j], ys[j]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xlabel("H36M PA-MPJPE (mm) ↓ (better ←)")
    plt.ylabel("3DPW PA-MPJPE (mm) ↓ (better ↓)")
    plt.title("3DPW vs H36M trade-off (PA-MPJPE, mm) — " + RUN_ID)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig_stem(OUTDIR / "ch6_pareto_3dpw_vs_h36m")


# ---------- Prompt 2 ----------
def per_metric_ranks(
    records: list[dict], key_ds: str, mkey: str
) -> dict[int | None, int]:
    items: list[tuple[int, float, int | None]] = []
    for i, r in enumerate(records):
        st = r.get("step")
        v = (r.get("results") or {}).get(key_ds, {}).get(mkey)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        items.append((i, float(v), st))
    items.sort(key=lambda x: (x[1], x[0]))
    rankmap: dict[int | None, int] = {}
    for rank, (i, _v, st) in enumerate(items, start=1):
        rankmap[st] = rank
    return rankmap


def build_checkpoint_selection(records: list[dict]) -> list[dict]:
    ranks = c6.composite_rank_matrix(records, RANK_3D)
    r3 = per_metric_ranks(records, "3DPW-TEST", "mode_re")
    rh = per_metric_ranks(records, "H36M-VAL-P2", "mode_re")
    comp = c6.compute_composite_best(records, RANK_3D)
    comp_step = (comp or {}).get("step")
    def _pa(r: dict, ds: str) -> float | None:
        v = (r.get("results") or {}).get(ds, {}).get("mode_re")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)

    c3 = [r for r in records if _pa(r, "3DPW-TEST") is not None]
    best3 = min(c3, key=lambda x: _pa(x, "3DPW-TEST") or 1e9) if c3 else None
    c5 = [r for r in records if _pa(r, "H36M-VAL-P2") is not None]
    besth = min(c5, key=lambda x: _pa(x, "H36M-VAL-P2") or 1e9) if c5 else None
    best3s = best3.get("step") if best3 else None
    besths = besth.get("step") if besth else None
    out: list[dict] = []
    for i, r in enumerate(records):
        st = r.get("step")
        res = r.get("results") or {}
        d3 = res.get("3DPW-TEST", {})
        h36 = res.get("H36M-VAL-P2", {})
        rr = ranks.get(i, {}) if ranks else {}
        rs = int(sum(rr.values())) if rr else -1
        out.append(
            {
                "step": st,
                "3dpw_mpjpe": d3.get("mode_mpjpe", ""),
                "3dpw_pa": d3.get("mode_re", ""),
                "h36m_mpjpe": h36.get("mode_mpjpe", ""),
                "h36m_pa": h36.get("mode_re", ""),
                "rank_3dpw": r3.get(st, ""),
                "rank_h36m": rh.get(st, ""),
                "rank_sum": rs if rs >= 0 else "",
                "is_3dpw_best": "1" if st == best3s else "0",
                "is_h36m_best": "1" if st == besths else "0",
                "is_composite_best": "1" if st == comp_step else "0",
            }
        )
    out.sort(key=lambda x: (x["step"] is None, x["step"] or 0))
    return out


def topk_rank_sum_order(records: list[dict], k: int) -> list[tuple[int, int | None, int]]:
    """返回 [(rank1based, step, rank_sum), ...] rank_sum 升序，并列按 mean_rank 与 ch6 一致用 list_composite_ranked。"""
    ranked = c6.list_composite_ranked(records, RANK_3D, top_k=None)
    return [
        (pos, row.get("step"), int(row.get("rank_sum", -1)))
        for pos, row in enumerate(ranked[:k], start=1)
    ]


# ---------- Prompt 3 ----------
def nearest_rows(rows: list[dict], targets: list[int]) -> list[dict]:
    st = np.array([r["step"] for r in rows], dtype=int)
    out: list[dict] = []
    for t in targets:
        j = int(np.abs(st - t).argmin())
        delta = int(st[j] - t)
        rr = {**rows[j], "target_step": t, "nearest_step": int(st[j]), "step_delta": delta}
        out.append(rr)
    return out


# ---------- Prompt 4: lightweight ----------
def _fmt(x, empty="—"):
    if x is None:
        return empty
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return empty
    if isinstance(x, (int, float)):
        return f"{float(x):.1f}"
    return str(x)


def _load_json_metrics(jpath: Path) -> tuple[dict, dict, str]:
    with open(jpath, encoding="utf-8") as f:
        j = json.load(f)
    r = j.get("results", {})
    d3 = r.get("3DPW-TEST", {})
    h36 = r.get("H36M-VAL-P2", {})
    return d3, h36, jpath.name


def export_lightweight() -> None:
    d3, h36, cname = _load_json_metrics(EXAMPLE_JSON)
    rows: list[dict[str, object]] = [
        {
            "method": "METRO",
            "params_m": 231.8,
            "h36m_mpjpe": 54.0,
            "h36m_pa": 36.7,
            "dpw_mpve": 88.2,
            "dpw_mpjpe": 77.1,
            "dpw_pa": 47.9,
            "note": "文献/论文表",
        },
        {
            "method": "Mesh Graphormer",
            "params_m": 215.7,
            "h36m_mpjpe": 51.2,
            "h36m_pa": 34.5,
            "dpw_mpve": 87.7,
            "dpw_mpjpe": 74.7,
            "dpw_pa": 45.6,
            "note": "文献/论文表",
        },
    ]
    if NVIT_3DPW_ORIENTED_JSON:
        p3 = Path(NVIT_3DPW_ORIENTED_JSON)
        if p3.is_file():
            d3a, h36a, n3 = _load_json_metrics(p3)
            m = re.search(r"step_(\d+)", p3.name)
            stg = f"~{m.group(1)}" if m else ""
            rows.append(
                {
                    "method": "NViT (Ch6, 3DPW-oriented)",
                    "params_m": DEFAULT_NVIT_PARAMS,
                    "h36m_mpjpe": h36a.get("mode_mpjpe"),
                    "h36m_pa": h36a.get("mode_re"),
                    "dpw_mpve": None,
                    "dpw_mpjpe": d3a.get("mode_mpjpe"),
                    "dpw_pa": d3a.get("mode_re"),
                    "note": f"eval: {n3} (3DPW sweet-spot, step {stg} ckpt; same arch as composite)",
                }
            )
    rows.append(
        {
            "method": "NViT (Ch6, composite-best)",
            "params_m": DEFAULT_NVIT_PARAMS,
            "h36m_mpjpe": h36.get("mode_mpjpe"),
            "h36m_pa": h36.get("mode_re"),
            "dpw_mpve": None,
            "dpw_mpjpe": d3.get("mode_mpjpe"),
            "dpw_pa": d3.get("mode_re"),
            "note": f"eval: {cname} (rank-sum over 3DPW-TEST + H36M-VAL-P2; same arch as 3DPW row)",
        }
    )
    csv_path = OUTDIR / "ch6_lightweight_baseline_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "params_m",
                "h36m_mpjpe",
                "h36m_pa",
                "dpw_mpve",
                "dpw_mpjpe",
                "dpw_pa",
                "note",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "method": row["method"],
                    "params_m": row["params_m"],
                    "h36m_mpjpe": _fmt(row["h36m_mpjpe"]),
                    "h36m_pa": _fmt(row["h36m_pa"]),
                    "dpw_mpve": _fmt(row["dpw_mpve"]),
                    "dpw_mpjpe": _fmt(row["dpw_mpjpe"]),
                    "dpw_pa": _fmt(row["dpw_pa"]),
                    "note": row["note"],
                }
            )
    def _esc_method(s: str) -> str:
        return s.replace("_", r"\_").replace("&", r"\&")

    tex_rows: list[list[str]] = [
        [
            _esc_method(str(r["method"])),
            f'{float(r["params_m"]):.1f}',
            _fmt(r.get("h36m_mpjpe")),
            _fmt(r.get("h36m_pa")),
            _fmt(r.get("dpw_mpve")),
            _fmt(r.get("dpw_mpjpe")),
            _fmt(r.get("dpw_pa")),
        ]
        for r in rows
    ]
    if NVIT_3DPW_ORIENTED_JSON and Path(NVIT_3DPW_ORIENTED_JSON).is_file():
        cap_extra = (
            f"\\textbf{{Primary (report) NViT}}: \\texttt{{NViT (Ch6, composite-best)}} from \\texttt{{{EXAMPLE_JSON.name}}} (rank-sum). "
            r"\textbf{Diagnostic/3DPW-oriented row}: \texttt{NViT (Ch6, 3DPW-oriented)}. "
            r"Both differ only in checkpoint; same network---FLOP/Lat/VRAM: Tab.~\\ref{tab:ch6_resource}. "
            f"3DPW json: \\texttt{{{Path(NVIT_3DPW_ORIENTED_JSON).name}}}. "
        )
    else:
        cap_extra = (
            f"NViT row: rank-sum / composite from \\texttt{{{EXAMPLE_JSON.name}}}. "
        )
    tex = booktabs_table(
        [
            "Method",
            "Params (M)",
            "H36M MPJPE",
            "H36M PA",
            "3DPW MPVE",
            "3DPW MPJPE",
            "3DPW PA",
        ],
        tex_rows,
        "Lightweight baselines vs NViT (mm; lower is better for errors). " + cap_extra,
        "tab:ch6_lightweight",
    )
    (OUTDIR / "ch6_lightweight_baseline_table.tex").write_text(tex, encoding="utf-8")
    # 图
    names = [r["method"] for r in rows]
    pr = [float(r["params_m"]) for r in rows]  # type: ignore[arg-type]
    pa3: list[float] = []
    for r in rows:
        v = r.get("dpw_pa")
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v))):
            pa3.append(float(v))
        else:
            pa3.append(0.0)
    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(max(6.0, 0.7 * len(names) + 3.0), 4.2))
    w = 0.35
    ax1.bar(x - w / 2, pr, width=w, color="#6baed6", label="Params (M)")
    ax1.set_ylabel("Params (M) ↓")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=12, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, pa3, "o-", color="darkorange", lw=1.5, ms=7, label="3DPW PA-MPJPE (mm)")
    ax2.set_ylabel("3DPW PA-MPJPE (mm) ↓")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    ax1.set_title("Params vs 3DPW PA (mm; lower is better) — " + RUN_ID, fontsize=9)
    fig.tight_layout()
    savefig_stem(OUTDIR / "ch6_lightweight_baseline_plot")


# ---------- Prompt 5: layerwise ----------
def export_layerwise(layer_json: Path) -> None:
    if not layer_json.is_file():
        print("WARN: layer json missing, skip", layer_json, file=sys.stderr)
        return
    with open(layer_json, encoding="utf-8") as f:
        raw = json.load(f)
    # keys "0".."L-1"
    L = max(int(k) for k in raw) + 1
    kti = []
    eff = []
    for i in range(L):
        b = raw[str(i)]
        km = b.get("kmi") or []
        rk = b.get("rank") or []
        kti.append(float(np.mean(km)) if km else 0.0)
        eff.append(float(np.mean(rk)) if rk else float("nan"))
    x = np.arange(L)
    fig, ax1 = plt.subplots(figsize=(7.5, 4.0))
    ax1.set_xlabel("Layer index")
    c_kti = "#d62728"
    c_r = "#1f77b4"
    if L > 1:
        ax1.axvspan(0, min(7.5, L - 0.5), color="#cfe8ff", alpha=0.35, zorder=0)
        if L > 8:
            ax1.axvspan(7.5, min(9.5, L - 0.5), color="#fff2cc", alpha=0.4, zorder=0)
        if L > 9:
            ax1.axvspan(9.5, L - 0.5, color="#e6e6e6", alpha=0.35, zorder=0)
    l1, = ax1.plot(x, kti, "o-", color=c_kti, label="KTI (higher = stronger topology signal)")
    ax1.set_ylabel("KTI (proxy, batch mean)", color=c_kti)
    ax1.tick_params(axis="y", labelcolor=c_kti)
    ax2 = ax1.twinx()
    l2, = ax2.plot(
        x,
        eff,
        "s-",
        color=c_r,
        label="Effective rank (higher = more subspace use)",
    )
    ax2.set_ylabel("Effective rank (higher = more use)", color=c_r)
    ax2.tick_params(axis="y", labelcolor=c_r)
    # 8–10 区标注
    if L > 8:
        ax1.annotate(
            "largest change @ ViT→Mamba; see text",
            xy=(7.4, kti[7] if L > 7 else 0),
            xytext=(7.0, (max(kti) if kti else 0) * 0.6),
            fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="0.35", lw=0.7),
        )
    ax1.set_title("Layerwise KTI & effective rank (see " + RUN_ID + " notes in .md)")
    hdl = [l1, l2]
    leg = [ln.get_label() for ln in hdl]
    ax1.legend(hdl, leg, loc="center right", fontsize=7, frameon=True)
    fig.tight_layout()
    savefig_stem(OUTDIR / "ch6_layerwise_kti_effrank")
    # 第二图：多指标
    ent = []
    for i in range(L):
        b = raw[str(i)]
        e = b.get("entropy") or []
        ent.append(float(np.mean(e)) if e else float("nan"))
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.5), sharex=True)
    series = [
        (ent, "Mean attention entropy", "higher = more spread"),
        (kti, "KTI (proxy, batch mean)", "higher is better; Mamba blocks may be 0 in this hook"),
        (eff, "Effective rank (batch mean)", "higher = less collapse"),
    ]
    for ax, (yy, name, up) in zip([axes[0, 0], axes[0, 1], axes[1, 0]], series):
        ax.plot(x, yy, "o-", lw=1.1, ms=3.5, color="#1f77b4")
        for x0, x1, col in [(0, 7, "#cfe8ff"), (7, 9, "#fff2cc"), (9, L, "#e6e6e6")]:
            if L > x0:
                ax.axvspan(
                    x0, min(x1, L - 0.1), color=col, alpha=0.2, zorder=0
                )
        ax.set_ylabel(f"{name}\n{up}", fontsize=7.5)
    axes[1, 0].set_xlabel("Layer index")
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.55,
        "MAD / geodesic dist: no `dist` in this run (empty).\nIncrease --diag_batches or use another split to plot.",
        fontsize=8.5,
        va="center",
    )
    fig.suptitle("Layerwise multi-metric (same run as kti+eff) — " + RUN_ID, fontsize=10, y=1.02)
    fig.tight_layout()
    savefig_stem(OUTDIR / "ch6_layerwise_multi_metrics")


# ---- Fig. 6-4: HMR2 vs Mamba 混合型 — 5 个诊断指标单图多子图 + LaTeX 表（见论文 Tab.~tab:ch6_diagnostic_comparison） ----
# 默认数值与正文表一致；可用 CH6_DIAG_JSON 指向
# {"hmr2": {...}, "mamba": {...}} 覆写；键名见 _diag_read_pair()
CH6_DIAG_BAR_PNG = "各项指标对比图"  # LaTeX: 0228/图表/chapter06/各项指标对比图.png

# 与学位论文表「对照组 / 本研究 NViT 最优」一致；图像 x 轴简写（仅中文，见 export_diagnostic）
DIAG_HMR2_LABEL = "基线 (HMR2)"
DIAG_NVIT_LABEL = "NViT 本研究最优 (step=492k)"
DIAG_EXTRA_DEFAULT_LABEL = "hmr2_mid_heavy (pruned pth)"
DIAG_EXTRA_DEFAULT_LAYER = (
    REPO / "outputs/eval_global/Ch6A/hmr2_mid_heavy_internal/layer_metrics_Control.json"
)
DIAG_EXTRA_DEFAULT_PARAMS_M = float(os.environ.get("CH6_DIAG_EXTRA_PARAMS_M", "416.331"))


def _parse_vit_layer_inclusive(s: str) -> tuple[int, int]:
    """如 \"0-6\" 含端点；与 ViT 段 0~6 层聚合一致。"""
    t = (s or "0-6").strip()
    if "-" in t:
        a, b = t.split("-", 1)
        return int(a.strip()), int(b.strip())
    n = int(t)
    return n, n


def _aggregate_nvit_vit_block(
    path: Path, vstart: int, vend: int
) -> dict[str, float | None]:
    """
    从 layer_metrics JSON 对 ViT 子块层做先层内、再层间 mean。
    返回 eff_rank(来自 rank 字段)、KTI(来自 kmi)、熵、MAD(来自非空 dist；缺省为 None)。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    layer_ranks, layer_kmi, layer_ent, layer_mad = [], [], [], []
    for i in range(vstart, vend + 1):
        b = raw.get(str(i))
        if not isinstance(b, dict):
            continue
        rnk = b.get("rank")
        if rnk and len(rnk) > 0:
            layer_ranks.append(float(np.mean(np.asarray(rnk, dtype=np.float64))))
        km = b.get("kmi")
        if km and len(km) > 0:
            layer_kmi.append(float(np.mean(np.asarray(km, dtype=np.float64))))
        en = b.get("entropy")
        if en and len(en) > 0:
            layer_ent.append(float(np.mean(np.asarray(en, dtype=np.float64))))
        dist = b.get("dist")
        if dist and len(dist) > 0:
            layer_mad.append(float(np.mean(np.asarray(dist, dtype=np.float64))))
    mad_v: float | None
    if layer_mad:
        mad_v = float(np.mean(np.asarray(layer_mad, dtype=np.float64)))
    else:
        mad_v = None
    return {
        "eff_rank": float(np.mean(np.asarray(layer_ranks)))
        if layer_ranks
        else 0.0,
        "kti": float(np.mean(np.asarray(layer_kmi))) if layer_kmi else 0.0,
        "entropy": float(np.mean(np.asarray(layer_ent))) if layer_ent else 0.0,
        "mad": mad_v,
    }

# mamba 键: NViT 一列的默认值与 ch6_main_step492000 聚合一致（MAD 无 dist 时为占位，见下）
DEFAULT_DIAG: dict[str, dict[str, float]] = {
    "hmr2": {
        "params_m": 672.27,
        "eff_rank": 106.3454,
        "mad": 45.9441,
        "kti": 0.0458,
        "entropy": 3.6885,
    },
    "mamba": {
        "params_m": 208.128,
        "eff_rank": 67.2870,
        "mad": 34.5694,
        "kti": 0.1392,
        "entropy": 2.5287,
    },
}


def _diag_read_pair() -> dict[str, dict[str, float]]:
    """环境变量 CH6_DIAG_JSON=path 可覆写 hmr2 / mamba（或 HMR2 / nvit）各标量键。"""
    p = (os.environ.get("CH6_DIAG_JSON") or "").strip()
    out: dict[str, dict[str, float]] = {
        "hmr2": copy.deepcopy(DEFAULT_DIAG["hmr2"]),
        "mamba": copy.deepcopy(DEFAULT_DIAG["mamba"]),
    }
    if not p or not Path(p).is_file():
        return out
    raw = json.loads(Path(p).read_text(encoding="utf-8"))
    mapping = (
        ("hmr2", "hmr2"),
        ("HMR2", "hmr2"),
        ("mamba", "mamba"),
        ("Mamba", "mamba"),
        ("nvit", "mamba"),
        ("NViT", "mamba"),
    )
    allowed = set(DEFAULT_DIAG["hmr2"].keys())
    for from_k, to_k in mapping:
        dct = raw.get(from_k)
        if not isinstance(dct, dict):
            continue
        for kk, v in dct.items():
            sk = str(kk)
            if sk in allowed and isinstance(v, (int, float)):
                out[to_k][sk] = float(v)
    return out


def _setup_cjk_font() -> None:
    """
    论文图面强制中文标签：优先用 CH6_FONT 指向的 .ttf/.ttc/.otf，
    再尝试系统常见 CJK 路径与已注册字体名。
    """
    from matplotlib import font_manager

    def _add_from_path(fp: str) -> bool:
        p = Path(fp)
        if not p.is_file():
            return False
        try:
            font_manager.fontManager.addfont(str(p))
            pr = font_manager.FontProperties(fname=str(p))
            plt.rcParams["font.sans-serif"] = [pr.get_name(), "DejaVu Sans"]
        except OSError:
            return False
        return True

    for env_key in ("CH6_FONT", "CJK_FONT"):
        t = (os.environ.get(env_key) or "").strip()
        if t and _add_from_path(t):
            plt.rcParams["axes.unicode_minus"] = False
            return

    static_tries: list[Path] = [
        REPO / "artifacts" / "fonts" / "NotoSansSC-Regular.otf",
        REPO / "artifacts" / "fonts" / "NotoSansCJKsc-Regular.otf",
        REPO / "artifacts" / ".fonts" / "NotoSansSC-Regular.otf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path(
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
        ),
        Path("/usr/share/fonts/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/wqy/wqy-microhei.ttc"),
    ]
    for sp in static_tries:
        if _add_from_path(str(sp)):
            plt.rcParams["axes.unicode_minus"] = False
            return

    try:
        avail = {f.name for f in font_manager.fontManager.ttflist}
        for name in (
            "WenQuanYi Zen Hei",
            "WenQuanYi Micro Hei",
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "Noto Sans CJK TC",
            "Source Han Sans SC",
            "Source Han Sans CN",
            "SimHei",
            "Microsoft YaHei",
        ):
            if name in avail:
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                break
        else:
            for f in font_manager.fontManager.ttflist:
                fp = (getattr(f, "fname", "") or "").lower()
                if "noto" in fp and "cjk" in fp and fp.endswith((".otf", ".ttf", ".ttc")):
                    font_manager.fontManager.addfont(f.fname)
                    plt.rcParams["font.sans-serif"] = [f.name, "DejaVu Sans"]
                    break
    except OSError:
        pass
    plt.rcParams["axes.unicode_minus"] = False


def _cjk_sans_ready() -> bool:
    s = plt.rcParams.get("font.sans-serif") or []
    first = s[0] if s else ""
    return any(
        x in first
        for x in (
            "CJK",
            "WenQuan",
            "Noto Sans CJK",
            "Noto Sans SC",
            "Noto Sans TC",
            "Noto Sans JP",
            "Source Han",
            "SimHei",
            "STHei",
            "Heiti",
        )
    )


def export_diagnostic_hmr2_vs_nvit_bars() -> None:
    """
    输出: 各项诊断指标条形图（2×3，右下空）+ 同表 LaTeX.
    论文 Fig.~fig:ch6_metrics_compare: 各项指标对比图.png
    论文 Tab.~tab:ch6_diagnostic_comparison: ch6_diagnostic_comparison_table.tex

    NViT 柱: 主 run 最优 step=492000 的 `layer_metrics_Control.json` 聚合（ViT 段层，默认 0–6）;
    参数量: `NVIT_CH6_PARAMS_M`（默认 208.128）。HMR2: `CH6_DIAG_JSON` 的 hmr2 与默认表。
    MAD: 当 JSON 中各层 `dist` 均为空时保留 `mamba.mad`（默认或 `CH6_DIAG_MAD_NVIT`）。
    图面文字**仅使用中文**（`CH6_FONT` 可指定 CJK 字体文件）。
    """
    import warnings

    _setup_cjk_font()
    if not _cjk_sans_ready():
        warnings.warn(
            "未检测到中文字体，图中汉语可能显示为方框。请设置环境变量 CH6_FONT 指向 .otf/.ttc 文件，"
            "或安装系统 fonts-noto-cjk，或将 NotoSansSC-Regular.otf 放入: "
            + str(REPO / "artifacts" / "fonts" / "")
            + "。",
            UserWarning,
            stacklevel=1,
        )
    data = _diag_read_pair()
    h, m = data["hmr2"], data["mamba"]
    layer_p = (os.environ.get("CH6_DIAG_LAYER_JSON") or str(CH6_BEST_STEP492K_LAYER)).strip()
    v0, v1 = _parse_vit_layer_inclusive(
        (os.environ.get("CH6_DIAG_VIT_LAYERS") or "0-6").strip()
    )
    lp = Path(layer_p)
    if lp.is_file():
        agg = _aggregate_nvit_vit_block(lp, v0, v1)
        m["eff_rank"] = float(agg["eff_rank"])
        m["kti"] = float(agg["kti"])
        m["entropy"] = float(agg["entropy"])
        if agg["mad"] is not None:
            m["mad"] = float(agg["mad"])
    tmd = (os.environ.get("CH6_DIAG_MAD_HMR2") or "").strip()
    if tmd:
        h["mad"] = float(tmd)
    tmn = (os.environ.get("CH6_DIAG_MAD_NVIT") or "").strip()
    if tmn:
        m["mad"] = float(tmn)
    m["params_m"] = float(DEFAULT_NVIT_PARAMS)
    extra_layer_p = (
        os.environ.get("CH6_DIAG_EXTRA_LAYER_JSON") or str(DIAG_EXTRA_DEFAULT_LAYER)
    ).strip()
    extra_label = (os.environ.get("CH6_DIAG_EXTRA_LABEL") or DIAG_EXTRA_DEFAULT_LABEL).strip()
    extra_enabled = bool(extra_layer_p and Path(extra_layer_p).is_file())
    extra = None
    if extra_enabled:
        extra = {
            "params_m": float(
                (os.environ.get("CH6_DIAG_EXTRA_PARAMS_M") or DIAG_EXTRA_DEFAULT_PARAMS_M)
            )
        }
        ex_agg = _aggregate_nvit_vit_block(Path(extra_layer_p), v0, v1)
        extra["eff_rank"] = float(ex_agg["eff_rank"])
        extra["kti"] = float(ex_agg["kti"])
        extra["entropy"] = float(ex_agg["entropy"])
        extra["mad"] = (
            float(ex_agg["mad"]) if ex_agg["mad"] is not None else float(m["mad"])
        )
        tme = (os.environ.get("CH6_DIAG_MAD_EXTRA") or "").strip()
        if tme:
            extra["mad"] = float(tme)

    if extra_enabled and extra is not None:
        # 用户期望固定顺序：baseline -> heavy -> nvit
        xlabels = [DIAG_HMR2_LABEL, extra_label, DIAG_NVIT_LABEL]
        model_stats = [h, extra, m]
    else:
        xlabels = [DIAG_HMR2_LABEL, DIAG_NVIT_LABEL]
        model_stats = [h, m]
    w = 0.55
    bmsg = "越小越好"
    bbig = "越大越好"

    def _one_bar(
        ax, vals: list[float], ylabel: str, title: str, direction: str, colors
    ) -> None:
        xs = np.arange(len(vals))
        bars = ax.bar(xs, vals, width=w, color=colors, edgecolor="0.3", linewidth=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, rotation=10, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(f"{title}（{direction}）", fontsize=9.5, pad=6)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
        for b in bars:
            hgt = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                hgt * 1.01,
                f"{hgt:.2f}" if hgt >= 0.1 else f"{hgt:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        y_top = max(vals) * 1.12
        if y_top <= 0:
            y_top = 0.1
        ax.set_ylim(0, y_top)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2))
    # 所有子图使用同一组模型颜色映射，保证视觉一致性
    model_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"][: len(model_stats)]
    _one_bar(
        axes[0, 0],
        [float(s["params_m"]) for s in model_stats],
        "参数 (M)",
        "模型大小",
        bmsg,
        model_colors,
    )
    _one_bar(
        axes[0, 1],
        [float(s["eff_rank"]) for s in model_stats],
        "有效秩",
        "冗余度",
        bmsg,
        model_colors,
    )
    _one_bar(
        axes[0, 2],
        [float(s["mad"]) for s in model_stats],
        "平均绝对偏差 (MAD)",
        "注意力距离",
        bmsg,
        model_colors,
    )
    _one_bar(
        axes[1, 0],
        [float(s["kti"]) for s in model_stats],
        "KTI",
        "拓扑对齐",
        bbig,
        model_colors,
    )
    _one_bar(
        axes[1, 1],
        [float(s["entropy"]) for s in model_stats],
        "熵",
        "注意力离散度",
        bmsg,
        model_colors,
    )
    axes[1, 2].axis("off")

    # 论文标准版：不加总标题，仅保留子图标题与轴标签
    fig.tight_layout()
    stem = OUTDIR / CH6_DIAG_BAR_PNG
    savefig_stem(stem)
    hmr2_row = [
        "对照组 (HMR2)",
        f"{h['params_m']:.2f}",
        f"{h['eff_rank']:.4f}",
        f"{h['mad']:.4f}",
        f"{h['kti']:.4f}",
        f"{h['entropy']:.4f}",
    ]
    m_row = [
        "NViT 本研究最优 (step=492k)",
        f"{m['params_m']:.2f}",
        f"{m['eff_rank']:.4f}",
        f"{m['mad']:.4f}",
        f"{m['kti']:.4f}",
        f"{m['entropy']:.4f}",
    ]
    extra_row = None
    if extra_enabled and extra is not None:
        extra_row = [
            extra_label,
            f"{extra['params_m']:.2f}",
            f"{extra['eff_rank']:.4f}",
            f"{extra['mad']:.4f}",
            f"{extra['kti']:.4f}",
            f"{extra['entropy']:.4f}",
        ]
    tex_rows = [" & ".join(hmr2_row) + r" \\"]
    if extra_row is not None:
        tex_rows.append(" & ".join(extra_row) + r" \\")
    tex_rows.append(" & ".join(m_row) + r" \\")
    caption = "第6章：诊断指标与模型参数量对比 (HMR2 基线 vs. NViT 最优, step=492k)"
    if extra_row is not None:
        caption = "第6章：诊断指标与模型参数量对比 (HMR2 基线 / NViT 最优 / hmr2_mid_heavy)"
    tex = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption}}}",
            r"\label{tab:ch6_diagnostic_comparison}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\textbf{模型} & \textbf{参数量 (M)} & \textbf{有效秩 $\downarrow$} & \textbf{MAD $\downarrow$} & \textbf{KTI $\uparrow$} & \textbf{空间熵 $\downarrow$} \\",  # noqa: W605
            r"\midrule",
            *tex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (OUTDIR / "ch6_diagnostic_comparison_table.tex").write_text(tex, encoding="utf-8")
    write_md(
        OUTDIR / f"{CH6_DIAG_BAR_PNG}.md",
        f"""# {CH6_DIAG_BAR_PNG}（HMR2 与 NViT 诊断条形图，中文图注）

- **图**: `{CH6_DIAG_BAR_PNG}.png` + `{CH6_DIAG_BAR_PNG}.pdf`（同 stem）。  
- **HMR2** 行: 默认与学位论文表一致；`CH6_DIAG_JSON` 可覆写 `hmr2` 各标量。  
- **NViT** 行: 自 `CH6_DIAG_LAYER_JSON`（缺省为 `ch6_main_step492000/layer_metrics_Control.json`）对 `CH6_DIAG_VIT_LAYERS`（默认 `0-6`）做层内/层间聚合；`NVIT_CH6_PARAMS_M` 为参数量。  
- **可选第 3 行（默认开启）**: 若 `CH6_DIAG_EXTRA_LAYER_JSON`（默认 `outputs/eval_global/Ch6A/hmr2_mid_heavy_internal/layer_metrics_Control.json`）存在，则自动追加 `CH6_DIAG_EXTRA_LABEL`（默认 `hmr2_mid_heavy (pruned pth)`）。参数量由 `CH6_DIAG_EXTRA_PARAMS_M` 控制（默认 `416.331`）。  
- **MAD** 当 layer JSON 中 `dist` 为空时沿用 `mamba.mad` 或设 `CH6_DIAG_MAD_NVIT`；HMR2 的 MAD 可设 `CH6_DIAG_MAD_HMR2`。  
- **字体**: 设置 `CH6_FONT=/path/to/NotoSansCJK-*.otf`（或已装系统 Noto/WQY）以正确显示中文。  
- 生成: `python3 {REPO / "artifacts" / "generate_chapter06_artifacts.py"}`  
""",
    )


# ---------- Prompt 6 ----------
def export_resource_placeholder() -> None:
    """无 GPU 或未跑 --bench 时的占位表；列与 ch6_bench_resources 输出一致 (vram_mb)。"""
    data = [
        {
            "architecture": "HMR2 ViT-32 (full backbone)",
            "params_m": 224.0,
            "flops_g": "—",
            "latency_ms": "—",
            "vram_mb": "—",
            "note": "typical HMR2 scale; run: python3 artifacts/ch6_bench_resources.py",
        },
        {
            "architecture": "Trunc. ViT 0–11 (12 blocks)",
            "params_m": 86.0,
            "flops_g": "—",
            "latency_ms": "—",
            "vram_mb": "—",
            "note": "illustrative; set --ckpt-trunc + bench",
        },
        {
            "architecture": "KTI-guided hybrid (ch6, depth=11)",
            "params_m": 208.128,
            "flops_g": "—",
            "latency_ms": "—",
            "vram_mb": "—",
            "note": f"from {RUN_ID} config; full numbers via ch6_bench_resources",
        },
    ]
    p = OUTDIR / "ch6_resource_table.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "architecture",
                "params_m",
                "flops_g",
                "latency_ms",
                "vram_mb",
                "note",
            ],
        )
        w.writeheader()
        w.writerows(data)


def render_resource_tex_and_plot(csv_path: Path) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        data = list(rdr)
    rows_tex = []
    pnums: list[float] = []
    for d in data:
        rows_tex.append(
            [
                d.get("architecture", "—") or "—",
                d.get("params_m", "—") or "—",
                d.get("flops_g", "—") or "—",
                d.get("latency_ms", "—") or "—",
                d.get("vram_mb", d.get("vram_gb", "—")) or "—",
            ]
        )
        try:
            pnums.append(float(str(d.get("params_m", "nan"))))
        except (TypeError, ValueError):
            pnums.append(0.0)
    (OUTDIR / "ch6_resource_table.tex").write_text(
        booktabs_table(
            [
                "Architecture",
                "Params (M)",
                "FLOPs (G)",
                "Latency (ms) b1",
                "VRAM (MB)",
            ],
            rows_tex,
            "Resource (lower is better for Params, GFLOPs, Latency, peak VRAM). "
            "Forward only, no data loading. "
            r"GFLOPs computed as $2\times$ MACs using THOP on 256$\times$192, batch$=$1, FP32; if \texttt{thop} is unavailable, FLOPs is~---. "
            r"Two NViT rows in the lightweight table (if listed) use the \emph{same} architecture; only checkpoints differ---resource values for NViT should be identical (repeat or footnote, no double count).",
            "tab:ch6_resource",
        ),
        encoding="utf-8",
    )
    n = len(pnums)
    if n == 0:
        return
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.0 * n + 2.0), 3.6))
    ax.bar(x, pnums, color=["#6baed6", "#9ecae1", "#fd8d3c"] * ((n + 2) // 3), edgecolor="0.3", lw=0.4)
    short = [d.get("architecture", str(i))[:20] for i, d in enumerate(data)]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=18, ha="right", fontsize=7.5)
    for i, v in enumerate(pnums):
        if v and v == v:  # not NaN
            ax.text(i, v + 2, f"{v:.0f}" if v > 50 else f"{v:.1f}", ha="center", fontsize=7)
    ax.set_ylabel("Params (M) (lower is better)")
    ax.set_title("Resource (params) — FLOPs/Lat/VRAM: ch6_resource_table.csv")
    fig.tight_layout()
    savefig_stem(OUTDIR / "ch6_resource_plot")


def run_resource_bench() -> None:
    from ch6_bench_resources import run_bench  # type: ignore

    out = OUTDIR / "ch6_resource_table.csv"
    dev = os.environ.get("CH6_BENCH_DEVICE", "cuda:0")
    run_bench(
        out_csv=out,
        hmr2_ckpt=None,
        trunc_ckpt=None,
        nvit_ckpt=None,
        device=dev,
    )
    if out.is_file():
        render_resource_tex_and_plot(out)


def export_resource(bench: bool) -> None:
    p = OUTDIR / "ch6_resource_table.csv"
    do = bench or (os.environ.get("DO_BENCH", "").strip() in ("1", "true", "yes"))
    if do:
        try:
            run_resource_bench()
        except Exception as e:  # noqa: BLE001
            print("WARN: bench 失败, 用占位 CSV: ", e, file=sys.stderr)
            if not p.is_file():
                export_resource_placeholder()
    else:
        if not p.is_file():
            export_resource_placeholder()
    if p.is_file():
        render_resource_tex_and_plot(p)


def main(bench: bool = False) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    records = load_run_records()
    if not records:
        raise SystemExit("No records for run " + RUN_ID)
    # Prompt 1
    rows = build_step_table(records)
    with open(OUTDIR / "ch6_step_dynamics_filtered.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["step", "3dpw_mpjpe", "3dpw_pa", "h36m_mpjpe", "h36m_pa"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k]) for k in w.fieldnames if k in r}
            )
    plot_step_dynamics(rows)
    plot_pareto(rows)
    # Prompt 2
    sel = build_checkpoint_selection(records)
    with open(OUTDIR / "ch6_checkpoint_selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "3dpw_mpjpe",
                "3dpw_pa",
                "h36m_mpjpe",
                "h36m_pa",
                "rank_3dpw",
                "rank_h36m",
                "rank_sum",
                "is_3dpw_best",
                "is_h36m_best",
                "is_composite_best",
            ],
        )
        w.writeheader()
        w.writerows(sel)
    ranked = c6.list_composite_ranked(records, RANK_3D, top_k=10)
    # LaTeX top-5 + 强制 step 行
    def row_tex(st: int | None) -> list[str] | None:
        if st is None:
            return None
        s0 = next((s for s in sel if s["step"] == st), None)
        if not s0:
            return None
        return [
            str(s0["step"]),
            f'{float(s0["3dpw_mpjpe"]):.1f}' if s0["3dpw_mpjpe"] else "—",
            f'{float(s0["3dpw_pa"]):.1f}' if s0["3dpw_pa"] else "—",
            f'{float(s0["h36m_mpjpe"]):.1f}' if s0["h36m_mpjpe"] else "—",
            f'{float(s0["h36m_pa"]):.1f}' if s0["h36m_pa"] else "—",
            str(s0["rank_3dpw"]),
            str(s0["rank_h36m"]),
            str(s0["rank_sum"]),
        ]
    spec_steps = {274_000, 360_000, 438_000, 492_000}
    tex_rows: list[list[str]] = []
    seen: set[str] = set()
    for row in ranked[:5]:
        stp = row.get("step")
        t = row_tex(int(stp)) if stp is not None else None
        if t and t[0] not in seen:
            tex_rows.append(t)
            seen.add(t[0])
    for s in (274_000, 492_000):
        t = row_tex(s)
        if t and t[0] not in seen:
            tex_rows.append(t)
            seen.add(t[0])
    (OUTDIR / "ch6_checkpoint_selection.tex").write_text(
        booktabs_table(
            [
                "Step",
                "3DPW M",
                "3DPW PA",
                "H36M M",
                "H36M PA",
                r"R$_{3D}$",
                r"R$_{H}$",
                r"$\sum$R",
            ],
            tex_rows,
            "Checkpoint selection: top-5 by rank-sum (PA) + steps 274k, 492k. Lower error / rank is better. "
            + RUN_ID,
            "tab:ch6_ckpt",
        )
        + "\n% 特殊 step 在 top-10: 见 ch6_checkpoint_selection_annotations.txt\n",
        encoding="utf-8",
    )
    # 注解文件
    by_sum = c6.list_composite_ranked(records, RANK_3D, top_k=None)
    pos_by_step: dict[int, int] = {}
    for pos, r in enumerate(by_sum, start=1):
        stp = r.get("step")
        if isinstance(stp, int):
            pos_by_step[stp] = pos
    top10_set = {r.get("step") for r in by_sum[:10] if isinstance(r.get("step"), int)}
    with open(OUTDIR / "ch6_checkpoint_selection_annotations.txt", "w", encoding="utf-8") as f:
        f.write(f"rank_sum order (1=best) among {len(by_sum)} checkpoints (both datasets present).\n")
        for s in sorted(spec_steps):
            p = pos_by_step.get(s)
            if p is None:
                f.write(f"step {s}: not in list (可能未评测到该步)\n")
            elif s in top10_set and p is not None:
                f.write(f"step {s}: rank in rank_sum order = {p}  (在 top-10 内, top10 指 rank 1..10)\n")
            else:
                f.write(
                    f"step {s}: 全序排名 = {p}  (不在 top-10; top-10 为 rank 1-10 的 step 集合)\n"
                )
    # Prompt 3
    want = [10_000, 100_000, 274_000, 360_000, 428_000, 438_000, 492_000, 504_000]
    near = nearest_rows(rows, want)
    with open(OUTDIR / "ch6_key_steps_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "target_step",
                "nearest_step",
                "step_delta",
                "3dpw_mpjpe",
                "3dpw_pa",
                "h36m_mpjpe",
                "h36m_pa",
            ],
        )
        w.writeheader()
        for n in near:
            w.writerow(
                {
                    "target_step": n["target_step"],
                    "nearest_step": n["nearest_step"],
                    "step_delta": n["step_delta"],
                    "3dpw_mpjpe": f'{n["3dpw_mpjpe"]:.4f}' if n.get("3dpw_mpjpe") is not None else "",
                    "3dpw_pa": f'{n["3dpw_pa"]:.4f}' if n.get("3dpw_pa") is not None else "",
                    "h36m_mpjpe": f'{n["h36m_mpjpe"]:.4f}' if n.get("h36m_mpjpe") is not None else "",
                    "h36m_pa": f'{n["h36m_pa"]:.4f}' if n.get("h36m_pa") is not None else "",
                }
            )
    kr = []
    for n in near:
        kr.append(
            [
                f"{n['nearest_step']}",
                f'{n["3dpw_mpjpe"]:.1f}',
                f'{n["3dpw_pa"]:.1f}',
                f'{n["h36m_mpjpe"]:.1f}',
                f'{n["h36m_pa"]:.1f}',
                f"($\\Delta$={n['step_delta']:+d})" if n["step_delta"] != 0 else "—",
            ]
        )
    (OUTDIR / "ch6_key_steps_table.tex").write_text(
        booktabs_table(
            [r"Step$^\ast$", "3DPW M", "3DPW PA", "H36M M", "H36M PA", "nearest"],
            kr,
            r"Key training steps: \texttt{nearest} aligns to last-eval checkpoint if exact step missing. "
            r"Metrics in mm; lower is better. Saving $\approx$14k steps. Run " + RUN_ID + ".",
            "tab:ch6_keystep",
        ),
        encoding="utf-8",
    )
    # Prompt 4,5,6
    export_lightweight()
    layer_path = Path(os.environ.get("LAYER_METRICS_JSON", str(DEFAULT_LAYER_JSON)))
    export_layerwise(layer_path)
    export_diagnostic_hmr2_vs_nvit_bars()
    export_resource(bench)
    # 写 md
    cmd = f"python3 {REPO / 'artifacts' / 'generate_chapter06_artifacts.py'}"
    data_note = f"`{MASTER}` 中 `chapter=ch6` 且 `json_path` 含 `{RUN_ID}` ；`MPJPE_mm`/`PA_MPJPE_mm` 为 mm。lower is better。"
    write_md(
        OUTDIR / "README_chapter06.md",
        f"""# Chapter 6 产物总览（论文附件级图表产线）

## 一键生成
```bash
export PYTHONPATH={REPO}
python3 {REPO / "artifacts" / "generate_chapter06_artifacts.py"}
# 可选：同表填资源实测（需 GPU，且建议安装 thop 以算 FLOPs）
python3 {REPO / "artifacts" / "generate_chapter06_artifacts.py"} --bench
# 或: DO_BENCH=1 …
```

## 资源 bench 统一协议（写死再测，避免不可比）
- **输入**: RGB **256×192**，**batch=1**（与 `ch6_bench_resources.py` 中 `H,W` 一致）
- **精度**: **FP32**（若论文需 AMP 另起表并写清）
- **设备**: 在结果或 `ch6_resource_bench_meta.json` 中注明 **GPU 型号**；推荐 `CH6_BENCH_DEVICE=cuda:0`
- **计时**: 仅 **forward**（无 dataloader / 后处理）；`torch.cuda.synchronize()` + `time.perf_counter`；**warmup=50, repeat=200**（与脚本一致，可改代码但须留记录）
- **FLOPs**: 使用 **thop**（与 `nvit/efficiency_profiler` 相同：`GFLOPs ≈ 2×MACs/1e9`）；未安装 thop 时列为 `—`
- **显存**: `torch.cuda.max_memory_allocated()`，forward 前 `reset_peak_memory_stats`；单位 **MB**（表头）
- **三行模型**: 通过环境变量提供 **本地 ckpt 路径**（缺则该行 `—`）  
  - `CH6_BENCH_HMR2_CKPT`：HMR2 / 32L  
  - `CH6_BENCH_TRUNC_CKPT`：浅层/截断 ViT  
  - `CH6_BENCH_NVIT_CKPT`：默认不填则使用 ch6 360k hybrid 路径

也可单独跑：  
`python3 {REPO / "artifacts" / "ch6_bench_resources.py"} --out {OUTDIR / "ch6_resource_table.csv"}`

## 定稿口径（与论文「不打架」版）
- **主结果**：以 **composite (rank-sum)** 选点为主（`EXAMPLE_JSON` 对应 `NViT (Ch6, composite-best)`）。  
- **3DPW**：在 **训练动态 / Pareto** 中讨论 sweet spot（如 ~274k）与**跨数据集 trade-off**；不替代主选点。  
- **轻量化表**（推荐 B）：`CH6_NVIT_3DPW_JSON` 指向 3DPW 取向的 eval json 时，与 composite 行并存，**固定行名**  
  - `NViT (Ch6, 3DPW-oriented)`  
  - `NViT (Ch6, composite-best)`  
- **同结构、仅 ckpt 不同**：两行人 **Params/FLOPs/Lat/VRAM 相同**；`ch6_resource_table` 仅对 NViT **架构**测一次；caption 中已说明。

## 论文章节与文件对应（便于 LaTeX `\\includegraphics` / `\\input`）
- **Fig.~\\ref{{fig:ch6_metrics_compare}} 各项指标对比**（HMR2 vs Mamba 混合型，5 子图条形图）: `{CH6_DIAG_BAR_PNG}.png` + `ch6_diagnostic_comparison_table.tex`（`\\input` tab:ch6_diagnostic_comparison）  
- 层诊断: `ch6_layerwise_kti_effrank` + `ch6_layerwise_multi_metrics`
- 训练与 trade-off: `ch6_step_dynamics` + `ch6_pareto_3dpw_vs_h36m` + `ch6_key_steps_table` + `ch6_checkpoint_selection`
- 与 METRO 等对比: `ch6_lightweight_baseline_*`
- 资源: `ch6_resource_table` + `ch6_resource_plot`

## `scientific_diagnostics.py` 缩进/Import 问题
若 `import` / 运行 `global_evaluator` 在 `scientific_diagnostics.py` 报 **IndentationError**（多因错误插入的 `import` 块），请 **同步本仓库** 中已修复的 `nvit/skills/evaluate_model/scientific_diagnostics.py`（SMPL 路径块应连续缩进，无行内误插 `from nvit.utils...`）。

## 主数据
- `metrics_master.csv`: `{MASTER}`，run id `{RUN_ID}`  
- KTI 层曲线默认: `{DEFAULT_LAYER_JSON}`，或 `LAYER_METRICS_JSON=...`
""",
    )
    write_md(
        OUTDIR / "ch6_step_dynamics.md",
        f"""# ch6_step_dynamics / Pareto

- **数据**: {data_note}
- **脚本**: {cmd}
- **输出文件**: `ch6_step_dynamics` / `ch6_pareto_3dpw_vs_h36m` 的 png+pdf，及 `ch6_step_dynamics_filtered.csv`。
""",
    )
    write_md(
        OUTDIR / "ch6_pareto_3dpw_vs_h36m.md",
        f"""# Pareto: 3DPW PA vs H36M PA

- 同一步 checkpoint 的横纵轴均为 **PA-MPJPE (mm)**，**lower is better**。
- 高亮: 与 274k/360k/492k 最近的已评测步（容差 16k steps）。
- {data_note}  
- 命令: `{cmd}`
""",
    )
    write_md(
        OUTDIR / "ch6_checkpoint_selection.md",
        f"""# Rank-sum checkpoint 表

- **规则**: 与 `scripts/unified_eval_batch.py` 的 `compute_composite_best` 一致，数据集 `3DPW-TEST`+`H36M-VAL-P2`，`rank_metric_3d=mode_re` (PA-MPJPE)。  
- **rank_3dpw / rank_h36m**: 各数据集上按 PA 升序名次（1=最好）。**rank_sum**=两列之和。  
- 命令: `{cmd}`  
- 见 `ch6_checkpoint_selection_annotations.txt` 中 274k/360k/438k/492k 与 top-10 关系。
""",
    )
    write_md(
        OUTDIR / "ch6_key_steps_table.md",
        f"""# 关键步数表 (nearest)

- 目标步来自 `ch6_step_dynamics_filtered.csv` 的 **最近邻** step；保存周期约 14k，故常含非零 `step_delta`。
- 所有误差为 **mm**，lower is better。  
- `{cmd}`
""",
    )
    write_md(
        OUTDIR / "ch6_lightweight_baseline_plot.md",
        f"""# 轻量化方法对比

- **NViT (composite)**: 由 `{EXAMPLE_JSON.name}` 自动解析；**禁止手填**。
- 若设 `CH6_NVIT_3DPW_JSON` 指向 **3DPW 取向** step 的 eval json，会多一行 **NViT (3DPW-oriented)**，与 composite 行并存，避免「正文写 274k 好、表里却用 360k」的歧义。
- **METRO / Mesh Graphormer**: 文献表数字，请在正文写清论文与表号。
- 双轴图: Params 柱 + 3DPW PA 折线；**mm**，lower is better。  
- `{cmd}`
""",
    )
    write_md(
        OUTDIR / "ch6_layerwise_kti_effrank.md",
        f"""# Layerwise KTI / Effective rank

- **数据来源**: 实测 `global_evaluator` 诊断，checkpoint `step_step=360000`，run label `ch6_paper_layerwise`，`layer_metrics_Control.json` 路径:  
  `{DEFAULT_LAYER_JSON}`（可用 `LAYER_METRICS_JSON=...` 覆写）。  
- **KTI 曲线**: 本实现中 ViT 段有非零 KTI，**Mamba 段钩子返回的 kmi 可能为 0**：属实现/蒙版路径差异，正文可配合 eff-rank 与 entropy 解读。  
- **区段着色**: 0–7 / 8–9 / 10+ 对应该模型 depth=11 (layer index 0..10).  
- 复现命令（示例，需 GPU）:  
  `cd {REPO} && export PYTHONPATH=. && python -m nvit.global_evaluator --chapter Ch6A --checkpoint_path <ckpt> --run_label <label> --diag_batches 2 --limit_batches 1 --datasets 3DPW-TEST`  
- 多指标子图: `ch6_layerwise_multi_metrics`（同数据）。
""",
    )
    write_md(
        OUTDIR / "ch6_resource_plot.md",
        f"""# 资源开销 (Params / FLOPs / Latency / VRAM)

- **本表为初稿占位**: 224M / 86M / 208.128M 中后两者为示意/配置值；**FLOPs、Latency、显存**在 CSV 中记为 "—" 时，请用同机、同输入 `256×192`、**batch=1** 的 profiler/计时脚本填实。  
- 硬件: 建议注明 **A100-80GB** 或你们实测 GPU。  
- 指标方向: 参数量/算量/时延/显存 **在可比设置下均 lower 更优** (非 mm 误差类)。  
- 实测量: `python3 {REPO / "artifacts" / "ch6_bench_resources.py"}` 或 `{cmd} --bench` 。
""",
    )
    write_md(
        OUTDIR / "ch6_checkpoint_narrative_zh.md",
        r"""# 选点与跨域权衡（中文正文可用骨架，Fig/Tab 编号请自行替换）

**Checkpoint 选择与跨域权衡说明**：本章所有对外汇报的主结果默认采用 **composite-best** 选点规则，即在 \(\{ \text{3DPW-TEST}, \text{H36M-VAL-P2} \}\) 上以 **PA-MPJPE** 为排序指标分别排名并进行 rank-sum（Borda-style）汇总，选择总排名最优的 checkpoint（见 Tab.~X，算法与脚本 `unified_eval_batch.compute_composite_best` 一致）。与此同时，我们对同一条训练轨迹的 step 级离线评测曲线进行诊断性分析（Fig.~Y、Fig.~Z），观察到在约 \(\sim274\)k steps 附近存在对 **3DPW** 更有利的区域，而在 \(\gtrsim300\)k steps 后模型继续优化更倾向于提升 **H36M** 指标、对 3DPW 改善有限甚至略有回落。基于该观察，我们在轻量化方法横向对比表中同时列出 **NViT 的 composite-best** 与 **3DPW-oriented** 两个 checkpoint（Tab.~W），以避免单域最优与多域折中选点的语义混淆。上述现象均基于单次训练轨迹的离线评测结果，属于经验性诊断结论，不做跨种子/跨设置的普适性外推。

---

**英文学位论文/摘要可用一句**：

> We report the **composite-best** checkpoint selected by a rank-sum rule over {3DPW-TEST, H36M-VAL-P2}. We additionally analyze the observed 3DPW-favorable region around ~274k steps to illustrate cross-dataset trade-offs.
""",
    )
    print("OK ->", OUTDIR)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench",
        action="store_true",
        help="跑 GPU 资源 bench（需 CUDA + thop 可算 FLOPs；见 README_chapter06.md）",
    )
    a = ap.parse_args()
    main(bench=a.bench)
