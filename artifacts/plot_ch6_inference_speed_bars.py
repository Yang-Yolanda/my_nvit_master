#!/usr/bin/env python3
"""
合并多份推理速度 CSV，画一张「ms/图」柱状图（论文用）。

默认读:
  outputs/eval_global/Ch6A/hmr2_vs_nvit_bench.csv
  outputs/eval_global/Ch6A/metro_meshg_inference_speed.csv（若存在）

跳过无有效 ms_per_image 的行（含 SKIPPED）。
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

_REPO = Path(__file__).resolve().parents[1]


def _read_rows(p: Path) -> list[tuple[str, float]]:
    if not p.is_file():
        return []
    out: list[tuple[str, float]] = []
    with p.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            name = (row.get("name") or "").strip()
            ms = (row.get("ms_per_image") or "").strip()
            if not name or not ms:
                continue
            if "SKIP" in ms.upper() or ms.upper() == "SKIPPED":
                continue
            try:
                v = float(ms)
            except ValueError:
                continue
            out.append((name, v))
    return out


def _order_key(name: str) -> tuple[int, str]:
    n = (name or "").strip().lower()
    if "hmr2" in n and "mid_heavy" not in n:
        return (0, n)  # baseline
    if "mid_heavy" in n or "heavy" in n:
        return (1, n)
    if "nvit" in n:
        return (2, n)
    if "metro" in n:
        return (3, n)
    if "meshgraphormer" in n or "mesh" in n:
        return (4, n)
    return (9, n)


def _color_for_name(name: str) -> str:
    n = (name or "").strip().lower()
    # 与 chapter06 其它图统一：baseline(蓝) / heavy(橙) / nvit(绿)
    if "hmr2" in n and "mid_heavy" not in n:
        return "#1f77b4"
    if "mid_heavy" in n or "heavy" in n:
        return "#ff7f0e"
    if "nvit" in n:
        return "#2ca02c"
    if "metro" in n:
        return "#9467bd"
    if "meshgraphormer" in n or "mesh" in n:
        return "#8c564b"
    return "#7f7f7f"


def _setup_cjk_font() -> None:
    def _add_from_path(fp: str) -> bool:
        p = Path(fp)
        if not p.is_file():
            return False
        try:
            font_manager.fontManager.addfont(str(p))
            pr = font_manager.FontProperties(fname=str(p))
            plt.rcParams["font.sans-serif"] = [pr.get_name(), "DejaVu Sans"]
            return True
        except OSError:
            return False

    for env_key in ("CH6_FONT", "CJK_FONT", "NVIT_CJK_FONT"):
        t = (os.environ.get(env_key) or "").strip()
        if t and _add_from_path(t):
            plt.rcParams["axes.unicode_minus"] = False
            return

    static_tries = [
        _REPO / "artifacts" / "fonts" / "NotoSansSC-Regular.otf",
        _REPO / "artifacts" / "fonts" / "NotoSansCJKsc-Regular.otf",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/wqy/wqy-microhei.ttc"),
    ]
    for sp in static_tries:
        if _add_from_path(str(sp)):
            plt.rcParams["axes.unicode_minus"] = False
            return

    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hmr2-nvit-csv",
        type=Path,
        default=_REPO / "outputs" / "eval_global" / "Ch6A" / "hmr2_vs_nvit_bench.csv",
    )
    ap.add_argument(
        "--metro-meshg-csv",
        type=Path,
        default=_REPO / "outputs" / "eval_global" / "Ch6A" / "metro_meshg_inference_speed.csv",
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=_REPO / "outputs" / "eval_global" / "Ch6A" / "ch6_inference_speed_all.png",
    )
    args = ap.parse_args()

    pairs: list[tuple[str, float]] = []
    pairs.extend(_read_rows(args.hmr2_nvit_csv))
    pairs.extend(_read_rows(args.metro_meshg_csv))
    if not pairs:
        raise SystemExit("没有可用的速度行（检查两份 CSV 是否存在且含 ms_per_image）")
    pairs = sorted(pairs, key=lambda x: _order_key(x[0]))

    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = [_color_for_name(n) for n in names]

    _setup_cjk_font()
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(names) + 2), 4.2))
    ax.bar(range(len(names)), vals, color=colors, edgecolor="0.25", linewidth=0.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("推理时延（毫秒/图，越小越好）")
    ax.set_title("推理速度对比")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(args.out_png)


if __name__ == "__main__":
    main()
