#!/usr/bin/env python3
"""
Ch5 遮挡消融：两张分组柱状图（MPJPE、PA-MPJPE），无图题；中文坐标轴与图例文字。
默认读 occlusion_table.csv；六组同图时 M3/M5 量级偏大，纵轴默认对数刻度。
中文字体：优先环境变量 NVIT_FONT_OTF、本目录 artifacts/fonts/NotoSansSC-Regular.otf，
否则下载到 ~/.cache/nvit_zh_font/NotoSansSC-Regular.otf（Noto SC 子集，约 8MB+，仅首次）。
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm

plt.rcParams["axes.unicode_minus"] = False
# 与 Ch5 其它图统一：偏大字号、无总标题（本脚本不调用 suptitle/title）
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)

_FONT_URLS = (
    "https://raw.githubusercontent.com/googlefonts/noto-cjk/"
    "main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf",
    "https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf",
)


def _setup_zh_font(verbose: bool = False) -> None:
    """注册可显示简中汉字的字体，失败则 stderr 提示。"""
    candidates: list[Path] = []
    env_p = os.environ.get("NVIT_FONT_OTF", "").strip()
    if env_p:
        candidates.append(Path(env_p).expanduser())
    candidates.append(Path(__file__).resolve().parent / "fonts" / "NotoSansSC-Regular.otf")
    cache = Path.home() / ".cache" / "nvit_zh_font" / "NotoSansSC-Regular.otf"
    candidates.append(cache)

    # 官方子集 OTF 完整约 16MB；低于阈值则重下（自备字体路径不受此限）
    _min_cache_bytes = 2_500_000

    chosen: Path | None = None
    for p in candidates:
        if not p.is_file():
            continue
        sz = p.stat().st_size
        if sz < 500_000:
            continue
        if p == cache and sz < _min_cache_bytes:
            try:
                p.unlink()
            except OSError:
                pass
            continue
        chosen = p
        break

    if chosen is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"正在下载中文字体到 {cache} …", file=sys.stderr)
        last_err: Exception | None = None
        for u in _FONT_URLS:
            tmp = cache.with_suffix(".otf.part")
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "nvit-plot/1"})
                with urllib.request.urlopen(req, timeout=900) as r, tmp.open("wb") as w:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        w.write(chunk)
                sz = tmp.stat().st_size
                if sz < _min_cache_bytes:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"字体下载不完整({sz} bytes)")
                tmp.replace(cache)
                chosen = cache
                last_err = None
                break
            except Exception as e:
                last_err = e
                tmp.unlink(missing_ok=True)
                continue
        if chosen is None:
            print(
                f"警告：无法下载中文字体（{last_err}）。"
                f"请设置 NVIT_FONT_OTF=/path/to/NotoSansSC-Regular.otf 或把字体放到 artifacts/fonts/。",
                file=sys.stderr,
            )
            return

    try:
        fm.fontManager.addfont(str(chosen))
        name = fm.FontProperties(fname=str(chosen)).get_name()
        plt.rcParams["font.family"] = [name]
        plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    except Exception as e:
        if chosen == cache:
            try:
                chosen.unlink()
            except OSError:
                pass
        print(f"警告：注册字体失败: {e}", file=sys.stderr)

GROUP_ZH = {
    "M0_NoMask": "无掩码",
    "M1_Pos16": "位置编码16",
    "M2_Pos24": "位置编码24",
    "M3_8PlusSoft": "8层后软掩码",
    "M4_AdaptiveKTI": "自适应KTI",
    "M5_8PlusHard": "8层后硬掩码",
}

OCC_ZH = {
    0.0: "0%",
    0.1: "10%",
    0.2: "20%",
    0.3: "30%",
    0.4: "40%",
    0.5: "50%",
}


def _load_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["occlusion_ratio"] = df["occlusion_ratio"].astype(float)
    return df


def _pivot(df: pd.DataFrame, value_col: str) -> tuple[list[str], np.ndarray, list[float]]:
    """返回 (组显示名顺序, occ 列表, 矩阵 shape [n_group, n_occ])"""
    groups = sorted(df["group"].unique(), key=lambda g: list(GROUP_ZH.keys()).index(g) if g in GROUP_ZH else 99)
    occs = sorted(df["occlusion_ratio"].unique())
    mat = np.zeros((len(groups), len(occs)), dtype=float)
    labels = [GROUP_ZH.get(g, g) for g in groups]
    for i, g in enumerate(groups):
        for j, o in enumerate(occs):
            row = df[(df["group"] == g) & (df["occlusion_ratio"] == o)]
            mat[i, j] = float(row[value_col].iloc[0]) if len(row) else np.nan
    return labels, occs, mat


def _plot_grouped_bars(
    ax: plt.Axes,
    x_labels: list[str],
    occs: list[float],
    values: np.ndarray,
    ylabel: str,
    use_log: bool,
) -> None:
    n_g, n_o = values.shape
    x = np.arange(n_g, dtype=float)
    width = min(0.22, 0.8 / (n_o + 1))
    offsets = (np.arange(n_o) - (n_o - 1) / 2.0) * width

    colors = plt.cm.tab10(np.linspace(0, 0.92, n_o))
    for j, occ in enumerate(occs):
        heights = np.clip(values[:, j], 1e-6, None)
        ax.bar(
            x + offsets[j],
            heights,
            width * 0.92,
            label=OCC_ZH.get(float(occ), f"{occ:.0%}"),
            color=colors[j],
            edgecolor="0.25",
            linewidth=0.35,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    if use_log:
        ax.set_yscale("log")
    ax.legend(ncol=3, frameon=True, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs"
        / "eval_global"
        / "Ch5"
        / "occlusion_3dpw_n4096"
        / "occlusion_table.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录（默认同 csv 所在目录）",
    )
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument(
        "--linear",
        action="store_true",
        help="纵轴用线性刻度（与六组同图时矮柱会被压扁，建议配合 --exclude-broken）",
    )
    ap.add_argument(
        "--exclude-broken",
        action="store_true",
        help="不绘制 M3/M5（训练失效组），便于线性纵轴展示其余四组。",
    )
    ap.add_argument(
        "-q", "--quiet", action="store_true", help="不打印字体下载提示"
    )
    args = ap.parse_args()

    _setup_zh_font(verbose=not args.quiet)

    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_table(args.csv)
    if args.exclude_broken:
        df = df[~df["group"].isin(["M3_8PlusSoft", "M5_8PlusHard"])].copy()
    x_labels, occs, mpjpe = _pivot(df, "MPJPE")
    _, _, pamp = _pivot(df, "PA-MPJPE")

    use_log = not args.linear
    if args.linear and not args.exclude_broken and set(df["group"]) >= {
        "M3_8PlusSoft",
        "M5_8PlusHard",
    }:
        print(
            "提示：当前为线性纵轴且仍含 M3/M5，矮柱可能难以辨认；可加 --exclude-broken 或去掉 --linear。",
            file=sys.stderr,
        )

    fig_w, fig_h = 10.0, 4.6
    # 图1：MPJPE
    fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
    y_mp = (
        "平均关节位置误差(毫米,对数纵轴)"
        if use_log
        else "平均关节位置误差(毫米)"
    )
    _plot_grouped_bars(
        ax1,
        x_labels,
        occs,
        mpjpe,
        ylabel=y_mp,
        use_log=use_log,
    )
    p1 = out_dir / "ch5_occlusion_mpjpe_bars_zh.png"
    fig1.savefig(p1, dpi=args.dpi)
    plt.close(fig1)

    # 图2：PA-MPJPE
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
    y_pa = (
        "对齐后平均关节位置误差(毫米,对数纵轴)"
        if use_log
        else "对齐后平均关节位置误差(毫米)"
    )
    _plot_grouped_bars(
        ax2,
        x_labels,
        occs,
        pamp,
        ylabel=y_pa,
        use_log=use_log,
    )
    p2 = out_dir / "ch5_occlusion_pampjpe_bars_zh.png"
    fig2.savefig(p2, dpi=args.dpi)
    plt.close(fig2)

    print(f"已保存: {p1}\n{p2}")


if __name__ == "__main__":
    main()
