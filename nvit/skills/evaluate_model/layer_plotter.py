import os
import re
import logging
from collections.abc import Mapping
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from nvit.utils.path_utils import get_project_root


def _legend_label_from_run_name(run_name: str) -> str:
    """
    e.g. ch5_M4_AdaptiveKTI__2026-04-18_11-36-59 -> M4_AdaptiveKTI
    Strips 'ch*_' prefix and '__timestamp' tail when present; otherwise shortens as possible.
    """
    head = run_name.split("__", 1)[0] if "__" in run_name else run_name
    m = re.match(r"^ch[0-9A-Za-z]+_(.+)$", head)
    if m:
        return m.group(1)
    return head


def _run_sort_key(run_name: str) -> tuple:
    """
    稳定排序：先按 M0/M1/…，同章内再按目录名字符串，避免每次 iterdir 顺序不同导致色条与实验不对应。
    无 M 前缀的目录（如仅时间戳）排在该章曲线之后。
    """
    m = re.search(r"_M(\d+)_", run_name) or re.search(r"(?:^|_)M(\d+)_", run_name)
    if m:
        return (0, int(m.group(1)), run_name)
    m2 = re.search(r"^M(\d+)[_\s]", run_name)
    if m2:
        return (0, int(m2.group(1)), run_name)
    return (1, 0, run_name)


# 用于「仅两曲线」论文图（NVIT_LAYER_PLOT_ONLY_RUNS）
_DEFAULT_PLOT_LABELS: dict[str, str] = {
    "ch6_step492000": "step 492000 (NViT)",
    "hmr2_e35_1M_baseline_internal": "HMR2 baseline (e35 / 1M)",
}


def _parse_label_overrides_from_env() -> dict[str, str]:
    """
    环境变量 NVIT_LAYER_PLOT_LABELS: run_dir:显示名,run2:名2
    冒号在「显示名」中可不必支持；显示名用最后一个冒号后截断，或整段无冒号时跳过。
    """
    raw = (os.environ.get("NVIT_LAYER_PLOT_LABELS") or "").strip()
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.split(","):
        p = part.strip()
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        k, v = k.strip(), v.strip()
        if k and v:
            out[k] = v
    return out


def _only_run_names_from_env() -> set[str] | None:
    raw = (os.environ.get("NVIT_LAYER_PLOT_ONLY_RUNS") or "").strip()
    if not raw:
        return None
    return {p.strip() for p in raw.split(",") if p.strip()}


def _display_legend_map(run_names: list[str], forced_labels: dict[str, str] | None = None) -> dict[str, str]:
    """
    同一简名多目录（如两次 M0）时在图例中加区分，避免「色条在图上、不知对应哪次跑」。
    """
    from collections import defaultdict

    if forced_labels:
        base = {r: (forced_labels[r] if r in forced_labels else _legend_label_from_run_name(r)) for r in run_names}
    else:
        base = {r: _legend_label_from_run_name(r) for r in run_names}
    inv: dict[str, list[str]] = defaultdict(list)
    for r, b in base.items():
        inv[b].append(r)

    out: dict[str, str] = {}
    for b, rlist in inv.items():
        if len(rlist) == 1:
            out[rlist[0]] = b
            continue
        for r in sorted(rlist):
            if "__" in r and r.split("__", 1)[-1]:
                tag = r.split("__", 1)[-1]
            else:
                tag = r
            if len(tag) > 18:
                tag = tag[:8] + "…" + tag[-8:]
            out[r] = f"{b} 〔{tag}〕"
    return out


# 统一章节工作流：若某章存在“chapter_M*__*”规范目录，则优先只使用这套目录作对比图
_CANONICAL_M_RUN_TMPL = r"^{prefix}_M\d+_.+__.+$"


def _chapter_aliases(chapter: str) -> list[str]:
    c = (chapter or "").strip().casefold()
    if not c:
        return []
    out = {c}
    # Ch6A / Ch6B 同时允许 ch6_ 前缀
    m = re.match(r"^(ch\d+)[ab]$", c)
    if m:
        out.add(m.group(1))
    return sorted(out)


def _is_canonical_m_run(chapter: str, run_name: str) -> bool:
    for pfx in _chapter_aliases(chapter):
        pat = _CANONICAL_M_RUN_TMPL.format(prefix=re.escape(pfx))
        if re.match(pat, run_name, flags=re.IGNORECASE):
            return True
    return False


# 色觉友好；M5 固定绿色，避免与 M0 橙色系接近
_M_TAG = re.compile(r"_M(\d+)_")
_RUN_COLOR_BY_M: dict[int, str] = {
    0: "#E69F00",  # 橙
    1: "#56B4E9",  # 天蓝
    2: "#AB467E",  # 紫品红
    3: "#F0E442",  # 黄
    4: "#0072B2",  # 深蓝
    5: "#009E73",  # 绿
}

def _color_for_run_name(run_name: str, seq_index: int) -> str:
    m = _M_TAG.search(run_name)
    if m:
        k = int(m.group(1))
        if k in _RUN_COLOR_BY_M:
            return _RUN_COLOR_BY_M[k]
    return _PLOT_COLORS[seq_index % len(_PLOT_COLORS)]


# 回退/额外 run 用
_PLOT_COLORS = [
    "#D55E00", "#CC79A7", "#5BA39C", "#C77CFF", "#4E79A7", "#000000", "#2ca02c",
    "#9467bd", "#17becf",
]
_LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 3))]
_MARKERS = "os^vD>ph*+x"


# OFL, from notofonts/noto-cjk; 若 notofonts 未改发布文件，与 CDN 上当前体积一致
_NOTO_CJKSC_REGULAR_BYTES = 16_437_364

# OFL, from notofonts/noto-cjk; cached under ~/.cache/nvit/fonts
_NOTO_SC_OTF_CANDIDATES = [
    # 镜像可缓解 raw.githubusercontent 慢/超时；体积约 16MB
    "https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
]

# Setup logging before helpers that may log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LayerPlotter")


def _download_noto_cjksc_otf(dest: Path, urls: list[str]) -> bool:
    """
    优先用 curl 断点续传，直到 OTF 字节与官方一致，避免半包 OTF 无法被 matplotlib 注册。
    无 curl 时用 urllib 分块直写，仍易因超时得到不全文件；建议有 curl 或本地下好 NVIT_CJK_FONT。
    """
    ex = _NOTO_CJKSC_REGULAR_BYTES
    curl = shutil.which("curl")

    for url in urls:
        if not curl and dest.is_file():
            try:
                dest.unlink()
            except OSError:
                pass

        if curl:
            for _ in range(20):
                if dest.is_file() and dest.stat().st_size == ex:
                    return True
                try:
                    subprocess.run(
                        [
                            curl, "-fL", "-S", "-sS",
                            "--connect-timeout", "30",
                            "--max-time", "1200",
                            "-C", "-",
                            "-o", str(dest), url,
                        ],
                        timeout=1250,
                    )
                except (OSError, subprocess.TimeoutExpired) as e:
                    logger.warning("curl 取字库一步失败: %s", e)
                if dest.is_file() and dest.stat().st_size == ex:
                    return True
            try:
                if dest.is_file():
                    dest.unlink()
            except OSError:
                pass
            continue

        try:
            with urllib.request.urlopen(url, timeout=30) as resp, open(
                dest, "wb"
            ) as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        except (urllib.error.URLError, OSError) as e:
            logger.warning("CJK 字库 urllib 源失败 %s: %s", url[:64], e)
        if dest.is_file() and dest.stat().st_size == ex:
            return True
        if dest.is_file():
            try:
                dest.unlink()
            except OSError:
                pass

    return False


def _cjk_sans_serif_families() -> list[str]:
    """
    论文/中文图：需真实 CJK 字库。顺序：已装 CJK、NVIT_CJK_FONT、<repo>/assets/fonts/、~/.cache、首次下载 Noto CJK。
    若仅有 DejaVu 且无下载，将退回 DejaVu（可能显示为方框）。
    """
    for name in (
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
    ):
        try:
            path = font_manager.findfont(
                name, fallback_to_default=False, rebuild_if_missing=False
            )
        except (ValueError, OSError, RuntimeError):
            continue
        if not path or "dejavu" in str(path).lower():
            continue
        p = font_manager.FontProperties(family=name)
        return [p.get_name(), "DejaVu Sans"]

    env = os.environ.get("NVIT_CJK_FONT", "").strip()
    if env and Path(env).is_file():
        font_manager.fontManager.addfont(env)
        f = font_manager.FontProperties(fname=env).get_name()
        return [f, "DejaVu Sans"]

    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    otf: Path
    prj: Path | None
    try:
        prj = get_project_root() / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
    except (OSError, TypeError, ValueError):
        prj = None
    if prj is not None and prj.is_file():
        otf = prj
    else:
        fdir = cache / "nvit" / "fonts"
        fdir.mkdir(parents=True, exist_ok=True)
        otf = fdir / "NotoSansCJKsc-Regular.otf"
        if otf.is_file() and otf.stat().st_size < 1_000_000:
            try:
                otf.unlink()
            except OSError:
                pass
        if not otf.is_file():
            no_dl = os.environ.get("NVIT_NO_FONT_DOWNLOAD", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if no_dl:
                logger.info(
                    "已设置 NVIT_NO_FONT_DOWNLOAD，未下载字库。请用 NVIT_CJK_FONT=…、"
                    "或安装系统中文字体，或删除该环境变量以允许首次下载。"
                )
            else:
                logger.info(
                    "未检测到系统中文字体，将尝试拉取 Noto CJK（约 16MB）至 %s；"
                    "可设 NVIT_CJK_FONT=已有.otf，或 NVIT_NO_FONT_DOWNLOAD=1 跳过。",
                    otf,
                )
                if not _download_noto_cjksc_otf(otf, _NOTO_SC_OTF_CANDIDATES) or (
                    not otf.is_file()
                ):
                    logger.warning("无法取得 Noto CJK 字库，图中文可能为方框；见上文环境变量。")
            if not otf.is_file():
                return ["DejaVu Sans"]

    try:
        font_manager.fontManager.addfont(str(otf))
    except (OSError, ValueError) as e:
        logger.warning("CJK 字体注册失败，中文可能为方框: %s", e)
        return ["DejaVu Sans"]
    f = font_manager.FontProperties(fname=str(otf)).get_name()
    return [f, "DejaVu Sans"]


def _cjk_rccontext():
    return plt.rc_context(
        {
            "font.sans-serif": _cjk_sans_serif_families(),
            "axes.unicode_minus": False,
        }
    )


def generate_comparative_plots(
    chapter,
    output_base="/home/yangz/NViT-master/outputs/eval_global",
    only_runs: set[str] | None = None,
    run_label_overrides: Mapping[str, str] | None = None,
):
    """
    Scans the chapter directory for run subdirectories, extracting layer_metrics_Control.json
    and overlays them on a 2x2 layer-wise diagnostic grid.
    环境变量:
      NVIT_LAYER_PLOT_ONLY_RUNS: 仅绘制这些子目录/附加名，逗号分隔
      NVIT_LAYER_PLOT_LABELS: 可选图例，格式 run1:名1,run2:名2
    参数 only_runs / run_label_overrides 优先于环境变量。
    """
    chapter_dir = Path(output_base) / chapter
    if not chapter_dir.exists():
        logger.warning(f"Chapter directory {chapter_dir} does not exist.")
        return

    # Find all layer_metrics_Control.json
    candidate_metrics: dict[str, dict] = {}

    # We walk chapter_dir. Run subdirs are 1 depth deep.
    for run_dir in chapter_dir.iterdir():
        if not run_dir.is_dir():
            continue

        # Search in 'diagnostics' subfolder or root
        json_file = run_dir / "diagnostics" / "layer_metrics_Control.json"
        if not json_file.exists():
            json_file = run_dir / "layer_metrics_Control.json"

        if json_file.exists():
            with open(json_file, "r") as f:
                try:
                    data = json.load(f)
                    candidate_metrics[run_dir.name] = data
                except Exception as e:
                    logger.error(f"Failed to parse JSON for {run_dir.name}: {e}")

    # 附加已知 baseline（若存在）：HMR2 原始模型 internal layer diagnostics
    # 注：该文件来自历史 internal 诊断结果，不需要重新跑 checkpoint 推理即可叠加到图中。
    extra_sources: list[tuple[str, Path]] = [
        (
            "hmr2_e35_1M_baseline_internal",
            get_project_root() / "nvit" / "results" / "4D-Humans" / "layer_metrics_Control.json",
        ),
    ]
    for run_name, json_file in extra_sources:
        if run_name in candidate_metrics:
            continue
        if not json_file.exists():
            continue
        try:
            with open(json_file, "r") as f:
                candidate_metrics[run_name] = json.load(f)
            logger.info("Added extra baseline source: %s -> %s", run_name, json_file)
        except Exception as e:
            logger.warning("Failed to load extra baseline %s: %s", json_file, e)

    if not candidate_metrics:
        logger.warning(f"No layer metric diagnostics found in {chapter_dir}")
        return

    filter_set = only_runs if only_runs is not None else _only_run_names_from_env()
    if filter_set and len(filter_set) > 0:
        # 显式只画若干条曲线时，必须从 candidate_metrics 取，否则
        # 「仅 canonical M*」会丢掉 ch6_step492000 等非 M 目录。
        run_metrics = {k: v for k, v in candidate_metrics.items() if k in filter_set}
        for m in (x for x in filter_set if x not in candidate_metrics):
            logger.warning("NVIT_LAYER_PLOT_ONLY_RUNS: no metrics for %s (skipped)", m)
        if not run_metrics:
            logger.warning("After ONLY_RUNS filter, no runs left; abort plot.")
            return
        has_canonical = False
    else:
        has_canonical = any(_is_canonical_m_run(chapter, rn) for rn in candidate_metrics)
        if has_canonical:
            run_metrics = {
                rn: d for rn, d in candidate_metrics.items() if _is_canonical_m_run(chapter, rn)
            }
        else:
            run_metrics = candidate_metrics

    label_overrides: dict[str, str] = {}
    label_overrides.update(_DEFAULT_PLOT_LABELS)
    label_overrides.update(_parse_label_overrides_from_env())
    if run_label_overrides:
        label_overrides.update(dict(run_label_overrides))

    logger.info(
        "Plot source for %s: %d/%d runs (%s)",
        chapter,
        len(run_metrics),
        len(candidate_metrics),
        "canonical M-runs" if has_canonical else "all detected runs",
    )

    with _cjk_rccontext():
        # 稳定顺序 + 图例去重名，与颜色/线型一一对应
        run_order = sorted(
            list(run_metrics.items()), key=lambda kv: _run_sort_key(kv[0])
        )
        all_names = [a[0] for a in run_order]
        label_map = _display_legend_map(all_names, label_overrides)
        n_runs = len(run_order)

        # No figure-level title (论文插图通常单独编号说明)
        # 底部留给统一图例
        fig, axs = plt.subplots(2, 2, figsize=(15, 10.2))

        metrics_map = {
            "rank": (axs[0, 0], "有效秩（特征多样性）"),
            "entropy": (axs[0, 1], "香农熵（路径选择均匀性）"),
            "kmi": (axs[1, 0], "拓扑一致性指数（人体结构对齐）"),
            "dist": (axs[1, 1], "平均注意力距离（跨区域关联）"),
        }

        for i, (run_name, data) in enumerate(run_order):
            color = _color_for_run_name(run_name, i)
            ls = _LINE_STYLES[i % len(_LINE_STYLES)]
            mkr = _MARKERS[i % len(_MARKERS)]
            legend_label = label_map[run_name]
            z = 5 + i

            # Sort layers to ensure correct progression (0, 1, 2... 11)
            try:
                sorted_layers = sorted([int(k) for k in data.keys()])
            except ValueError:
                sorted_layers = sorted(data.keys())

            x_layers = [str(k) for k in sorted_layers]

            # Extract the mean scalar for each metric per layer
            run_plot_data = {"rank": [], "entropy": [], "kmi": [], "dist": []}

            for k_layer in sorted_layers:
                layer_data = data[str(k_layer)]
                for m_key in run_plot_data.keys():
                    if m_key == "kmi":
                        vals = None
                        for kti_key in ("kmi_edge_ratio", "kmi", "kti"):
                            v = layer_data.get(kti_key, [])
                            if v and len(v) > 0:
                                vals = v
                                break
                        if vals and len(vals) > 0:
                            run_plot_data[m_key].append(np.mean(vals))
                        else:
                            run_plot_data[m_key].append(np.nan)
                    else:
                        vals = layer_data.get(m_key, [])
                        if vals and len(vals) > 0:
                            run_plot_data[m_key].append(np.mean(vals))
                        else:
                            run_plot_data[m_key].append(np.nan)

            for m_key, (ax, title) in metrics_map.items():
                y_vals = run_plot_data[m_key]

                if not all(np.isnan(y) for y in y_vals):
                    ax.plot(
                        x_layers,
                        y_vals,
                        color=color,
                        linestyle=ls,
                        marker=mkr,
                        linewidth=2.2,
                        markersize=5.2,
                        markeredgewidth=0.8,
                        markeredgecolor="0.2",
                        clip_on=False,
                        zorder=z,
                        label=legend_label,
                    )

        # 统一图例（四子图曲线一致，避免子图小窗图例被裁切或色条对不上号）
        for ax, _ in metrics_map.values():
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        first_ax = axs[0, 0]
        h, l = first_ax.get_legend_handles_labels()
        ncol = 4 if n_runs > 4 else n_runs
        fig.legend(
            h,
            l,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncol,
            fontsize=10.5,
            frameon=True,
            edgecolor="0.5",
            fancybox=False,
            labelspacing=0.55,
            handlelength=2.4,
        )

        for ax, _ in metrics_map.values():
            ax.set_xlabel("层", fontsize=12)
            ax.set_ylabel("各层平均", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.5)

        for ax, t in metrics_map.values():
            ax.set_title(t, fontsize=13)
            ax.tick_params(axis="both", labelsize=10.5)

        try:
            fig.tight_layout(rect=(0, 0.12, 1, 0.99))
        except (ValueError, RuntimeError):
            fig.tight_layout()

        save_path = chapter_dir / "layer_metrics_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        plt.close()
    
    logger.info(f"✅ Successfully generated comparative layer plots: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Layer Metrics for a Chapter.")
    parser.add_argument("--chapter", type=str, required=True, help="Chapter name (e.g., Ch6A)")
    parser.add_argument(
        "--output-base",
        type=str,
        default="",
        help="eval_global 父目录，默认同仓库 outputs/eval_global",
    )
    parser.add_argument(
        "--only-runs",
        type=str,
        default="",
        help="逗号分隔的 run 子目录名，等价于 NVIT_LAYER_PLOT_ONLY_RUNS",
    )
    args = parser.parse_args()
    if args.output_base.strip():
        out_base = Path(args.output_base)
    else:
        out_base = get_project_root() / "outputs" / "eval_global"
    oruns = {p.strip() for p in args.only_runs.split(",") if p.strip()} if args.only_runs.strip() else None
    generate_comparative_plots(str(args.chapter), out_base, only_runs=oruns)
