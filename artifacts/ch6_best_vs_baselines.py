#!/usr/bin/env python3
"""
从 artifacts/eval_unified/metrics_master.csv 中筛选 chapter=ch6 的评测行，
按与 scripts/unified_eval_batch.py 相同的「多数据集 rank-sum」规则选出 composite best，
并与论文表中的 METRO / Mesh Graphormer 对齐打印对比表。

（与 run_ch6_external_mnt_nvit_compare.sh 的分工：后者用于 mnt 镜像上外部权重的
全量 standard_eval 与 ch6_external_vs_ch6best.* 产物；**日常改表、对齐论文行不必每次跑
那套 bash**，把 unified 评测写进 metrics_master 后，用本脚本出表即可。）

用法:
  python3 artifacts/ch6_best_vs_baselines.py
  python3 artifacts/ch6_best_vs_baselines.py --metrics-csv path/to/metrics_master.csv
  python3 artifacts/ch6_best_vs_baselines.py --rank-metric-3d mode_mpjpe   # 用 MPJPE 做排名而非 PA
  python3 artifacts/ch6_best_vs_baselines.py --table-mode single --single-checkpoint-contains ch6   # 仅一行 NViT
  python3 artifacts/ch6_best_vs_baselines.py --table-mode compare3 --full-train-checkpoint-contains ch6_full  # 再加全量训练 composite 一行
  python3 artifacts/ch6_best_vs_baselines.py --bench-speed   # 增加「推理速度」列：NViT/hmr2 行实测；METRO/Mesh 读 outputs/.../metro_meshg_inference_speed.csv 或环境变量
  NVIT_CH6_BASELINE_METRO_MS=42.3 NVIT_CH6_BASELINE_MESHG_MS=44.1 python3 artifacts/ch6_best_vs_baselines.py  # 手工填入论文基线 ms/图

默认 --table-mode compare：与 METRO / Mesh Graphormer 对比 **两行** NViT（ch6 主 run composite + ch6_phase2_unfreeze5 composite）。compare3 为三行 NViT（+ 全量训练 run，路径子串需唯一）。

环境变量（可选）:
  NVIT_CH6_PARAMS_M              覆盖 NViT Params (M)，默认 208.128
  NVIT_CH6_TRAIN_PARAMS_M        主 run 行 Train Params (M)，默认 69.2
  NVIT_CH6_PHASE2_TRAIN_PARAMS_M phase2 行 Train Params (M)，默认 108.5
  NVIT_CH6_FULL_TRAIN_PARAMS_M   全量训练行 Train Params (M)，未设时与 --nvit-train-params-full-m 或总参 208.128 同逻辑
  CH6_PHASE2_SUBSTR              phase2 路径子串，默认 ch6_phase2_unfreeze5
  CH6_FULL_TRAIN_SUBSTR         全量训练 run 的路径子串（与 compare3 配合）
  CH6_PHASE2_UNFREEZE0_SUBSTR   compare 表尾「unfreeze0 from80k」行的路径子串（默认 ch6_phase2_unfreeze0_from80k）
  HMR2_MID_HEAVY_CKPT_SUBSTR    metrics 里外部 hmr2_mid_heavy 的 checkpoint 子串
  HMR2_MID_HEAVY_PARAMS_M       hmr2_mid_heavy 行 Params (M)，默认 416.331（剪枝 pth 实测）
  HMR2_MID_HEAVY_TRAIN_PARAMS_M 同上 Train Params (M)，默认 416.331（该权重全参可训）
  NVIT_CH6_UNFREEZE0_TRAIN_PARAMS_M  unfreeze0 行 Train Params (M)，默认与 phase2 相同
  NVIT_CH6_BASELINE_METRO_MS / NVIT_CH6_BASELINE_MESHG_MS  论文基线推理 ms/图（覆盖 CSV）
  基线（METRO / Mesh Graphormer）的 Train Params 与 Params 相同
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 默认：本仓库 ch6 Guided 模型总参数量、可训练参数量（百万）；可用环境变量覆盖
DEFAULT_NVIT_PARAMS_M = 208.128
# 旧 ch6 主 run（约 FREEZE_DEPTH=7 预训练+guided）与 ch6 phase2 微调（FREEZE_DEPTH=5，更多层参与训练）
DEFAULT_NVIT_TRAIN_PARAMS_M = 69.2
# 与论文表/终端表对齐的 phase2 行可训练参数量（四舍五入 108.5M）
DEFAULT_NVIT_PHASE2_TRAIN_PARAMS_M = 108.5
# 参考：ch6 phase2 unfreeze5 composite 某次表显快照（实际打印仍以 metrics_master 为准）
REFERENCE_NVIT_CH6_PHASE2_COMPOSITE: dict[str, Any] = {
    "method": "NViT (ch6, phase2 unfreeze5, best composite)",
    "params_m": 208.128,
    "train_params_m": 108.5,
    "h36m_mpjpe": 123.4,
    "h36m_pa_mpjpe": 70.8,
    "dpw_mpve": "—",
    "dpw_mpjpe": 117.9,
    "dpw_pa_mpjpe": 70.7,
}
# 默认认为 phase2 的 checkpoint 路径里含此子串
DEFAULT_CH6_PHASE2_SUBSTR = "ch6_phase2_unfreeze5"
# phase2 另一路：从 80k 起 unfreeze0（与 unfreeze5 分开占一行 composite）
DEFAULT_CH6_PHASE2_UNFREEZE0_SUBSTR = "ch6_phase2_unfreeze0_from80k"
# metrics 里 ch6_external + 该子串 对应 hmr2_mid_heavy 剪枝 pth 的评测
DEFAULT_HMR2_MID_HEAVY_CKPT_SUBSTR = "hmr2_mid_heavy"
# hmr2_mid_heavy.pth（按 ref Lightning 建网后 strict 装载）实测总参 / 可训练参一致
DEFAULT_HMR2_MID_HEAVY_PARAMS_M = 416.331
DEFAULT_HMR2_MID_HEAVY_TRAIN_PARAMS_M = 416.331

# 论文 / 公开结果表（用户给定）
BASELINES: list[dict[str, Any]] = [
    {
        "method": "METRO",
        "params_m": 231.8,
        "h36m_mpjpe": 54.0,
        "h36m_pa_mpjpe": 36.7,
        "dpw_mpve": 88.2,
        "dpw_mpjpe": 77.1,
        "dpw_pa_mpjpe": 47.9,
    },
    {
        "method": "Mesh Graphormer",
        "params_m": 215.7,
        "h36m_mpjpe": 51.2,
        "h36m_pa_mpjpe": 34.5,
        "dpw_mpve": 87.7,
        "dpw_mpjpe": 74.7,
        "dpw_pa_mpjpe": 45.6,
    },
]

RANK_DATASETS = ("3DPW-TEST", "H36M-VAL-P2")

DEFAULT_SPEED_CSV = (
    REPO_ROOT / "outputs" / "eval_global" / "Ch6A" / "metro_meshg_inference_speed.csv"
)
DEFAULT_PLOTS_DIR = REPO_ROOT / "outputs" / "eval_global" / "Ch6A"


def _load_metro_mesh_speed_csv(path: Path) -> dict[str, tuple[float, float]]:
    """
    读 run_metro_meshg_speed.sh 产出的 CSV；name -> (ms_per_image, fps)。
    跳过 SKIPPED 或无效行。
    """
    out: dict[str, tuple[float, float]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("name") or "").strip()
            ms_s = (row.get("ms_per_image") or "").strip()
            if not name or not ms_s or "SKIP" in ms_s.upper():
                continue
            try:
                ms = float(ms_s)
            except ValueError:
                continue
            if ms <= 0 or math.isnan(ms):
                continue
            fps = 1000.0 / ms
            out[name] = (ms, fps)
    return out


def _baseline_infer_ms_from_env() -> dict[str, tuple[float, float]]:
    """覆盖论文基线行的测速：NVIT_CH6_BASELINE_METRO_MS / NVIT_CH6_BASELINE_MESHG_MS（毫秒/图）。"""
    out: dict[str, tuple[float, float]] = {}
    for env_key, method in (
        ("NVIT_CH6_BASELINE_METRO_MS", "METRO"),
        ("NVIT_CH6_BASELINE_MESHG_MS", "Mesh Graphormer"),
    ):
        raw = (os.environ.get(env_key) or "").strip()
        if not raw:
            continue
        try:
            ms = float(raw)
        except ValueError:
            continue
        if ms > 0:
            out[method] = (ms, 1000.0 / ms)
    return out


def benchmark_hmr2_style_ckpt(
    ckpt: str,
    *,
    gpu: str,
    warmup: int,
    iters: int,
    batch_size: int,
    pth_ref_ckpt: str | None,
    use_amp: bool,
) -> tuple[float, float] | None:
    """
    与 artifacts/bench_hmr2_vs_nvit_inference.py 一致：batch_d['img'] 为 (B,3,256,256)。
    返回 (ms/图, fps)；失败返回 None。
    """
    try:
        import torch
        from nvit.utils.model_io import load_model_from_ckpt
    except Exception as e:
        print(f"# bench-speed: 导入失败，跳过: {e}", file=sys.stderr)
        return None

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("# bench-speed: 无 CUDA，跳过 NViT/HMR2 checkpoint 测速", file=sys.stderr)
        return None

    ck = Path(ckpt)
    if not ck.is_file():
        return None

    ref = (
        pth_ref_ckpt
        or os.environ.get("HMR2_PTH_REF_CKPT")
        or os.environ.get("NVIT_PTH_REF_CKPT")
    )
    if str(ck).lower().endswith((".pth", ".pt")) and not ref:
        print(
            "# bench-speed: 剪枝 .pth 需指定 --pth-ref-ckpt 或环境变量 HMR2_PTH_REF_CKPT",
            file=sys.stderr,
        )
        return None
    try:
        if ref and str(ck).lower().endswith((".pth", ".pt")):
            from nvit.utils.hmr2_pruned_pth import load_model_hmr2_pth_or_ckpt

            model = load_model_hmr2_pth_or_ckpt(str(ck), str(device), ref)
        else:
            model = load_model_from_ckpt(str(ck), device=str(device))
    except Exception as e:
        print(f"# bench-speed: 加载 {ck} 失败: {e}", file=sys.stderr)
        return None

    model.eval()
    img = torch.randn(batch_size, 3, 256, 256, device=device, dtype=torch.float32)
    batch_d = {"img": img}

    @torch.inference_mode()
    def _run() -> None:
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _ = model(batch_d)
        else:
            _ = model(batch_d)

    for _ in range(warmup):
        _run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _run()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    per_step_ms = 1000.0 * (t1 - t0) / iters
    per_img_ms = per_step_ms / batch_size
    fps = 1000.0 / per_img_ms if per_img_ms > 0 else 0.0
    return per_img_ms, fps


def attach_inference_speed(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """为表行写入 infer_ms / infer_fps：论文基线来自 env + metro_meshg CSV；NViT 等来自 --bench-speed。"""
    csv_lut = _load_metro_mesh_speed_csv(args.speed_csv)
    env_lut = _baseline_infer_ms_from_env()
    csv_by_method = {
        "METRO": csv_lut.get("METRO"),
        "Mesh Graphormer": csv_lut.get("MeshGraphormer"),
    }
    for row in rows:
        method = str(row.get("method") or "")
        if method in ("METRO", "Mesh Graphormer"):
            if method in env_lut:
                row["infer_ms"], row["infer_fps"] = env_lut[method]
            else:
                tup = csv_by_method.get(method)
                if tup:
                    row["infer_ms"], row["infer_fps"] = tup
            continue
        if not args.bench_speed:
            continue
        ck = row.get("checkpoint")
        if not ck:
            continue
        r = benchmark_hmr2_style_ckpt(
            str(ck),
            gpu=args.bench_gpu,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
            batch_size=args.bench_batch_size,
            pth_ref_ckpt=args.pth_ref_ckpt,
            use_amp=args.bench_amp,
        )
        if r:
            row["infer_ms"], row["infer_fps"] = r[0], r[1]


def _save_bar_chart(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("# 未安装 matplotlib，跳过绘图", file=sys.stderr)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(labels) + 2), 4.2))
    ax.bar(range(len(labels)), values, color="#4C72B0", edgecolor="0.25", linewidth=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"# plot: {out_path}")


def save_ch6_comparison_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """3DPW PA-MPJPE 与推理延迟两张柱状图（与 ch6_baseline_vs_nvit_speed.png 命名一致）。"""
    lab_pa: list[str] = []
    val_pa: list[float] = []
    for r in rows:
        v = r.get("dpw_pa_mpjpe")
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            lab_pa.append(str(r["method"])[:48])
            val_pa.append(float(v))
    if lab_pa:
        _save_bar_chart(
            lab_pa,
            val_pa,
            title="3DPW PA-MPJPE (mm)",
            ylabel="PA-MPJPE ↓",
            out_path=out_dir / "ch6_baseline_vs_nvit_best.png",
        )
    lab_ms: list[str] = []
    val_ms: list[float] = []
    for r in rows:
        v = r.get("infer_ms")
        if isinstance(v, (int, float)) and v is not None and float(v) > 0:
            lab_ms.append(str(r["method"])[:48])
            val_ms.append(float(v))
    if lab_ms:
        _save_bar_chart(
            lab_ms,
            val_ms,
            title="Inference latency (ms / image)",
            ylabel="ms/img ↓",
            out_path=out_dir / "ch6_baseline_vs_nvit_speed.png",
        )


def _step_from_checkpoint_path(ck: str) -> int | None:
    """从 Lightning ckpt 文件名里解析 step。"""
    name = Path(ck).name
    m = re.search(r"step[=:](\d+)", name, re.I)
    if not m:
        return None
    return int(m.group(1))


def _checkpoint_experiment_id(ck: str) -> str:
    """
    唯一、可读的 experiment 标签。metrics 里 ch6/step_XXXXXX 会在不同 run 上重复，故不用它聚合。
    """
    step = _step_from_checkpoint_path(ck)
    s = str(ck).replace("\\", "/")
    s_part = f"step_{step}" if step is not None else Path(ck).stem
    # 须先区分 unfreeze5 / unfreeze0：路径均含 ch6_phase2，不能仅用子串 ch6_phase2
    if "ch6_phase2_unfreeze5" in s:
        return f"ch6/phase2_unfreeze5/{s_part}"
    if "ch6_phase2_unfreeze0" in s or "unfreeze0_from80k" in s:
        return f"ch6/phase2_unfreeze0/{s_part}"
    if "/ch6/" in s and "ch5_prior" not in s and "ch6_phase" not in s:
        return f"ch6/main/{s_part}"
    return f"ch6/other/{s_part}"


def _float_cell(x: str) -> float | None:
    if x is None or str(x).strip() == "":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    if math.isnan(v):
        return None
    return v


def rank_value_for_dataset(
    metrics: dict[str, Any],
    ds_name: str,
    rank_3d: str,
) -> tuple[float | None, str]:
    """与 unified_eval_batch.rank_value_for_dataset 一致：用于 composite 排名的标量。"""
    if not metrics:
        return None, ""
    if ds_name in ("3DPW-TEST", "3DPW-OCC-TEST", "MPI-INF-3DHP-TEST", "H36M-VAL-P2"):
        key = rank_3d if rank_3d in metrics else "mode_re"
        if key not in metrics and "mode_mpjpe" in metrics:
            key = "mode_mpjpe"
        v = metrics.get(key)
    elif "mode_kpl2" in metrics:
        key = "mode_kpl2"
        v = metrics.get(key)
    else:
        for cand in ("mode_re", "mode_mpjpe", "mode_kpl2"):
            if cand in metrics:
                v = metrics[cand]
                key = cand
                break
        else:
            return None, ""
    if not isinstance(v, (int, float)):
        return None, ""
    fv = float(v)
    if math.isnan(fv):
        return None, ""
    return fv, key


def composite_rank_matrix(
    records: list[dict[str, Any]], rank_3d: str
) -> dict[int, dict[str, int]] | None:
    """
    与 unified_eval_batch.compute_composite_best 同构：返回每个 record 索引在各 dataset 上的名次（1=最好）。
    """
    if not records:
        return None
    n = len(records)
    per_ds: dict[str, list[tuple[int, float]]] = {}
    for i, rec in enumerate(records):
        results = rec.get("results") or {}
        for ds_name, m in results.items():
            val, _ = rank_value_for_dataset(m, ds_name, rank_3d)
            if val is None:
                continue
            per_ds.setdefault(ds_name, []).append((i, val))
    if not per_ds:
        return None
    ranks: dict[int, dict[str, int]] = {i: {} for i in range(n)}
    for ds_name, items in per_ds.items():
        items_sorted = sorted(items, key=lambda x: x[1])
        for rank, (idx, _) in enumerate(items_sorted, start=1):
            ranks[idx][ds_name] = rank
    return ranks


def list_composite_ranked(
    records: list[dict[str, Any]], rank_3d: str, top_k: int | None = None
) -> list[dict[str, Any]]:
    """
    按 rank-sum 升序（和 composite best 的胜负规则一致；并列比 mean_rank 升序）得到全表排序，可选截断 top_k。
    每条为一条「best 风格」的 dict（含 per_dataset_rank / checkpoint / experiment 等）。
    """
    ranks = composite_rank_matrix(records, rank_3d)
    if not ranks:
        return []
    n = len(records)
    scored: list[dict[str, Any]] = []
    for i in range(n):
        rdict = ranks[i]
        if not rdict:
            continue
        s = int(sum(rdict.values()))
        mean_r = float(s / len(rdict))
        scored.append(
            {
                "record_index": i,
                "rank_sum": s,
                "mean_rank": mean_r,
                "datasets_ranked": sorted(rdict.keys()),
                "per_dataset_rank": rdict,
                "checkpoint": records[i].get("checkpoint"),
                "step": records[i].get("step"),
                "experiment": records[i].get("experiment"),
                "json_path": records[i].get("json_path"),
                "results": records[i].get("results"),
            }
        )
    scored.sort(key=lambda x: (x["rank_sum"], x["mean_rank"]))
    if top_k is not None:
        scored = scored[:top_k]
    return scored


def compute_composite_best(
    records: list[dict[str, Any]],
    rank_3d: str,
) -> dict[str, Any] | None:
    """与 unified_eval_batch.compute_composite_best 一致（rank-sum，越小越好）。"""
    ranked = list_composite_ranked(records, rank_3d, top_k=1)
    if not ranked:
        return None
    return ranked[0]


def _parse_ts(s: str) -> datetime:
    s = (s or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.min


def load_metric_records_from_master(
    csv_path: Path,
    require_datasets: tuple[str, ...],
    *,
    chapter: str | tuple[str, ...] = "ch6",
    checkpoint_contains: str | None = None,
    family: str | None = None,
) -> list[dict[str, Any]]:
    """
    按 **checkpoint 路径** 聚合；同一 (checkpoint, dataset) 保留最新 timestamp 的一行。
    chapter 可为单个或元组；可选按 family、checkpoint 子串过滤（用于 ch6_external 等）。
    """
    chapters = (chapter,) if isinstance(chapter, str) else chapter
    raw: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("chapter") or "") not in chapters:
                continue
            if (row.get("status") or "").lower() != "ok":
                continue
            ck0 = (row.get("checkpoint") or "").strip()
            if checkpoint_contains and checkpoint_contains not in ck0:
                continue
            if family is not None and (row.get("family") or "") != family:
                continue
            raw.append(row)  # type: ignore[arg-type]

    raw.sort(key=lambda x: _parse_ts(x.get("timestamp_utc", "")), reverse=True)
    picked: dict[tuple[str, str], dict[str, str]] = {}
    for row in raw:
        ck = (row.get("checkpoint") or "").strip()
        if not ck:
            continue
        ds = row.get("dataset") or ""
        key = (ck, ds)
        if key not in picked:
            picked[key] = row

    by_ck: dict[str, dict[str, dict[str, float]]] = {}
    json_by: dict[str, str] = {}
    for (ck, _ds), row in picked.items():
        mp = _float_cell(row.get("MPJPE_mm", ""))
        pa = _float_cell(row.get("PA_MPJPE_mm", ""))
        if mp is None and pa is None:
            continue
        m: dict[str, float] = {}
        if mp is not None:
            m["mode_mpjpe"] = mp
        if pa is not None:
            m["mode_re"] = pa
        ds = row.get("dataset") or ""
        by_ck.setdefault(ck, {})[ds] = m
        if row.get("json_path"):
            json_by[ck] = row["json_path"]

    records: list[dict[str, Any]] = []
    for ck, results in sorted(by_ck.items()):
        if not all(ds in results for ds in require_datasets):
            continue
        step = _step_from_checkpoint_path(ck)
        rec = {
            "experiment": _checkpoint_experiment_id(ck),
            "step": step,
            "checkpoint": ck,
            "json_path": json_by.get(ck, ""),
            "results": results,
        }
        records.append(rec)
    return records


def load_ch6_records_from_master(
    csv_path: Path,
    require_datasets: tuple[str, ...],
) -> list[dict[str, Any]]:
    """chapter=ch6 的 NViT 主池（与 load_metric_records_from_master 等价）。"""
    return load_metric_records_from_master(csv_path, require_datasets, chapter="ch6")


def filter_ch6_records(
    records: list[dict[str, Any]],
    checkpoint_contains: str | None = None,
    checkpoint_excludes: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    excl: list[str] = []
    if isinstance(checkpoint_excludes, str) and checkpoint_excludes.strip():
        excl = [checkpoint_excludes]
    elif isinstance(checkpoint_excludes, list):
        excl = [e for e in checkpoint_excludes if (e and str(e).strip())]
    out: list[dict[str, Any]] = []
    for r in records:
        ck = str(r.get("checkpoint") or "")
        if checkpoint_contains and checkpoint_contains not in ck:
            continue
        if any(x in ck for x in excl):
            continue
        out.append(r)
    return out


def _nvit_table_row(
    best: dict[str, Any],
    method: str,
    params_m: float,
    train_params_m: float,
) -> dict[str, Any]:
    res = best.get("results") or {}
    h36 = res.get("H36M-VAL-P2") or {}
    dpw = res.get("3DPW-TEST") or {}
    return {
        "method": method,
        "params_m": round(params_m, 3),
        "train_params_m": train_params_m,
        "h36m_mpjpe": h36.get("mode_mpjpe"),
        "h36m_pa_mpjpe": h36.get("mode_re"),
        "dpw_mpve": "—",
        "dpw_mpjpe": dpw.get("mode_mpjpe"),
        "dpw_pa_mpjpe": dpw.get("mode_re"),
        "checkpoint": best.get("checkpoint") or "",
    }


def fmt(v: float | None, missing: str = "—") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return missing
    return f"{v:.1f}"


def _fmt_mpve_cell(row: dict[str, Any]) -> str:
    v = row.get("dpw_mpve")
    if isinstance(v, str):
        return v
    return fmt(v if isinstance(v, (int, float)) else None)


def _train_params_str(row: dict[str, Any]) -> str:
    """NViT 用 train_params_m；基线未设时与 Params (M) 相同。"""
    if "train_params_m" in row and row["train_params_m"] is not None:
        t = float(row["train_params_m"])
        return f"{t:.1f}"
    return str(row["params_m"])


def _fmt_speed_cell(row: dict[str, Any]) -> str:
    ms = row.get("infer_ms")
    fps = row.get("infer_fps")
    if ms is None or (isinstance(ms, float) and (math.isnan(ms) or ms <= 0)):
        return "—"
    ms_f = float(ms)
    if fps is not None and not (isinstance(fps, float) and math.isnan(fps)):
        return f"{ms_f:.2f} ms ({float(fps):.1f} fps)"
    return f"{ms_f:.2f} ms"


def _table_row_cells(row: dict[str, Any]) -> list[str]:
    return [
        str(row["method"]),
        str(row["params_m"]),
        _train_params_str(row),
        fmt(row.get("h36m_mpjpe")),
        fmt(row.get("h36m_pa_mpjpe")),
        _fmt_mpve_cell(row),
        fmt(row.get("dpw_mpjpe")),
        fmt(row.get("dpw_pa_mpjpe")),
        _fmt_speed_cell(row),
    ]


def print_table(rows: list[dict[str, Any]], col_gap: int = 2) -> None:
    """制表：按列取最大显示宽度，空格左对齐（等宽终端下各列竖线对齐）。"""
    headers = [
        "Method",
        "Params (M)",
        "Train Params (M)",
        "H36M MPJPE ↓",
        "H36M PA-MPJPE ↓",
        "3DPW MPVE ↓",
        "3DPW MPJPE ↓",
        "3DPW PA-MPJPE ↓",
        "推理速度",
    ]
    data_rows = [_table_row_cells(r) for r in rows]
    n = len(headers)
    widths = [len(headers[i]) for i in range(n)]
    for cells in data_rows:
        for i in range(n):
            widths[i] = max(widths[i], len(cells[i]))
    gap = " " * col_gap

    def _line(cells: list[str]) -> str:
        return gap.join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print(_line(headers))
    for cells in data_rows:
        print(_line(cells))


def _table_rows_with_baselines(tail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """复制 BASELINES 字典，避免 attach_inference_speed 污染模块级常量。"""
    return [{**b} for b in BASELINES] + tail


def emit_table_with_speed_and_plots(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    attach_inference_speed(rows, args)
    print_table(rows)
    if not args.no_plots:
        save_ch6_comparison_plots(rows, args.plots_dir)


def optional_extra_compare_rows(
    csv_path: Path,
    records: list[dict[str, Any]],
    rank_3d: str,
    params_m: float,
    train_phase2: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    compare / compare3 表尾可选两行：hmr2_mid_heavy（ch6_external）、
    ch6_phase2_unfreeze0_from80k composite best。
    """
    row_h: dict[str, Any] | None = None
    row_u0: dict[str, Any] | None = None
    if not args.omit_hmr2_mid_heavy:
        sub = (args.hmr2_mid_heavy_substr or "").strip() or DEFAULT_HMR2_MID_HEAVY_CKPT_SUBSTR
        hrecs = load_metric_records_from_master(
            csv_path,
            RANK_DATASETS,
            chapter="ch6_external",
            family="external_mnt",
            checkpoint_contains=sub,
        )
        bh = compute_composite_best(hrecs, rank_3d) if hrecs else None
        if bh:
            row_h = _nvit_table_row(
                bh,
                args.hmr2_method_label,
                float(args.hmr2_params_m),
                float(args.hmr2_train_params_m),
            )
            print(
                f"# best hmr2_mid_heavy: experiment={bh.get('experiment')} "
                f"step={bh.get('step')} rank_sum={bh.get('rank_sum')}"
            )
            print(f"# checkpoint: {bh.get('checkpoint')}")
        else:
            print(
                f"# 警告: metrics 中无 ch6_external + external_mnt + ckpt 含 {sub!r} 的完整"
                f" {RANK_DATASETS} 行，跳过 hmr2_mid_heavy 表行。",
                file=sys.stderr,
            )
    if not args.omit_phase2_unfreeze0:
        u0s = (args.phase2_unfreeze0_substr or "").strip() or DEFAULT_CH6_PHASE2_UNFREEZE0_SUBSTR
        u0_recs = filter_ch6_records(records, checkpoint_contains=u0s)
        bu0 = compute_composite_best(u0_recs, rank_3d) if u0_recs else None
        train_u0 = float(
            args.nvit_train_params_unfreeze0_m
            if args.nvit_train_params_unfreeze0_m is not None
            else os.environ.get(
                "NVIT_CH6_UNFREEZE0_TRAIN_PARAMS_M", str(train_phase2)
            )
        )
        if bu0:
            row_u0 = _nvit_table_row(
                bu0, args.nvit_method_label_unfreeze0, params_m, train_u0
            )
            print(
                f"# best unfreeze0 from80k: experiment={bu0.get('experiment')} "
                f"step={bu0.get('step')} rank_sum={bu0.get('rank_sum')}"
            )
            print(f"# checkpoint: {bu0.get('checkpoint')}")
        else:
            print(
                f"# 警告: 无 path 含 {u0s!r} 的 ch6 composite 记录，跳过 unfreeze0 表行。",
                file=sys.stderr,
            )
    return row_h, row_u0


def main() -> None:
    ap = argparse.ArgumentParser(description="ch6 composite best from metrics_master vs METRO table")
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="默认: <NViT-master>/artifacts/eval_unified/metrics_master.csv",
    )
    ap.add_argument(
        "--rank-metric-3d",
        choices=("mode_re", "mode_mpjpe"),
        default="mode_re",
        help="composite rank-sum 使用的 3D 指标（与 unified_eval_batch --rank-metric-3d 一致，默认 PA-MPJPE）",
    )
    ap.add_argument(
        "--table-mode",
        choices=("compare", "single", "compare3"),
        default="compare",
        help="compare: METRO、Mesh、主 run composite、phase2 unfreeze5 composite；默认再追加 hmr2_mid_heavy（ch6_external）"
        "与 ch6_phase2_unfreeze0_from80k composite 两行（可用 --omit-* 关闭）。"
        "compare3: 在 compare 基础上再增加「全量训练」composite 一行（需 --full-train-checkpoint-contains）。"
        "single: 仅一行 NViT，由 --single-checkpoint-contains 筛选。",
    )
    ap.add_argument(
        "--phase2-checkpoint-contains",
        default=os.environ.get("CH6_PHASE2_SUBSTR", DEFAULT_CH6_PHASE2_SUBSTR),
        help="table-mode=compare 或 compare3 时 phase2 行的 checkpoint 子串（默认 ch6_phase2_unfreeze5）",
    )
    ap.add_argument(
        "--single-checkpoint-contains",
        default="",
        help="table-mode=single 时只保留路径含此子串的 checkpoint 再算 composite；空则使用全部 ch6。",
    )
    ap.add_argument(
        "--nvit-method-label-legacy",
        default="NViT (ch6, best composite, main run)",
        help="table-mode=compare 时旧 ch6 行 Method 列",
    )
    ap.add_argument(
        "--nvit-method-label-phase2",
        default="NViT (ch6, phase2 unfreeze5, best composite)",
        help="table-mode=compare 时 phase2 行 Method 列",
    )
    ap.add_argument(
        "--nvit-method-label",
        default="NViT (ch6, best composite)",
        help="table-mode=single 时 NViT 行 Method 列",
    )
    ap.add_argument(
        "--nvit-train-params-m",
        type=float,
        default=None,
        help="single 模式 NViT Train Params (M)。默认 69.2 或 NVIT_CH6_TRAIN_PARAMS_M",
    )
    ap.add_argument(
        "--nvit-train-params-legacy-m",
        type=float,
        default=None,
        help="compare 模式主 run 行，默认 69.2 或 NVIT_CH6_TRAIN_PARAMS_M",
    )
    ap.add_argument(
        "--nvit-train-params-phase2-m",
        type=float,
        default=None,
        help="compare 模式 phase2 行，默认 108.5 或 NVIT_CH6_PHASE2_TRAIN_PARAMS_M",
    )
    ap.add_argument(
        "--omit-hmr2-mid-heavy",
        action="store_true",
        help="compare/compare3 时不追加 hmr2_mid_heavy（ch6_external）行",
    )
    ap.add_argument(
        "--omit-phase2-unfreeze0",
        action="store_true",
        help="compare/compare3 时不追加 ch6_phase2_unfreeze0_from80k composite 行",
    )
    ap.add_argument(
        "--hmr2-mid-heavy-substr",
        default=os.environ.get(
            "HMR2_MID_HEAVY_CKPT_SUBSTR", DEFAULT_HMR2_MID_HEAVY_CKPT_SUBSTR
        ),
        help="metrics 中 hmr2_mid_heavy 的 checkpoint 路径子串",
    )
    ap.add_argument(
        "--phase2-unfreeze0-substr",
        default=os.environ.get(
            "CH6_PHASE2_UNFREEZE0_SUBSTR", DEFAULT_CH6_PHASE2_UNFREEZE0_SUBSTR
        ),
        help="unfreeze0 from80k 实验在 checkpoint 路径中的子串",
    )
    ap.add_argument(
        "--hmr2-method-label",
        default="hmr2_mid_heavy (pruned pth, external)",
        help="hmr2_mid_heavy 行的 Method 列",
    )
    ap.add_argument(
        "--nvit-method-label-unfreeze0",
        default="NViT (ch6, phase2 unfreeze0 from80k, best composite)",
        help="unfreeze0 from80k 行的 Method 列",
    )
    ap.add_argument(
        "--hmr2-params-m",
        type=float,
        default=float(
            os.environ.get(
                "HMR2_MID_HEAVY_PARAMS_M", str(DEFAULT_HMR2_MID_HEAVY_PARAMS_M)
            )
        ),
        help=f"hmr2_mid_heavy 行 Params (M)，默认 {DEFAULT_HMR2_MID_HEAVY_PARAMS_M}（可 env 覆盖）",
    )
    ap.add_argument(
        "--hmr2-train-params-m",
        type=float,
        default=float(
            os.environ.get(
                "HMR2_MID_HEAVY_TRAIN_PARAMS_M",
                str(DEFAULT_HMR2_MID_HEAVY_TRAIN_PARAMS_M),
            )
        ),
        help=f"hmr2_mid_heavy 行 Train Params (M)，默认 {DEFAULT_HMR2_MID_HEAVY_TRAIN_PARAMS_M}",
    )
    ap.add_argument(
        "--nvit-train-params-unfreeze0-m",
        type=float,
        default=None,
        help="unfreeze0 行 Train Params (M)；默认与 phase2 或 NVIT_CH6_UNFREEZE0_TRAIN_PARAMS_M",
    )
    ap.add_argument(
        "--full-train-checkpoint-contains",
        default=os.environ.get("CH6_FULL_TRAIN_SUBSTR", "").strip(),
        help="table-mode=compare3 时，全量训练 run 在 checkpoint 路径中必须包含的子串（如 ch6_full 或你的 run 目录名）",
    )
    ap.add_argument(
        "--nvit-method-label-full",
        default="NViT (ch6, full-train, best composite)",
        help="compare3 时第三行 NViT 的 Method 列",
    )
    ap.add_argument(
        "--nvit-train-params-full-m",
        type=float,
        default=None,
        help="compare3 全量训练行 Train Params (M)；默认与总参 NVIT_CH6_PARAMS_M 相同，或环境 NVIT_CH6_FULL_TRAIN_PARAMS_M",
    )
    ap.add_argument(
        "--bench-speed",
        action="store_true",
        help="对 NViT / hmr2_mid_heavy 等待测行用 GPU 跑前向计时（256×256，与 bench_hmr2_vs_nvit_inference 一致）",
    )
    ap.add_argument(
        "--speed-csv",
        type=Path,
        default=DEFAULT_SPEED_CSV,
        help="METRO / Mesh Graphormer 的 ms/图 CSV（artifacts/run_metro_meshg_speed.sh 产出）",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="保存 ch6_baseline_vs_nvit_best.png / ch6_baseline_vs_nvit_speed.png",
    )
    ap.add_argument("--no-plots", action="store_true", help="不生成柱状图")
    ap.add_argument("--bench-gpu", type=str, default="0")
    ap.add_argument("--bench-warmup", type=int, default=20)
    ap.add_argument("--bench-iters", type=int, default=80)
    ap.add_argument("--bench-batch-size", type=int, default=1)
    ap.add_argument("--bench-amp", action="store_true", help="CUDA autocast fp16")
    ap.add_argument(
        "--pth-ref-ckpt",
        type=str,
        default=None,
        help="剪枝 .pth 时的参考 Lightning ckpt（默认同 HMR2_PTH_REF_CKPT）",
    )
    args = ap.parse_args()

    repo = REPO_ROOT
    csv_path = args.metrics_csv or (repo / "artifacts" / "eval_unified" / "metrics_master.csv")
    if not csv_path.is_file():
        raise SystemExit(f"找不到 metrics CSV: {csv_path}")

    records = load_ch6_records_from_master(csv_path, RANK_DATASETS)
    if not records:
        raise SystemExit(
            f"没有可用的 ch6 记录（需同时含 {RANK_DATASETS} 且 status=ok）。请检查: {csv_path}"
        )

    params_m = float(os.environ.get("NVIT_CH6_PARAMS_M", str(DEFAULT_NVIT_PARAMS_M)))
    train_legacy = float(
        args.nvit_train_params_legacy_m
        if args.nvit_train_params_legacy_m is not None
        else os.environ.get("NVIT_CH6_TRAIN_PARAMS_M", str(DEFAULT_NVIT_TRAIN_PARAMS_M))
    )
    train_phase2 = float(
        args.nvit_train_params_phase2_m
        if args.nvit_train_params_phase2_m is not None
        else os.environ.get("NVIT_CH6_PHASE2_TRAIN_PARAMS_M", str(DEFAULT_NVIT_PHASE2_TRAIN_PARAMS_M))
    )

    if args.table_mode == "compare3":
        full_sub = (args.full_train_checkpoint_contains or "").strip()
        if not full_sub:
            raise SystemExit(
                "compare3 需要 --full-train-checkpoint-contains=…（或环境 CH6_FULL_TRAIN_SUBSTR）"
                "，为全量训练 run 在 metrics 里 checkpoint 路径的唯一定位子串。"
            )
        p2 = args.phase2_checkpoint_contains or DEFAULT_CH6_PHASE2_SUBSTR
        u0s = (args.phase2_unfreeze0_substr or "").strip() or DEFAULT_CH6_PHASE2_UNFREEZE0_SUBSTR
        legacy_recs = filter_ch6_records(
            records, checkpoint_excludes=[p2, full_sub, u0s]
        )
        phase2_recs = filter_ch6_records(
            records, checkpoint_contains=p2
        )
        full_recs = filter_ch6_records(
            records, checkpoint_contains=full_sub
        )
        if not full_recs:
            raise SystemExit(
                f"compare3: 无 path 含 {full_sub!r} 的记录；请先在 metrics_master 里纳入该 run 的 3DPW+H36M 评测。"
            )
        if not phase2_recs:
            raise SystemExit(
                f"compare3 需要 path 含 {p2!r} 的 phase2 记录；请检查 metrics 或 --phase2-checkpoint-contains。"
            )
        if not legacy_recs:
            print(
                f"# 警告: 无同时不含 {p2!r} 与 {full_sub!r} 的 ch6 主 run 行。",
                file=sys.stderr,
            )
        best_leg = (
            compute_composite_best(legacy_recs, args.rank_metric_3d)
            if legacy_recs
            else None
        )
        best_p2 = compute_composite_best(phase2_recs, args.rank_metric_3d)
        best_full = compute_composite_best(full_recs, args.rank_metric_3d)
        if not best_p2 or not best_full:
            raise SystemExit("无法计算 phase2 或 full-train 的 composite best。")
        if not best_leg:
            raise SystemExit("无法计算主 run composite best；请检查 metrics 或放宽筛选。")
        train_full = float(
            args.nvit_train_params_full_m
            if args.nvit_train_params_full_m is not None
            else os.environ.get("NVIT_CH6_FULL_TRAIN_PARAMS_M", str(params_m))
        )
        row_leg = _nvit_table_row(
            best_leg, args.nvit_method_label_legacy, params_m, train_legacy
        )
        row_p2 = _nvit_table_row(
            best_p2, args.nvit_method_label_phase2, params_m, train_phase2
        )
        row_full = _nvit_table_row(
            best_full, args.nvit_method_label_full, params_m, train_full
        )
        print(f"# metrics: {csv_path}")
        print(f"# rank_metric_3d: {args.rank_metric_3d}")
        print(
            f"# ch6: total {len(records)} | main (excl. {p2!r} & {full_sub!r} & {u0s!r}): {len(legacy_recs)} | "
            f"phase2: {len(phase2_recs)} | full: {len(full_recs)}"
        )
        print(
            f"# best main: experiment={best_leg.get('experiment')} step={best_leg.get('step')} rank_sum={best_leg.get('rank_sum')}"
        )
        print(f"# checkpoint: {best_leg.get('checkpoint')}")
        print(
            f"# best phase2: experiment={best_p2.get('experiment')} step={best_p2.get('step')} rank_sum={best_p2.get('rank_sum')}"
        )
        print(f"# checkpoint: {best_p2.get('checkpoint')}")
        print(
            f"# best full-train: experiment={best_full.get('experiment')} step={best_full.get('step')} rank_sum={best_full.get('rank_sum')}"
        )
        print(f"# checkpoint: {best_full.get('checkpoint')}")
        row_h, row_u0 = optional_extra_compare_rows(
            csv_path, records, args.rank_metric_3d, params_m, train_phase2, args
        )
        print()
        tail: list[dict[str, Any]] = [row_leg, row_p2, row_full]
        if row_h:
            tail.append(row_h)
        if row_u0:
            tail.append(row_u0)
        emit_table_with_speed_and_plots(_table_rows_with_baselines(tail), args)
        return

    if args.table_mode == "single":
        recs = records
        if args.single_checkpoint_contains:
            recs = filter_ch6_records(recs, checkpoint_contains=args.single_checkpoint_contains)
        if not recs:
            raise SystemExit("single 模式：筛选后无记录，请改 --single-checkpoint-contains。")
        best = compute_composite_best(recs, args.rank_metric_3d)
        if not best:
            raise SystemExit("无法计算 composite best。")
        train_m = (
            float(args.nvit_train_params_m)
            if args.nvit_train_params_m is not None
            else float(os.environ.get("NVIT_CH6_TRAIN_PARAMS_M", str(DEFAULT_NVIT_TRAIN_PARAMS_M)))
        )
        nvit_row = _nvit_table_row(best, args.nvit_method_label, params_m, train_m)
        print(f"# metrics: {csv_path}")
        print(f"# rank_metric_3d: {args.rank_metric_3d}")
        print(f"# ch6 checkpoints (after filter): {len(recs)}")
        print(
            f"# best composite: experiment={best.get('experiment')} step={best.get('step')} rank_sum={best.get('rank_sum')}"
        )
        print(f"# checkpoint: {best.get('checkpoint')}")
        print(f"# json: {best.get('json_path')}")
        print()
        emit_table_with_speed_and_plots(_table_rows_with_baselines([nvit_row]), args)
        return

    p2 = args.phase2_checkpoint_contains or DEFAULT_CH6_PHASE2_SUBSTR
    u0s = (args.phase2_unfreeze0_substr or "").strip() or DEFAULT_CH6_PHASE2_UNFREEZE0_SUBSTR
    legacy_recs = filter_ch6_records(records, checkpoint_excludes=[p2, u0s])
    phase2_recs = filter_ch6_records(records, checkpoint_contains=p2)
    if not phase2_recs:
        raise SystemExit(
            f"compare 模式需要 path 含 {p2!r} 的 phase2 记录；请检查 metrics 或 --phase2-checkpoint-contains。"
        )
    if not legacy_recs:
        print(
            f"# 警告: 无不含 {p2!r} 的 ch6 主 run 行，可能 metrics 里只有 phase2。",
            file=sys.stderr,
        )

    best_leg = compute_composite_best(legacy_recs, args.rank_metric_3d) if legacy_recs else None
    best_p2 = compute_composite_best(phase2_recs, args.rank_metric_3d)
    if not best_p2:
        raise SystemExit("无法计算 phase2 composite best。")
    if not best_leg:
        raise SystemExit("无法计算主 run composite best。")

    row_leg = _nvit_table_row(best_leg, args.nvit_method_label_legacy, params_m, train_legacy)
    row_p2 = _nvit_table_row(best_p2, args.nvit_method_label_phase2, params_m, train_phase2)

    print(f"# metrics: {csv_path}")
    print(f"# rank_metric_3d: {args.rank_metric_3d}")
    print(
        f"# ch6 checkpoints: total {len(records)} | main (excl. {p2!r} & {u0s!r}): {len(legacy_recs)} | "
        f"phase2 (unfreeze5): {len(phase2_recs)}"
    )
    print(
        f"# best main: experiment={best_leg.get('experiment')} step={best_leg.get('step')} rank_sum={best_leg.get('rank_sum')}"
    )
    print(f"# checkpoint: {best_leg.get('checkpoint')}")
    print(
        f"# best phase2: experiment={best_p2.get('experiment')} step={best_p2.get('step')} rank_sum={best_p2.get('rank_sum')}"
    )
    print(f"# checkpoint: {best_p2.get('checkpoint')}")
    row_h, row_u0 = optional_extra_compare_rows(
        csv_path, records, args.rank_metric_3d, params_m, train_phase2, args
    )
    print()
    tail: list[dict[str, Any]] = [row_leg, row_p2]
    if row_h:
        tail.append(row_h)
    if row_u0:
        tail.append(row_u0)
    emit_table_with_speed_and_plots(_table_rows_with_baselines(tail), args)


if __name__ == "__main__":
    main()
