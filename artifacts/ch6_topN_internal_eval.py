#!/usr/bin/env python3
"""
从 artifacts/eval_unified/metrics_master.csv 中读取 chapter=ch6 的评测行，
按与 ch6_best_vs_baselines / unified_eval_batch 相同的 rank-sum 规则排序，
对前 N 个 checkpoint 运行 nvit.global_evaluator（人类学指标 + 内部熵/距离/秩/KTI）。

多卡: 任务级并行，每进程设置 CUDA_VISIBLE_DEVICES=单张物理卡，子进程内使用 --gpu 0。
与训练 DDP 不同，适合 10 个相互独立的整段评测。

例:
  python3 artifacts/ch6_topN_internal_eval.py --top 10 --gpus 0,1,2,3,4,5,6,7,8,9
  CH6_INTERNAL_GPUS=0,1,2,3 python3 artifacts/ch6_topN_internal_eval.py --top 10
  python3 artifacts/ch6_topN_internal_eval.py --top 10  # 单卡顺序: --gpu 0
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from shlex import quote as shquote

_REPO = Path(__file__).resolve().parents[1]
_ART = _REPO / "artifacts"
if str(_ART) not in sys.path:
    sys.path.insert(0, str(_ART))

import ch6_best_vs_baselines as c6  # noqa: E402

RANK_DATASETS = c6.RANK_DATASETS
load_ch6_records_from_master = c6.load_ch6_records_from_master
list_composite_ranked = c6.list_composite_ranked


def _safe_label_for_rank(
    pos: int, rank_sum: int, mean_rank: float, experiment: str, step: int | None
) -> str:
    exp_short = re.sub(r"[^0-9A-Za-z._-]+", "_", (experiment or "exp"))[:80].strip("_")
    s = f"ch6_r{pos:02d}_rs{rank_sum}_m{mean_rank:.3f}"
    if step is not None:
        s += f"_s{step}"
    if exp_short:
        s += f"__{exp_short}"
    return s


def _parse_gpus(s: str | None) -> list[str] | None:
    if not s or not str(s).strip():
        return None
    out = [p.strip() for p in s.split(",") if p.strip()]
    return out or None


def _build_cmd(
    args: argparse.Namespace,
    ck: str,
    label: str,
    *,
    cuda_visible: str | None,
) -> list[str]:
    # 在 CUDA_VISIBLE_DEVICES=单卡 时，对子进程用 --gpu 0
    gpu_id = "0" if cuda_visible is not None else str(args.gpu)
    return [
        args.python,
        "-m",
        "nvit.global_evaluator",
        "--chapter",
        args.chapter,
        "--checkpoint_path",
        ck,
        "--run_label",
        label,
        "--gpu",
        gpu_id,
        "--diag_batches",
        str(args.diag_batches),
        "--datasets",
        args.datasets,
    ]


def _run_subprocess(
    cmd: list[str], repo: Path, cuda_visible: str | None
) -> int:
    env = {**os.environ, "PYTHONPATH": f"{repo}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    p = subprocess.run(cmd, cwd=str(repo), env=env)
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ch6 按 composite rank 前 N 个模型跑 global_evaluator 内部指标（可 --gpus 多卡任务并行）"
    )
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="默认: <repo>/artifacts/eval_unified/metrics_master.csv",
    )
    ap.add_argument(
        "--rank-metric-3d",
        choices=("mode_re", "mode_mpjpe"),
        default="mode_re",
        help="与 ch6_best_vs_baselines 一致，默认 PA-MPJPE",
    )
    ap.add_argument("--top", type=int, default=10, help="前 N 个（rank-sum 最小）")
    ap.add_argument(
        "--chapter",
        type=str,
        default="Ch6A",
        choices=["Ch6A", "Ch6B"],
        help="写入 outputs/eval_global/<chapter>/",
    )
    ap.add_argument(
        "--gpu",
        type=str,
        default=os.environ.get("CH6_INTERNAL_GPU", "0"),
        help="单卡串行时使用的设备 id。使用 --gpus 时，每进程在隔离环境下用 --gpu 0。",
    )
    ap.add_argument(
        "--gpus",
        type=str,
        default=os.environ.get("CH6_INTERNAL_GPUS") or None,
        metavar="0,1,2",
        help="逗号分隔的物理 GPU id 列表。设置后分波多任务并行。环境: CH6_INTERNAL_GPUS。",
    )
    ap.add_argument(
        "--diag-batches", type=int, default=int(os.environ.get("CH6_INTERNAL_DIAG", "50"))
    )
    ap.add_argument(
        "--datasets", type=str, default="ALL", help="传给 global_evaluator"
    )
    ap.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="用于 -m nvit.global_evaluator 的解释器",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="只打印将运行的命令与排名，不执行"
    )
    ap.add_argument(
        "--out-plan",
        type=Path,
        default=None,
        help="将排名与将执行的命令写入该文本文件",
    )
    args = ap.parse_args()

    gpus = _parse_gpus(args.gpus)
    use_parallel = gpus is not None and len(gpus) >= 1
    nphys = len(gpus) if gpus is not None else 0
    if use_parallel and nphys < 1:
        use_parallel = False

    repo = _REPO
    csv_path = args.metrics_csv or (repo / "artifacts" / "eval_unified" / "metrics_master.csv")
    if not csv_path.is_file():
        raise SystemExit(f"找不到 metrics CSV: {csv_path}")

    records = load_ch6_records_from_master(csv_path, RANK_DATASETS)
    if not records:
        raise SystemExit(
            f"没有 ch6 记录（需 {RANK_DATASETS} 且 status=ok）。{csv_path}"
        )

    ranked = list_composite_ranked(records, args.rank_metric_3d, top_k=args.top)
    if not ranked:
        raise SystemExit("无法对 ch6 记录计算 composite 排名。")

    lines: list[str] = [
        f"# metrics: {csv_path}",
        f"# rank_metric_3d: {args.rank_metric_3d}",
        f"# selected: top {len(ranked)} of {len(records)} ch6 experiments",
    ]
    if use_parallel and gpus:
        lines.append(f"# multi-gpu: {','.join(gpus)}  (per-process CUDA_VISIBLE_DEVICES, cuda:0 inside)")
    lines.append("")

    # (pos, ck, label, exp) 为有效待跑任务
    work: list[tuple[int, str, str, str]] = []
    for pos, row in enumerate(ranked, start=1):
        ck = (row.get("checkpoint") or "").strip()
        exp = row.get("experiment") or ""
        step = row.get("step")
        rs = int(row.get("rank_sum", -1))
        mr = float(row.get("mean_rank", 0.0))
        label = _safe_label_for_rank(
            pos, rs, mr, str(exp), step if isinstance(step, int) else None
        )
        block = f"{pos}\trank_sum={rs}\tmean={mr:.4f}\tckpt={ck}\n  label={label}\n  exp={exp}"
        print(block)
        lines.append(block)

        if not ck or not Path(ck).is_file():
            err = f"\n[skip pos {pos}] 无有效 checkpoint 文件: {ck!r}\n  experiment={exp}"
            print(err, file=sys.stderr)
            lines.append(err)
            continue
        work.append((pos, ck, label, exp))

    if not work:
        raise SystemExit("没有可执行的 checkpoint 任务。")

    # 多线程写 lines/打印时与 wave 间隔离
    _out_lock = threading.Lock()

    def one_serial(pos: int, ck: str, label: str) -> int:
        cmd = _build_cmd(args, ck, label, cuda_visible=None)
        cmd_s = " ".join(shquote(x) for x in cmd)
        print(f"  {cmd_s}\n")
        lines.append(f"  CMD: {cmd_s}\n")
        if args.dry_run:
            return 0
        return _run_subprocess(cmd, repo, cuda_visible=None)

    def one_parallel(phys: str, pos: int, ck: str, label: str) -> int:
        cmd = _build_cmd(args, ck, label, cuda_visible=phys)
        env_note = f"CUDA_VISIBLE_DEVICES={shquote(phys)} " if phys else ""
        cmd_s = env_note + " ".join(shquote(x) for x in cmd)
        with _out_lock:
            print(f"  {cmd_s}\n", flush=True)
            lines.append(f"  CMD: {cmd_s}\n")
        if args.dry_run:
            return 0
        return _run_subprocess(cmd, repo, cuda_visible=phys)

    if not use_parallel or not gpus:
        for pos, ck, label, _e in work:
            rc = one_serial(pos, ck, label)
            if rc != 0:
                raise SystemExit(
                    f"global_evaluator 失败 (returncode={rc}): pos {pos} label {label!r}"
                )
    else:
        wave = 0
        for b in range(0, len(work), nphys):
            chunk = work[b : b + nphys]
            wave += 1
            print(
                f"--- wave {wave} ({len(chunk)} 卡并行) ---",
                file=sys.stderr,
            )
            lines.append(f"--- wave {wave} jobs {b + 1}-{b + len(chunk)} ---\n")
            with ThreadPoolExecutor(max_workers=len(chunk)) as ex:
                fmap = {
                    ex.submit(
                        one_parallel, gpus[i % nphys], pos, ck, label
                    ): (pos, gpus[i % nphys], label)
                    for i, (pos, ck, label, _e) in enumerate(chunk)
                }
                for fut in as_completed(fmap):
                    pos, gname, label = fmap[fut]
                    try:
                        rc = fut.result()
                    except Exception as e:
                        raise SystemExit(
                            f"子任务异常 pos {pos} GPU {gname} label {label!r}: {e}"
                        ) from e
                    if rc != 0:
                        raise SystemExit(
                            f"global_evaluator 失败 (returncode={rc}): pos {pos} GPU {gname} label {label!r}"
                        )

    if args.out_plan:
        args.out_plan.write_text("\n".join(lines), encoding="utf-8")
        print(f"计划已写入: {args.out_plan}")

    print(
        f"\n汇总: {repo / 'outputs' / 'eval_global' / args.chapter / 'summary.csv'}"
    )


if __name__ == "__main__":
    main()
