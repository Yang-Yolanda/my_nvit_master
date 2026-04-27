#!/usr/bin/env python3
"""
Ch5 六组消融：对每组最新 train run 调 nvit.global_evaluator（人类学 + 内部熵/距离/秩/KTI）。

多卡: 与 ch6_topN 相同，任务级并行 —— 每进程 CUDA_VISIBLE_DEVICES=单张物理卡，子进程内 --gpu 0。

例:
  CH5_INTERNAL_GPUS=0,1,2,3,4,5 python3 artifacts/ch5_ablation_internal_eval.py
  # 4 张卡: 4+2 两波
  CH5_INTERNAL_GPUS=0,1,2,3 python3 artifacts/ch5_ablation_internal_eval.py
  # 单卡串行
  python3 artifacts/ch5_ablation_internal_eval.py
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

CH5_GROUPS = [
    "M0_NoMask",
    "M1_Pos16",
    "M2_Pos24",
    "M3_8PlusSoft",
    "M4_AdaptiveKTI",
    "M5_8PlusHard",
]


def _parse_gpus(s: str | None) -> list[str] | None:
    if not s or not str(s).strip():
        return None
    out = [p.strip() for p in s.split(",") if p.strip()]
    return out or None


def latest_run_for_group(ch5_base: Path, g: str) -> Path:
    d = ch5_base / g / "train" / "runs"
    if not d.is_dir():
        raise SystemExit(f"ERROR: 目录不存在: {d}")
    cands = sorted(
        [p for p in d.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    for run in reversed(cands):
        if (run / "checkpoints").is_dir():
            return run
    raise SystemExit(f"ERROR: 在 {d} 下未找到含 checkpoints/ 的 run。")


def _load_manual_runs(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _group_name_for_path(run_p: str) -> str:
    for g in CH5_GROUPS:
        if f"/{g}/" in run_p.replace("\\", "/"):
            return g
    return "unknown"


def _build_cmd(
    args: argparse.Namespace, run_path: str, run_label: str, cuda_visible: str | None
) -> list[str]:
    gpu_id = "0" if cuda_visible is not None else str(args.gpu)
    return [
        args.python,
        "-m",
        "nvit.global_evaluator",
        "--chapter",
        "Ch5",
        "--run_path",
        run_path,
        "--run_label",
        run_label,
        "--gpu",
        gpu_id,
        "--diag_batches",
        str(args.diag_batches),
        "--datasets",
        args.datasets,
    ]


def _run_subprocess(cmd: list[str], repo: Path, cuda_visible: str | None) -> int:
    env = {
        **os.environ,
        "PYTHONPATH": f"{repo}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    p = subprocess.run(cmd, cwd=str(repo), env=env)
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ch5 六组 internal 评测，可选 --gpus 多卡任务并行"
    )
    ap.add_argument(
        "--ch5-base",
        type=Path,
        default=None,
        help="默认同 CH5_BASE 环境 或 <repo>/output/ch5_prior_compare",
    )
    ap.add_argument(
        "--manual-list",
        type=Path,
        default=None,
        help="每行一个 run 目录，覆盖六组自动发现。也可用环境 CH5_MANUAL_LIST_FILE。",
    )
    ap.add_argument(
        "--gpu",
        type=str,
        default=os.environ.get("GPU", "0"),
        help="单卡串行时的设备 id。多卡时子进程在隔离环境下用 --gpu 0。",
    )
    ap.add_argument(
        "--gpus",
        type=str,
        default=os.environ.get("CH5_INTERNAL_GPUS") or None,
        metavar="0,1,2,3,4,5",
        help="物理 GPU 列表。设置则分波多任务并行。环境: CH5_INTERNAL_GPUS。",
    )
    ap.add_argument(
        "--diag-batches",
        type=int,
        default=int(os.environ.get("DIAG_BATCHES", "50")),
    )
    ap.add_argument("--datasets", type=str, default="ALL")
    ap.add_argument(
        "--python",
        type=str,
        default=os.environ.get("PYTHON") or sys.executable,
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = _REPO
    ch5_base = args.ch5_base
    if ch5_base is None:
        e = os.environ.get("CH5_BASE")
        ch5_base = Path(e) if e else (repo / "output" / "ch5_prior_compare")
    if not ch5_base.is_dir():
        raise SystemExit(f"CH5 根目录不存在: {ch5_base}")

    manual = args.manual_list
    if manual is None and os.environ.get("CH5_MANUAL_LIST_FILE"):
        manual = Path(os.environ["CH5_MANUAL_LIST_FILE"])
    if manual is not None and not manual.is_file():
        raise SystemExit(f"无效 manual 文件: {manual}")

    work: list[tuple[str, str, str]] = []  # group_name, run_path, run_label
    if manual is not None:
        for line in _load_manual_runs(manual):
            rp = line.strip()
            g = _group_name_for_path(rp)
            tail = Path(rp).name
            safe = re.sub(r"[^0-9A-Za-z._-]+", "_", f"ch5_{g}__{tail}").strip("_")
            work.append((g, rp, safe))
    else:
        for g in CH5_GROUPS:
            p = latest_run_for_group(ch5_base, g)
            rp = str(p)
            label = f"ch5_{g}__{p.name}"
            work.append((g, rp, label))

    gpus = _parse_gpus(args.gpus)
    use_parallel = gpus is not None and len(gpus) >= 1
    nphys = len(gpus) if gpus else 0
    if use_parallel and nphys < 1:
        use_parallel = False

    print(f"CH5_BASE={ch5_base}")
    for g, rp, _lab in work:
        print(f"  {g} -> {rp}")
    if use_parallel and gpus:
        print(f"多卡任务并行: {','.join(gpus)}  (per-process CUDA_VISIBLE_DEVICES, 内为 cuda:0)")

    _lock = threading.Lock()
    lines: list[str] = []

    def one_serial(g: str, run_path: str, run_label: str) -> int:
        cmd = _build_cmd(args, run_path, run_label, None)
        cmd_s = " ".join(shquote(x) for x in cmd)
        print(f"========== {g} ==========\nrun_path={run_path}\n{cmd_s}\n")
        lines.append(cmd_s)
        if args.dry_run:
            return 0
        return _run_subprocess(cmd, repo, None)

    def one_parallel(phys: str, g: str, run_path: str, run_label: str) -> int:
        cmd = _build_cmd(args, run_path, run_label, phys)
        env_note = f"CUDA_VISIBLE_DEVICES={shquote(phys)} "
        cmd_s = env_note + " ".join(shquote(x) for x in cmd)
        with _lock:
            print(f"========== {g} (GPU {phys}) ==========\nrun_path={run_path}\n{cmd_s}\n")
            lines.append(cmd_s)
        if args.dry_run:
            return 0
        return _run_subprocess(cmd, repo, phys)

    if not use_parallel or not gpus:
        for g, rp, lab in work:
            rc = one_serial(g, rp, lab)
            if rc != 0:
                raise SystemExit(
                    f"global_evaluator 失败 returncode={rc} group={g!r} label={lab!r}"
                )
    else:
        w = 0
        for b in range(0, len(work), nphys):
            chunk = work[b : b + nphys]
            w += 1
            print(f"--- wave {w} ({len(chunk)} 路并行) ---", file=sys.stderr)
            with ThreadPoolExecutor(max_workers=len(chunk)) as ex:
                fmap = {
                    ex.submit(
                        one_parallel, gpus[i], g, rp, lab
                    ): (g, gpus[i], lab)
                    for i, (g, rp, lab) in enumerate(chunk)
                }
                for fut in as_completed(fmap):
                    g, gname, lab = fmap[fut]
                    rc = fut.result()
                    if rc != 0:
                        raise SystemExit(
                            f"global_evaluator 失败 returncode={rc} group={g!r} GPU {gname} label={lab!r}"
                        )

    if args.dry_run and lines:
        print("--- 以上为 dry-run，未执行 ---")

    print(
        f"\n完成。汇总: {repo / 'outputs' / 'eval_global' / 'Ch5' / 'summary.csv'}"
    )
    print("各组 Run 列: --run_label 形如 ch5_M0_NoMask__<date> 。")


if __name__ == "__main__":
    main()
