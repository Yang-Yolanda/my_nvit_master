#!/usr/bin/env python3
"""
在 metrics_master 中按 ch6_best_vs_baselines 相同规则（3DPW+H36M、rank-sum）
对「checkpoint 路径含给定子串」的 NViT ch6 记录取 composite best，向指定文件 **追加一行**（便于跑完 cluster 后留档）。

示例:
  python3 artifacts/append_ch6_experiment_composite_best_line.py \\
    --checkpoint-path-contains ch6_phase2_unfreeze0_from80k \\
    --append-to artifacts/eval_unified/ch6_experiment_composite_best.log
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "artifacts") not in sys.path:
    sys.path.insert(0, str(_REPO / "artifacts"))
import ch6_best_vs_baselines as m  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="将指定 ch6 实验的 composite best 追加一行到文件")
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="默认 <repo>/artifacts/eval_unified/metrics_master.csv",
    )
    ap.add_argument(
        "--checkpoint-path-contains",
        required=True,
        help="与 CH6_EXPERIMENT_DIR 的目录名一致即可，如 ch6_phase2_unfreeze0_from80k",
    )
    ap.add_argument(
        "--append-to",
        type=Path,
        required=True,
        help="在该文件末尾追加一行 TSV",
    )
    ap.add_argument(
        "--rank-metric-3d",
        choices=("mode_re", "mode_mpjpe"),
        default="mode_re",
    )
    args = ap.parse_args()
    csv = args.metrics_csv or (_REPO / "artifacts" / "eval_unified" / "metrics_master.csv")
    if not csv.is_file():
        print(f"[append_ch6_best] 找不到 metrics: {csv}", file=sys.stderr)
        return
    recs = m.load_ch6_records_from_master(csv, m.RANK_DATASETS)
    sub = m.filter_ch6_records(recs, checkpoint_contains=args.checkpoint_path_contains)
    if not sub:
        print(
            f"[append_ch6_best] 无 path 含 {args.checkpoint_path_contains!r} 且含"
            f" {m.RANK_DATASETS} 的 ch6 记录（先跑 unified / cluster）。",
            file=sys.stderr,
        )
        return
    best = m.compute_composite_best(sub, args.rank_metric_3d)
    if not best:
        print("[append_ch6_best] 无法计算 composite best。", file=sys.stderr)
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res = best.get("results") or {}
    d3 = res.get("3DPW-TEST") or {}
    h36 = res.get("H36M-VAL-P2") or {}
    line = (
        f"{ts}\tsubstr={args.checkpoint_path_contains}\t"
        f"rank_metric={args.rank_metric_3d}\t"
        f"step={best.get('step')}\trank_sum={best.get('rank_sum')}\t"
        f"3DPW_PA={d3.get('mode_re')}\t3DPW_MPJPE={d3.get('mode_mpjpe')}\t"
        f"H36M_PA={h36.get('mode_re')}\tH36M_MPJPE={h36.get('mode_mpjpe')}\t"
        f"checkpoint={best.get('checkpoint')}\t"
        f"json_path={best.get('json_path')}\n"
    )
    args.append_to.parent.mkdir(parents=True, exist_ok=True)
    with args.append_to.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())


if __name__ == "__main__":
    main()
