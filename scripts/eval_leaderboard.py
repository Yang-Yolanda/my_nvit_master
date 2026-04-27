#!/usr/bin/env python3
"""Print sorted rows from artifacts/eval_unified/metrics_master.csv (for picking best ckpts / plots)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "artifacts" / "eval_unified" / "metrics_master.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--dataset", type=str, default="3DPW-TEST")
    p.add_argument("--chapter", type=str, default="", help="Filter: ch5 or ch6 or empty for all")
    p.add_argument("--family", type=str, default="NViT", help="NViT or SMPLer or empty for all")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    if not args.csv.is_file():
        print(f"Missing {args.csv}")
        return

    rows: list[dict[str, str]] = []
    with args.csv.open() as f:
        for r in csv.DictReader(f):
            if r.get("dataset") != args.dataset:
                continue
            if args.chapter and r.get("chapter") != args.chapter:
                continue
            if args.family and r.get("family") != args.family:
                continue
            try:
                r["_mpjpe"] = float(r.get("MPJPE_mm") or "nan")
            except ValueError:
                r["_mpjpe"] = float("nan")
            rows.append(r)

    rows.sort(key=lambda x: (x["_mpjpe"] if x["_mpjpe"] == x["_mpjpe"] else 1e9))

    print(f"dataset={args.dataset} chapter={args.chapter or '*'} family={args.family or '*'} — lower MPJPE is better\n")
    for i, r in enumerate(rows[: args.top], 1):
        mp = r.get("MPJPE_mm", "")
        pa = r.get("PA_MPJPE_mm", "")
        print(
            f"{i:2d}. MPJPE={mp:>8}  PA-MPJPE={pa:>8}  "
            f"[{r.get('family')}/{r.get('chapter')}] {r.get('experiment', '')}"
        )
        print(f"     {r.get('checkpoint', '')[:120]}")


if __name__ == "__main__":
    main()
