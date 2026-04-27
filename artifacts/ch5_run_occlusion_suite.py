#!/usr/bin/env python3
"""
Run random-box occlusion on 3DPW-TEST for all Ch5 ablation checkpoints and merge to CSV/JSON.

Example (full 3DPW-TEST, one GPU, serial groups):
  conda activate 4D-humans
  cd /path/to/NViT-master
  export PYTHONPATH=../4D-Humans:.
  python3 artifacts/ch5_run_occlusion_suite.py --out-dir outputs/eval_global/Ch5/occlusion_3dpw

Faster ablation (first N images per run):
  python3 artifacts/ch5_run_occlusion_suite.py --max-samples 4096
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_D4 = Path(
    os.environ.get(
        "CH5_OCC_PYTHON",
        "/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python3",
    )
)
PY = str(_D4) if _D4.is_file() else sys.executable

CH5: list[tuple[str, str]] = [
    ("M0_NoMask", "M0_NoMask/train/runs/2026-04-18_11-31-05/checkpoints/last.ckpt"),
    ("M1_Pos16", "M1_Pos16/train/runs/2026-04-18_11-36-34/checkpoints/last.ckpt"),
    ("M2_Pos24", "M2_Pos24/train/runs/2026-04-18_11-36-42/checkpoints/last.ckpt"),
    ("M3_8PlusSoft", "M3_8PlusSoft/train/runs/2026-04-18_11-36-53/checkpoints/last.ckpt"),
    ("M4_AdaptiveKTI", "M4_AdaptiveKTI/train/runs/2026-04-18_11-36-59/checkpoints/last.ckpt"),
    ("M5_8PlusHard", "M5_8PlusHard/train/runs/2026-04-18_11-37-06/checkpoints/last.ckpt"),
]


def _merge_per_group(out_dir: Path) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    merged: dict[str, object] = {}
    for g, _ in CH5:
        path = out_dir / f"occl_{g}.json"
        if not path.is_file():
            print(f"WARN: missing {path}, skip in merge", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        merged[g] = data
        for occ, m in data.items():
            if not isinstance(m, dict):
                continue
            rows.append(
                {
                    "group": g,
                    "occlusion_ratio": float(occ),
                    "MPJPE": m.get("MPJPE"),
                    "PA-MPJPE": m.get("PA-MPJPE"),
                }
            )
    rows.sort(key=lambda r: (r["group"], r["occlusion_ratio"]))
    merged_path = out_dir / "occlusion_all_groups.json"
    with merged_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    table_path = out_dir / "occlusion_table.csv"
    if rows:
        with table_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["group", "occlusion_ratio", "MPJPE", "PA-MPJPE"])
            w.writeheader()
            w.writerows(rows)
    return rows, f"wrote {merged_path} and {table_path} ({len(rows)} rows)\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ch5-compare-base",
        type=Path,
        default=REPO / "output" / "ch5_prior_compare",
        help="Directory with M*/train/.../last.ckpt (default: output/ch5_prior_compare).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "outputs" / "eval_global" / "Ch5" / "occlusion_3dpw",
        help="Output directory for per-group JSON + merged table.",
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only first N 3DPW-TEST images per group (faster, comparable across groups).",
    )
    ap.add_argument(
        "--python",
        type=str,
        default=None,
        help="Override Python (default: CH5_OCC_PYTHON or 4D-humans env).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only.",
    )
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    humans = REPO.parent / "4D-Humans"
    if not humans.is_dir():
        print(
            f"ERROR: expected 4D-Humans at {humans} (or set HUMANS_ROOT + edit script).",
            file=sys.stderr,
        )
        sys.exit(1)
    env = {
        **os.environ,
        "HUMANS_ROOT": str(humans),
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{humans}:{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }

    py_exec = str(Path(args.python)) if args.python else PY

    for g, rel in CH5:
        ckpt = args.ch5_compare_base / rel
        if not ckpt.is_file():
            print(f"ERROR: missing checkpoint {ckpt}", file=sys.stderr)
            sys.exit(1)
        oj = out / f"occl_{g}.json"
        cmd: list[str] = [
            py_exec,
            "-m",
            "nvit.eval_ch5_external_occlusion",
            "--ckpt",
            str(ckpt),
            "--group",
            g,
            "--output_json",
            str(oj),
            "--gpu",
            str(args.gpu),
        ]
        if args.max_samples is not None:
            cmd += ["--max_samples", str(args.max_samples)]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        print("RUN", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(REPO), env=env)
        if r.returncode != 0:
            print(f"FAILED: group {g} exit {r.returncode}", file=sys.stderr)
            sys.exit(r.returncode)

    if not args.dry_run:
        rows, msg = _merge_per_group(out)
        print(msg)
    else:
        print("dry-run: no merge", file=sys.stderr)


if __name__ == "__main__":
    # Allow `python3 artifacts/ch5_run_occlusion_suite.py` to resolve repo
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    main()
