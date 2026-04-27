#!/usr/bin/env python3
"""
Audit H36M-VAL-P2 images on disk against h36m_val_p2.npz (same path rules as
hmr2 ImageDataset / unified_eval_batch._image_path_candidates).

Usage:
  export HUMANS_ROOT=/cpfs_infra/shared/yangz/4D-Humans
  python3 scripts/audit_h36m_eval_images.py
  python3 scripts/audit_h36m_eval_images.py --out-missing /tmp/h36m_missing.txt

After audit, re-sync the flat jpgs, e.g.:
  bash scripts/sync_eval_data_from_oss.sh
  # or use scripts/sync_h36m_missing_from_oss.sh with the generated list
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _candidates(img_root: Path, rel: str) -> list[Path]:
    rel = rel.strip()
    out = [img_root / rel]
    if "/" not in rel:
        out.append(img_root / "images" / rel)
    parts = rel.split("/")
    if len(parts) > 1:
        out.append(img_root / parts[-1])
    out.append(img_root / "images" / rel)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--humans-root",
        type=Path,
        default=None,
        help="4D-Humans root (default: env HUMANS_ROOT or path_utils.get_humans_root())",
    )
    ap.add_argument(
        "--out-missing",
        type=Path,
        default=None,
        help="Write one missing relative filename per line (npz imgname, flat basename).",
    )
    ap.add_argument(
        "--max-report",
        type=int,
        default=20,
        help="Max example lines printed for missing (default 20).",
    )
    args = ap.parse_args()

    if args.humans_root is not None:
        hr = args.humans_root
    else:
        hr = Path(os.environ.get("HUMANS_ROOT", ""))
        if not hr:
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from nvit.utils.path_utils import get_humans_root

                hr = get_humans_root()
            except Exception:
                print("Set HUMANS_ROOT or pass --humans-root", file=sys.stderr)
                return 1

    npz = hr / "hmr2_evaluation_data" / "h36m_val_p2.npz"
    if not npz.is_file():
        # common layout: data/h36m_val_p2.npz
        alt = hr / "data" / "h36m_val_p2.npz"
        npz = alt if alt.is_file() else npz
    if not npz.is_file():
        print(f"ERROR: missing {npz}", file=sys.stderr)
        return 1

    import numpy as np

    z = np.load(npz, allow_pickle=True)
    names = z["imgname"]
    img_root = hr / "data" / "h36m"
    n = len(names)
    ok = 0
    missing: list[str] = []
    for i in range(n):
        rel = names[i]
        s = rel.decode() if isinstance(rel, bytes) else str(rel)
        s = s.strip()
        found = any(p.is_file() for p in _candidates(img_root, s))
        if found:
            ok += 1
        else:
            missing.append(s)

    cov = 100.0 * ok / n if n else 0.0
    print(f"HUMANS_ROOT={hr.resolve()}")
    print(f"NPZ={npz} entries={n}")
    print(f"IMG_ROOT={img_root}/")
    print(f"Found={ok}  Missing={len(missing)}  Coverage={cov:.2f}%")
    if missing and args.max_report > 0:
        print("Example missing (showing up to", args.max_report, "):")
        for line in missing[: args.max_report]:
            print("  ", line)

    if args.out_missing:
        args.out_missing.parent.mkdir(parents=True, exist_ok=True)
        args.out_missing.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
        print(f"Wrote {len(missing)} paths -> {args.out_missing}")
    if missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
