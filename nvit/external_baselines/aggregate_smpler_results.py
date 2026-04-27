#!/usr/bin/env python3
"""Merge SMPLer eval JSON outputs into artifacts/external_baselines/SMPLer/results.csv."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    args = p.parse_args()
    root = Path(args.project_root)
    art = root / "artifacts" / "external_baselines" / "SMPLer"
    rows = []

    for name, ds in [("3dpw", "3DPW-TEST"), ("h36m", "H36M-VAL-P2")]:
        jp = art / f"smpler_{name}.json"
        mpjpe, pare = "", ""
        status = "missing_json"
        if jp.is_file():
            try:
                data = json.loads(jp.read_text())
                r = data.get("results", {}).get(ds, {})
                mpjpe = r.get("mode_mpjpe", "")
                pare = r.get("mode_re", "")

                def _valid(m):
                    if m is None or m == "":
                        return False
                    if isinstance(m, float) and (math.isnan(m) or math.isinf(m)):
                        return False
                    return True

                status = "ok" if _valid(mpjpe) and _valid(pare) else "no_metrics"
            except Exception as e:
                status = f"parse_error:{e}"
        smpler_root = os.environ.get("SMPLER_ROOT", "/home/yangz/external_baselines/SMPLer")
        ck_hint = (
            f"{smpler_root}/pretrained/SMPLer_3dpw.pt"
            if name == "3dpw"
            else f"{smpler_root}/pretrained/SMPLer_h36m.pt"
        )
        rows.append(
            {
                "method": "SMPLer (SMPLerCH5Wrapper)",
                "dataset": ds,
                "MPJPE_mm": mpjpe,
                "PA_MPJPE_mm": pare,
                "checkpoint_expected": ck_hint,
                "status": status,
            }
        )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method", "dataset", "MPJPE_mm", "PA_MPJPE_mm", "checkpoint_expected", "status"],
        )
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
