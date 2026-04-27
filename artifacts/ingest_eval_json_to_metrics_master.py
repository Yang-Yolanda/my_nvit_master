#!/usr/bin/env python3
"""
将 standard_eval 产出的 JSON（与 unified_eval_batch 同构）追加到 metrics_master.csv。

- 容忍 JSON 中的 NaN / Infinity（与 Python json.dump 默认一致），先替换为 null 再解析。
- 若 master 中已存在相同 json_path 的行，则跳过（幂等），便于重复跑 bash。
- 默认 chapter 为 ch6_external，避免与 NViT 的 chapter=ch6 混进 ch6_best_vs_baselines 的 composite 池。
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FIELDNAMES = [
    "timestamp_utc",
    "family",
    "chapter",
    "experiment",
    "checkpoint",
    "json_path",
    "dataset",
    "MPJPE_mm",
    "PA_MPJPE_mm",
    "KPL2",
    "limit_batches",
    "status",
]


def _lenient_json_loads(raw: str) -> dict[str, Any]:
    s = re.sub(r"(?m)\bNaN\b", "null", raw)
    s = re.sub(r"(?m)\bInfinity\b", "null", s)
    s = re.sub(r"(?m)\b-Infinity\b", "null", s)
    return json.loads(s)


def _cell_num(v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return ""
        return str(v)
    return str(v)


def _master_has_json_path(master: Path, json_abs: str) -> bool:
    if not master.is_file():
        return False
    with master.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "json_path" not in r.fieldnames:
            return False
        for row in r:
            if (row.get("json_path") or "").strip() == json_abs:
                return True
    return False


def _append_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a+", newline="", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (AttributeError, OSError):
            pass
        try:
            empty = os.fstat(f.fileno()).st_size == 0
            w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
            if empty:
                w.writeheader()
            for row in rows:
                w.writerow(row)
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (AttributeError, OSError):
                pass


def _rows_from_eval_json(
    data: dict[str, Any],
    *,
    family: str,
    chapter: str,
    experiment: str,
    json_path: Path,
    status: str,
) -> list[dict[str, Any]]:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    args = data.get("args") or {}
    ckpt_abs = str(args.get("ckpt", json_path.resolve()))
    lim = args.get("limit_batches", "")
    lim_s = "" if lim in (None, "") else str(lim)
    results = data.get("results") or {}
    rows: list[dict[str, Any]] = []
    for ds_name, m in results.items():
        if not isinstance(m, dict):
            continue
        rows.append(
            {
                "timestamp_utc": ts,
                "family": family,
                "chapter": chapter,
                "experiment": experiment,
                "checkpoint": ckpt_abs,
                "json_path": str(json_path.resolve()),
                "dataset": ds_name,
                "MPJPE_mm": _cell_num(m.get("mode_mpjpe")),
                "PA_MPJPE_mm": _cell_num(m.get("mode_re")),
                "KPL2": _cell_num(m.get("mode_kpl2")),
                "limit_batches": lim_s,
                "status": status,
            }
        )
    if not rows:
        rows.append(
            {
                "timestamp_utc": ts,
                "family": family,
                "chapter": chapter,
                "experiment": experiment,
                "checkpoint": ckpt_abs,
                "json_path": str(json_path.resolve()),
                "dataset": "",
                "MPJPE_mm": "",
                "PA_MPJPE_mm": "",
                "KPL2": "",
                "limit_batches": lim_s,
                "status": "no_results_in_json",
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="将 standard_eval JSON 追加到 metrics_master.csv（按 json_path 幂等）"
    )
    ap.add_argument("--json", type=Path, required=True, help="standard_eval 输出的 .json")
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="默认 <repo>/artifacts/eval_unified/metrics_master.csv",
    )
    ap.add_argument("--family", default="external_mnt", help="metrics_master family 列")
    ap.add_argument(
        "--chapter",
        default="ch6_external",
        help="chapter 列；外部 mnt 评测请用 ch6_external，勿用 ch6（会污染 NViT composite）",
    )
    ap.add_argument(
        "--experiment",
        default="",
        help="experiment 列；默认用 json 文件名不含扩展名",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="即使已存在相同 json_path 也再追加一遍（一般不推荐）",
    )
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    master = args.metrics_csv or (repo / "artifacts" / "eval_unified" / "metrics_master.csv")
    jp = args.json.resolve()
    if not jp.is_file():
        raise SystemExit(f"找不到 JSON: {jp}")
    json_abs = str(jp)
    if not args.force and _master_has_json_path(master, json_abs):
        print(f"[ingest] skip（已存在 json_path）: {json_abs}", file=sys.stderr)
        return
    exp = (args.experiment or "").strip() or jp.stem
    try:
        data = _lenient_json_loads(jp.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"解析 JSON 失败: {e}") from e
    rows = _rows_from_eval_json(
        data,
        family=args.family,
        chapter=args.chapter,
        experiment=exp,
        json_path=jp,
        status="ok",
    )
    _append_rows(master, rows)
    print(f"[ingest] appended {len(rows)} row(s) -> {master}")


if __name__ == "__main__":
    main()
