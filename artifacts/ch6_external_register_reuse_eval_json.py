#!/usr/bin/env python3
"""
为已有 standard_eval JSON 生成 ch6 external 流水线所需的：
  - 追加 OK_JSON:: 行到 LIST_F（与 bash 评测成功时格式一致）
  - 写可被 `source` 的 bench 片段（追加 BENCH_ENTRY_ARGS）
  - 可选：幂等写入 metrics_master（与 ingest_eval_json_to_metrics_master 一致）
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_INGEST = _REPO / "artifacts" / "ingest_eval_json_to_metrics_master.py"


def _load_eval_json(p: Path) -> dict:
    raw = p.read_text(encoding="utf-8")
    raw = re.sub(r"(?m)\bNaN\b", "null", raw)
    raw = re.sub(r"(?m)\bInfinity\b", "null", raw)
    raw = re.sub(r"(?m)\b-Infinity\b", "null", raw)
    return json.loads(raw)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="注册已有 eval JSON，供 run_ch6_external_mnt_nvit_compare 跳过 standard_eval"
    )
    ap.add_argument("--list-file", type=Path, required=True)
    ap.add_argument("--bench-snippet", type=Path, required=True)
    ap.add_argument(
        "--also-ingest-metrics",
        type=Path,
        default=None,
        help="给定则对每个 JSON 调用 ingest_eval_json_to_metrics_master.py（已存在 json_path 则跳过）",
    )
    ap.add_argument("--delimiter", default=";", help="--paths-list 中使用的分隔符")
    ap.add_argument(
        "--paths-list",
        required=True,
        help=r"多个 JSON 路径，默认用分号分隔（如 a.json;b.json）",
    )
    args = ap.parse_args()
    paths = [
        x.strip()
        for x in args.paths_list.split(args.delimiter)
        if x and str(x).strip()
    ]
    if not paths:
        raise SystemExit("paths-list 解析后为空")

    rows: list[tuple[str, str, Path]] = []
    for s in paths:
        p = Path(s).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"找不到 JSON: {p}")
        data = _load_eval_json(p)
        ck = ""
        a = data.get("args")
        if isinstance(a, dict):
            ck = str(a.get("ckpt") or "").strip()
        if not ck:
            raise SystemExit(f"JSON 缺少 args.ckpt: {p}")
        nm = Path(ck).stem
        rows.append((nm, ck, p))

    with args.list_file.open("a", encoding="utf-8") as lf:
        for nm, ck, p in rows:
            lf.write(f"OK_JSON::{p}::{ck}::{nm}\n")

    args.bench_snippet.parent.mkdir(parents=True, exist_ok=True)
    with args.bench_snippet.open("w", encoding="utf-8") as bf:
        for nm, ck, _p in rows:
            bf.write(
                "BENCH_ENTRY_ARGS+=(--entry "
                + shlex.quote(f"{nm}={ck}")
                + ")\n"
            )

    if args.also_ingest_metrics:
        for nm, ck, p in rows:
            exp = f"{nm}_pth" if ck.lower().endswith(".pth") else nm
            subprocess.run(
                [
                    sys.executable,
                    str(_INGEST),
                    "--json",
                    str(p),
                    "--metrics-csv",
                    str(args.also_ingest_metrics),
                    "--family",
                    "external_mnt",
                    "--chapter",
                    "ch6_external",
                    "--experiment",
                    exp,
                ],
                check=False,
            )

    print(f"[reuse] 已注册 {len(rows)} 个 eval JSON -> {args.list_file}")


if __name__ == "__main__":
    main()
