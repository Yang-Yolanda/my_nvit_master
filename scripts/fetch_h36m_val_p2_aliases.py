#!/usr/bin/env python3
"""
Copy H36M val-P2 missing jpgs from OSS using filename alias (60457274 <-> 54138969 / 55011271),
saving under the **NPZ-expected** name so ImageDataset can open them.

  HUMANS_ROOT=/cpfs/.../4D-Humans python3 scripts/fetch_h36m_val_p2_aliases.py /tmp/h36m_val_p2_missing.txt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def oss_stat(oss_url: str) -> bool:
    r = subprocess.run(
        ["ossutil", "stat", oss_url],
        capture_output=True,
        timeout=120,
    )
    return r.returncode == 0


def oss_cp(src: str, dst: Path) -> bool:
    r = subprocess.run(
        ["ossutil", "cp", "-f", src, str(dst)],
        capture_output=True,
        timeout=300,
    )
    return r.returncode == 0


def _candidates(needed: str) -> list[str]:
    """
    npz 期待名里往往是 60457272，但 OSS 上同帧可能落在 54138969/55011271/58860488 等表；
    或动作名有 _1 / 无 _1 的差异。生成若干候选，按顺序去 OSS 上 stat。
    """
    hlist = ("54138969", "55011271", "58860488", "60457274")
    out: list[str] = []
    seen: set[str] = set()
    # 结构变体 + 多哈希
    bases = [needed]
    b = needed
    if "WalkDog.60457274" in b:
        bases.append(b.replace("WalkDog.60457274", "WalkDog_1.60457274", 1))
    if "WalkDog_1.60457274" in b:
        bases.append(b.replace("WalkDog_1.60457274", "WalkDog.60457274", 1))
    for act in (
        "Phoning_1",
        "Posing_1",
        "Eating_1",
        "Greeting_1",
        "Sitting_1",
        "Smoking_1",
    ):
        dot = act.replace("_1", "") + "."
        if f"{act}.60457274" in b:
            bases.append(b.replace(f"{act}.60457274", f"{dot}60457274", 1))
    for base in list(bases):
        for h in hlist:
            if "60457274" not in base and h == "60457274":
                continue
            t = base.replace("60457274", h, 1) if "60457274" in base else base
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def resolve_src(needed: str, oss_prefix: str) -> str | None:
    prefix = oss_prefix.rstrip("/")
    for cand in _candidates(needed):
        u = f"{prefix}/{cand}"
        if oss_stat(u):
            return cand
    return None


def one(
    needed: str,
    dest_dir: Path,
    oss_prefix: str,
) -> tuple[str, str]:
    """Return (status, detail). status in ok|skip|fail."""
    dst = dest_dir / needed
    if dst.is_file():
        return "skip", needed
    src = resolve_src(needed, oss_prefix)
    if src is None:
        return "fail", f"no oss key for {needed}"
    src_url = f"{oss_prefix.rstrip('/')}/{src}"
    if oss_cp(src_url, dst):
        return "ok", f"{needed} <- {src}"
    return "fail", f"cp {src_url}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("missing_list", type=Path, help="One filename per line (npz imgname)")
    ap.add_argument(
        "--humans-root",
        type=Path,
        default=None,
        help="default: env HUMANS_ROOT or nvit get_humans_root()",
    )
    ap.add_argument(
        "--oss-prefix",
        default=os.environ.get("OSS_EVAL_PREFIX", "oss://kai-ego/eval_shujuji") + "/h36m",
        help="OSS directory prefix for flat jpgs",
    )
    ap.add_argument("-j", "--jobs", type=int, default=8, help="parallel workers (default 8)")
    args = ap.parse_args()

    hr = args.humans_root
    if hr is None:
        v = os.environ.get("HUMANS_ROOT")
        if v:
            hr = Path(v)
        else:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from nvit.utils.path_utils import get_humans_root

            hr = get_humans_root()

    dest_dir = hr / "data" / "h36m"
    dest_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        ln.strip()
        for ln in args.missing_list.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    oss_p = args.oss_prefix.rstrip("/") + "/"

    ok = skip = fail = 0
    if args.jobs <= 1:
        for needed in lines:
            st, _ = one(needed, dest_dir, oss_p)
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(one, ln, dest_dir, oss_p): ln for ln in lines}
            for fu in as_completed(futs):
                st, msg = fu.result()
                if st == "ok":
                    ok += 1
                elif st == "skip":
                    skip += 1
                else:
                    fail += 1
                    print(msg, file=sys.stderr)

    print(f"fetch_h36m_val_p2_aliases: ok={ok} skip={skip} fail={fail} dest={dest_dir}/")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
