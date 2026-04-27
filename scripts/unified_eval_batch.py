#!/usr/bin/env python3
"""
Batch-eval NViT checkpoints under output/ch5_prior_compare and output/ch6,
optionally ingest SMPLer baselines, and write everything under artifacts/eval_unified/.

Outputs (single place for plotting):
  artifacts/eval_unified/metrics_master.csv   — one row per (checkpoint × dataset)
  artifacts/eval_unified/json/nvit/*.json   — per-run JSON from standard_eval (human-readable names:
      ch5: ch5_<Group>_step_<step>.json from exp path; ch6: includes run id + step; not path hashes)
  artifacts/eval_unified/json/smpler/*.json — optional SMPLer runs

Default checkpoint selection is --checkpoint-mode max-step-per-run: in each .../checkpoints/
folder, only the highest step_step=*.ckpt is used (last.ckpt is skipped — it is large and
includes optimizer state; better for shared-GPU / low VRAM).

Ch5 ablation (M0–M5, latest train/runs/<date>/checkpoints/, every step_step=*.ckpt):
  --ch5-ablation-all-steps
  Writes  artifacts/eval_unified/ablation/ch5/<Group>.json  (aggregate + composite best)
  and       artifacts/eval_unified/ablation/ch5/summary_best_composite.csv
  Parallel GPUs: use scripts/run_best_max_step_eval.sh cluster (six workers) or pass
  --ch5-summary-append per shard after deleting summary CSV once; metrics_master.csv appends use flock.
  “Composite best” = lowest sum of per-dataset ranks (Borda / rank-sum). For 3DPW/H36M the
  rank key defaults to PA-MPJPE (mode_re); 2D sets use mode_kpl2.

Ch6 (latest run, all step checkpoints — for horizontal compare vs SMPLer on ch6):
  --ch6-all-steps
  Writes  artifacts/eval_unified/ablation/ch6/ch6_latest_run.json

SMPLer rows are tagged with --smpler-chapter (default ch6) for ch6-only comparison.

Example (fast screen, 3DPW only, GPU 6):
  export PYTHON=/cpfs_infra/shared/yangz/opt/Miniconda3/envs/4D-humans/bin/python
  $PYTHON scripts/unified_eval_batch.py \\
    --gpu 0 --cuda-visible-devices 6 \\
    --datasets 3DPW-TEST --limit-batches 80 \\
    --chapters ch5,ch6

Full HMR2-style eval bundle (5 benchmarks; standard_eval --dataset ALL):
  $PYTHON scripts/unified_eval_batch.py --gpu 0 --cuda-visible-devices 6 \\
    --datasets ALL --use-mean-alignment --chapters ch5,ch6

Ch5 ablation + full metrics + composite CSV:
  $PYTHON scripts/unified_eval_batch.py --ch5-ablation-all-steps \\
    --datasets ALL --use-mean-alignment --cuda-visible-devices 6 --gpu 0

SMPLer ingest (baseline rows, chapter ch6 by default):
  $PYTHON scripts/unified_eval_batch.py --ingest-smpler-json \\
    --smpler-3dpw-json artifacts/external_baselines/SMPLer/smpler_3dpw.json

Eval image roots (NPZ already in hmr2_evaluation_data/):
  $PYTHON scripts/unified_eval_batch.py --report-eval-data
  $PYTHON scripts/unified_eval_batch.py --prepare-eval-layout [--fetch-hr-lspet]
  # or: bash scripts/run_best_max_step_eval.sh prepare-data   # optional FETCH_HR_LSPET=1
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Allow `import nvit` when running `python scripts/unified_eval_batch.py` without PYTHONPATH.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "eval_unified"
STANDARD_EVAL = PROJECT_ROOT / "nvit" / "skills" / "evaluate_model" / "standard_eval.py"

# 4D-Humans README: evaluation metadata (NPZ lists); images still need separate downloads.
EVAL_METADATA_TAR_URL = (
    "https://www.dropbox.com/scl/fi/kl79djemdgqcl6d691er7/"
    "hmr2_evaluation_data.tar.gz?rlkey=ttmbdu3x5etxwqqyzwk581zjl"
)

DEFAULT_CH5_GROUPS = [
    "M0_NoMask",
    "M1_Pos16",
    "M2_Pos24",
    "M3_8PlusSoft",
    "M4_AdaptiveKTI",
    "M5_8PlusHard",
]

# Mirrors standard_eval.py when --dataset ALL (5 benchmarks; MPI-INF not in datasets_eval.yaml).
# 用于 --report-eval-data / 完整性检查。MPI-INF 无 npz 时会报缺失（属预期，直至自备数据）。
ALL_EVAL_DATASETS = [
    "3DPW-TEST",
    "3DPW-OCC-TEST",
    "MPI-INF-3DHP-TEST",
    "H36M-VAL-P2",
    "COCO-VAL",
    "POSETRACK-VAL",
    "LSP-EXTENDED",
]

DATASET_NPZ = {
    "3DPW-TEST": "3dpw_test.npz",
    "3DPW-OCC-TEST": "3dpw_occ_test.npz",
    "MPI-INF-3DHP-TEST": "mpi_inf_3dhp_test.npz",
    "H36M-VAL-P2": "h36m_val_p2.npz",
    "COCO-VAL": "coco_val.npz",
    "POSETRACK-VAL": "posetrack_2018_val.npz",
    "LSP-EXTENDED": "hr-lspet_train.npz",
}

# Default IMG_DIR from 4D-Humans/hmr2/configs/datasets_eval.yaml (rebased via resolve_eval_img_dir).
YAML_EVAL_IMG = {
    "3DPW-TEST": "/home/yangz/4D-Humans/data/3DPW/",
    "3DPW-OCC-TEST": "/home/yangz/4D-Humans/data/3DPW/",
    "MPI-INF-3DHP-TEST": "/home/yangz/4D-Humans/data/mpi_inf_3dhp/",
    "H36M-VAL-P2": "/home/yangz/4D-Humans/data/h36m/",
    "COCO-VAL": "/home/yangz/4D-Humans/data/coco/",
    "POSETRACK-VAL": "/home/yangz/4D-Humans/data/posetrack/",
    "LSP-EXTENDED": "/home/yangz/4D-Humans/data/hr-lspet/image",
}

HR_LSPET_ZIP_URL = "https://datasets.d2.mpi-inf.mpg.de/hr-lspet/hr-lspet.zip"
# Content-Length from MPI server (used to verify complete download before unzip).
HR_LSPET_ZIP_BYTES = 2864955669


def _step_from_ckpt_name(name: str) -> int | None:
    """Parse training step from Lightning-style filenames (e.g. step_step=45000.ckpt, epoch=1-step=9000.ckpt)."""
    if name in ("last.ckpt", "best.ckpt"):
        return None
    matches = re.findall(r"step[=](\d+)", name, flags=re.IGNORECASE)
    if not matches:
        return None
    return max(int(x) for x in matches)


def latest_run_dir(exp_root: Path) -> Path | None:
    runs = exp_root / "train" / "runs"
    if not runs.is_dir():
        return None
    dirs = sorted([d for d in runs.iterdir() if d.is_dir()])
    return dirs[-1] if dirs else None


def latest_run_dir_with_step_ckpts(exp_root: Path) -> Path | None:
    """
    Prefer the newest train/runs/<date> directory that actually contains step_step=*.ckpt.
    Avoids picking an empty newer run folder (lexicographically last) and skipping eval.
    """
    runs_root = exp_root / "train" / "runs"
    if not runs_root.is_dir():
        return None
    dirs = sorted([d for d in runs_root.iterdir() if d.is_dir()], reverse=True)
    for d in dirs:
        ckdir = d / "checkpoints"
        if ckdir.is_dir() and all_step_checkpoints(ckdir):
            return d
    return None


def ch6_experiment_child_candidates(parent: Path) -> list[Path]:
    """
    当用户把 CH6_EXPERIMENT_DIR 设成多实验的父目录（如 .../nvit_output）时，
    本目录下没有 train/runs，但子目录 ch6_xxx/ 可能有。返回含有效 step ckpt 的子目录列表。
    """
    if not parent.is_dir():
        return []
    out: list[Path] = []
    try:
        for c in sorted(parent.iterdir()):
            if not c.is_dir():
                continue
            if latest_run_dir_with_step_ckpts(c) is not None:
                out.append(c)
    except OSError:
        pass
    return out


def all_step_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Every step_step=*.ckpt in a checkpoints folder, highest step first."""
    scored: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("*.ckpt"):
        s = _step_from_ckpt_name(p.name)
        if s is not None:
            scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored]


def parse_dataset_list(datasets_arg: str) -> list[str]:
    s = datasets_arg.strip()
    if s.upper() == "ALL":
        return list(ALL_EVAL_DATASETS)
    return [x.strip() for x in datasets_arg.split(",") if x.strip()]


def rank_value_for_dataset(
    metrics: dict[str, Any],
    ds_name: str,
    rank_3d: str,
) -> tuple[float | None, str]:
    """Return (value, key_used) for ranking; lower is better. None if missing."""
    if not metrics:
        return None, ""
    if ds_name in ("3DPW-TEST", "3DPW-OCC-TEST", "MPI-INF-3DHP-TEST", "H36M-VAL-P2"):
        key = rank_3d if rank_3d in metrics else "mode_re"
        if key not in metrics and "mode_mpjpe" in metrics:
            key = "mode_mpjpe"
        v = metrics.get(key)
    elif "mode_kpl2" in metrics:
        key = "mode_kpl2"
        v = metrics.get(key)
    else:
        for cand in ("mode_re", "mode_mpjpe", "mode_kpl2"):
            if cand in metrics:
                v = metrics[cand]
                key = cand
                break
        else:
            return None, ""
    if not isinstance(v, (int, float)):
        return None, ""
    fv = float(v)
    if math.isnan(fv):
        return None, ""
    return fv, key


def compute_composite_best(
    records: list[dict[str, Any]],
    rank_3d: str,
) -> dict[str, Any] | None:
    """
    Rank-sum across datasets: for each dataset, rank checkpoints by that dataset's metric
    (lower better). Best checkpoint = minimum total rank (ties broken by lower mean rank).
    """
    if not records:
        return None
    n = len(records)
    # dataset -> list of (record_index, value)
    per_ds: dict[str, list[tuple[int, float]]] = {}
    for i, rec in enumerate(records):
        results = rec.get("results") or {}
        for ds_name, m in results.items():
            val, _ = rank_value_for_dataset(m, ds_name, rank_3d)
            if val is None:
                continue
            per_ds.setdefault(ds_name, []).append((i, val))
    if not per_ds:
        return None
    # record_index -> {ds: rank}
    ranks: dict[int, dict[str, int]] = {i: {} for i in range(n)}
    for ds_name, items in per_ds.items():
        items_sorted = sorted(items, key=lambda x: x[1])
        for rank, (idx, _) in enumerate(items_sorted, start=1):
            ranks[idx][ds_name] = rank
    best_i: int | None = None
    best_sum = float("inf")
    best_mean = float("inf")
    for i in range(n):
        rdict = ranks[i]
        if not rdict:
            continue
        s = sum(rdict.values())
        mean_r = s / len(rdict)
        if s < best_sum or (s == best_sum and mean_r < best_mean):
            best_sum = s
            best_mean = mean_r
            best_i = i
    if best_i is None:
        return None
    out_ranks = ranks[best_i]
    return {
        "record_index": best_i,
        "rank_sum": best_sum,
        "mean_rank": best_mean,
        "datasets_ranked": sorted(out_ranks.keys()),
        "per_dataset_rank": out_ranks,
        "checkpoint": records[best_i].get("checkpoint"),
        "step": records[best_i].get("step"),
        "results": records[best_i].get("results"),
    }


def verify_eval_npz(data_dir: Path, datasets_arg: str) -> list[str]:
    """Return list of missing NPZ basenames for requested datasets."""
    missing: list[str] = []
    for ds in parse_dataset_list(datasets_arg):
        bn = DATASET_NPZ.get(ds)
        if not bn:
            continue
        if not (data_dir / bn).is_file():
            missing.append(bn)
    return missing


def resolved_eval_img_dir(ds_name: str, humans_root: Path) -> Path:
    """Same rebasing as ImageDataset / resolve_eval_img_dir."""
    from nvit.utils.path_utils import resolve_eval_img_dir

    y = YAML_EVAL_IMG[ds_name]
    s = resolve_eval_img_dir(ds_name, y)
    return Path(s.rstrip("/"))


def _image_path_candidates(img_root: Path, rel: str) -> list[Path]:
    rel = rel.strip()
    out = [img_root / rel]
    if "/" not in rel:
        out.append(img_root / "images" / rel)
    parts = rel.split("/")
    if len(parts) > 1:
        out.append(img_root / parts[-1])
    out.append(img_root / "images" / rel)
    return out


def report_eval_image_status(humans_root: Path, data_npz_dir: Path, sample: int = 48) -> str:
    """Human-readable coverage report for the 5 HMR2 eval image roots vs NPZ entries."""
    lines: list[str] = []
    lines.append(f"HUMANS_ROOT={humans_root}")
    lines.append(f"NPZ directory={data_npz_dir}")
    lines.append("")
    for ds in ALL_EVAL_DATASETS:
        lines.append(f"=== {ds} ===")
        npz_path = data_npz_dir / DATASET_NPZ[ds]
        if not npz_path.is_file():
            lines.append(f"  NPZ missing: {npz_path.name}")
            lines.append("")
            continue
        z = np.load(npz_path, allow_pickle=True)
        names = z["imgname"]
        ntot = len(names)
        root = resolved_eval_img_dir(ds, humans_root)
        lines.append(f"  NPZ entries={ntot}; IMG root={root}/")
        ok = 0
        check = min(sample, ntot)
        first_bad: str | None = None
        for idx in range(check):
            raw = names[idx]
            rel = raw.decode() if isinstance(raw, bytes) else str(raw)
            found = False
            for cand in _image_path_candidates(root, rel):
                if cand.is_file():
                    found = True
                    break
            if found:
                ok += 1
            elif first_bad is None:
                first_bad = rel
        lines.append(f"  Sample check: {ok}/{check} first entries resolve to an existing file.")
        if first_bad is not None and ok < check:
            lines.append(f"  First missing (example): {first_bad}")
        lines.append("")
    lines.append("Manual steps for missing sets:")
    lines.append("  - H36M: register at vision.imar.ro; prepare S9/S11 per SPIN README; place crops as flat")
    lines.append("    filenames under .../data/h36m/ or .../data/h36m/images/ (matches h36m_val_p2.npz).")
    lines.append("  - PoseTrack 2018: download from posetrack.net / torrent; layout images/val/... under")
    lines.append("    .../data/posetrack/ (see first NPZ imgname).")
    lines.append(f"  - HR-LSPET: unzip {HR_LSPET_ZIP_URL} so that im00001.png lives under .../data/hr-lspet/image/")
    lines.append("  - Training tars under NViT/hmr2_training_data/dataset_tars/h36m/ are NOT the val-P2 layout")
    lines.append("    (subjects S9/S11); do not symlink them as a substitute without preprocessing.")
    return "\n".join(lines) + "\n"


def _download_url_no_proxy(url: str, dest: Path, chunk: int = 8 * 1024 * 1024) -> None:
    """Stream download without using HTTP(S)_PROXY (broken localhost proxies are common on dev machines)."""
    req = urllib.request.Request(url, headers={"User-Agent": "NViT-unified-eval-batch/1"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    next_log = 100 * 1024 * 1024
    # Long timeout: HR-LSPET zip is multi-GB on slow links.
    with opener.open(req, timeout=3600) as resp, dest.open("wb") as out:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            out.write(block)
            written += len(block)
            if written >= next_log:
                print(f"  ... downloaded {written / (1024**3):.2f} GiB", flush=True)
                next_log += 100 * 1024 * 1024


def finish_hr_lspet_from_zip(humans_root: Path, zip_path: Path | None = None) -> int:
    """
    If HR-LSPET zip is present and size matches MPI Content-Length, extract under humans_root/data.
    Returns 0 on success, 1 on error.
    """
    zpath = zip_path or (humans_root / "_hr-lspet.zip")
    if not zpath.is_file():
        print(f"ERROR: missing {zpath}", file=sys.stderr)
        return 1
    got = zpath.stat().st_size
    if got != HR_LSPET_ZIP_BYTES:
        print(
            f"ERROR: incomplete zip ({got} bytes, expected {HR_LSPET_ZIP_BYTES}). "
            f"Resume with: env -u http_proxy -u https_proxy -u ALL_PROXY wget -c -O {zpath} {HR_LSPET_ZIP_URL}",
            file=sys.stderr,
        )
        return 1
    (humans_root / "data").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(humans_root / "data")
    print(f"Extracted {zpath.name} -> {humans_root / 'data'}")
    return 0


def prepare_eval_layout(
    humans_root: Path,
    data_npz_dir: Path,
    out_report: Path,
    fetch_hr_lspet: bool,
) -> None:
    """Create expected directories and write eval_data_status.txt; optionally fetch HR-LSPET."""
    for sub in ("data/h36m/images", "data/h36m", "data/posetrack", "data/hr-lspet/image"):
        (humans_root / sub).mkdir(parents=True, exist_ok=True)
    text = report_eval_image_status(humans_root, data_npz_dir)
    if fetch_hr_lspet:
        hr_dir = humans_root / "data" / "hr-lspet" / "image"
        has_png = hr_dir.is_dir() and any(hr_dir.glob("im*.png"))
        if has_png:
            text += f"\nHR-LSPET: already present under {hr_dir}\n"
        else:
            zpath = humans_root / "_hr-lspet.zip"
            text += f"\nHR-LSPET: target zip {zpath} (expected {HR_LSPET_ZIP_BYTES} bytes)\n"
            try:
                if zpath.is_file():
                    sz = zpath.stat().st_size
                    if sz == HR_LSPET_ZIP_BYTES:
                        text += "HR-LSPET: zip complete; extracting.\n"
                    elif sz > 0:
                        text += (
                            f"HR-LSPET: partial zip ({sz} bytes). Resume download (no broken proxy), then extract:\n"
                            f"  env -u http_proxy -u https_proxy -u ALL_PROXY wget -c -O {zpath} {HR_LSPET_ZIP_URL}\n"
                            f"  {sys.executable} {PROJECT_ROOT}/scripts/unified_eval_batch.py --finish-hr-lspet\n"
                        )
                        raise RuntimeError("partial_zip_resume_with_wget")
                    else:
                        zpath.unlink(missing_ok=True)
                if not zpath.is_file() or zpath.stat().st_size != HR_LSPET_ZIP_BYTES:
                    print(f"Downloading HR-LSPET (no proxy) to {zpath} ...", flush=True)
                    _download_url_no_proxy(HR_LSPET_ZIP_URL, zpath)
                rc = finish_hr_lspet_from_zip(humans_root, zpath)
                if rc == 0:
                    text += "HR-LSPET: unzip OK (verify im*.png under data/hr-lspet/image).\n"
                else:
                    text += "HR-LSPET: extraction failed; fix zip size then --finish-hr-lspet.\n"
            except RuntimeError as e:
                if str(e) != "partial_zip_resume_with_wget":
                    raise
            except Exception as e:
                text += f"HR-LSPET: automated download failed ({e}). Manual:\n"
                text += (
                    f"  env -u http_proxy -u https_proxy -u ALL_PROXY "
                    f"wget -c -O {zpath} {HR_LSPET_ZIP_URL}\n"
                    f"  {sys.executable} {PROJECT_ROOT}/scripts/unified_eval_batch.py --finish-hr-lspet\n"
                )
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out_report}")


def download_eval_metadata_tar(humans_root: Path) -> bool:
    """Download and extract 4D-Humans evaluation NPZ bundle into humans_root/hmr2_evaluation_data."""
    dest_dir = humans_root / "hmr2_evaluation_data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    marker = dest_dir / ".eval_metadata_ok"
    if any(dest_dir.glob("*.npz")):
        marker.write_text("ok\n")
        return True
    tar_path = humans_root / "hmr2_evaluation_data.tar.gz"
    print(f"Downloading evaluation metadata to {tar_path} ...", flush=True)
    try:
        _download_url_no_proxy(EVAL_METADATA_TAR_URL, tar_path)
    except Exception as e:
        print(f"ERROR: download failed: {e}", file=sys.stderr)
        return False
    rc = subprocess.run(
        ["tar", "-xzf", str(tar_path), "-C", str(humans_root)],
        cwd=str(humans_root),
    ).returncode
    if rc != 0:
        print("ERROR: tar extract failed.", file=sys.stderr)
        return False
    marker.write_text("ok\n")
    return True


def discover_ckpts(project: Path, chapters: set[str]) -> list[tuple[str, Path]]:
    """Return list of (chapter_label, ckpt_path) — every *.ckpt under checkpoints/."""
    out: list[tuple[str, Path]] = []
    if "ch5" in chapters:
        root = project / "output" / "ch5_prior_compare"
        if root.is_dir():
            for p in sorted(root.rglob("*.ckpt")):
                if "checkpoints" in p.parts:
                    out.append(("ch5", p))
    if "ch6" in chapters:
        root = project / "output" / "ch6"
        if root.is_dir():
            for p in sorted(root.rglob("*.ckpt")):
                if "checkpoints" in p.parts:
                    out.append(("ch6", p))
    return out


def discover_ckpts_max_step_per_run(project: Path, chapters: set[str]) -> list[tuple[str, Path]]:
    """
    One checkpoint per .../checkpoints/ directory: the file with the largest step number in its name.
    Skips last.ckpt / best.ckpt (usually full training state + optimizer; large VRAM on load).
    """
    out: list[tuple[str, Path]] = []
    roots: list[Path] = []
    if "ch5" in chapters:
        roots.append(project / "output" / "ch5_prior_compare")
    if "ch6" in chapters:
        roots.append(project / "output" / "ch6")
    for base in roots:
        if not base.is_dir():
            continue
        for ckdir in sorted(base.glob("**/checkpoints")):
            if not ckdir.is_dir():
                continue
            candidates: list[tuple[int, Path]] = []
            for p in ckdir.glob("*.ckpt"):
                sc = _step_from_ckpt_name(p.name)
                if sc is not None:
                    candidates.append((sc, p))
            if not candidates:
                print(
                    f"WARNING: no step-based *.ckpt in {ckdir} (skipping; avoid last.ckpt for eval).",
                    file=sys.stderr,
                )
                continue
            best = max(candidates, key=lambda t: t[0])
            chapter = "ch5" if "ch5_prior_compare" in str(ckdir) else "ch6"
            out.append((chapter, best[1]))
    return out


def _relpath_parts_under_project_output(ckpt: Path) -> list[str]:
    """
    Path components under <PROJECT_ROOT>/output. Both sides are resolved so symlinks
    (e.g. /cpfs/.../output -> /mnt/...) do not break Path.relative_to.
    """
    out = (PROJECT_ROOT / "output").resolve()
    ckr = Path(ckpt).resolve()
    try:
        return list(ckr.relative_to(out).parts)
    except ValueError:
        cks = str(ckr)
        for split_at in ("/output/", "nvit_output/"):
            if split_at in cks:
                tail = cks.split(split_at, 1)[1]
                return [p for p in tail.split("/") if p]
        if "output" in ckr.parts:
            return list(ckr.parts[ckr.parts.index("output") + 1 :])
        return [ckr.stem]


def exp_label(chapter: str, ckpt: Path) -> str:
    """Human-readable experiment id, e.g. ch5/M0_NoMask/step_45000."""
    parts = _relpath_parts_under_project_output(ckpt)
    # output/ch5_prior_compare/M0_NoMask/train/runs/DATE/checkpoints/xxx.ckpt
    # output/ch6/train/runs/DATE/checkpoints/xxx.ckpt
    tag_parts: list[str] = [chapter]
    if chapter == "ch5" and len(parts) >= 2:
        tag_parts.append(parts[1])  # M0_NoMask
    elif chapter == "ch6" and "train" in parts:
        idx = parts.index("train") if "train" in parts else 0
        if idx + 2 < len(parts):
            tag_parts.append(parts[idx + 2])  # run timestamp folder
    stem = ckpt.stem
    m = re.search(r"step[=:]?(\d+)", stem, re.I)
    if m:
        tag_parts.append(f"step_{m.group(1)}")
    else:
        tag_parts.append(stem)
    return "/".join(tag_parts)


def nvit_json_basename(chapter: str, ckpt: Path) -> str:
    """Filename under json/nvit/: human-readable (group / run id / step), not a path hash."""
    label = exp_label(chapter, ckpt)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.replace("/", "_"))
    return f"{safe}.json"


def append_csv_row(
    csv_path: Path,
    row: dict[str, Any],
    fieldnames: list[str],
) -> None:
    """Append one CSV row; uses flock(2) on POSIX so parallel eval jobs can share metrics_master.csv."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a+", newline="") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (AttributeError, OSError):
            pass
        try:
            empty = os.fstat(f.fileno()).st_size == 0
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if empty:
                w.writeheader()
            w.writerow(row)
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (AttributeError, OSError):
                pass


def run_standard_eval(
    python_exe: Path,
    ckpt: Path,
    datasets: str,
    gpu: str,
    limit_batches: int | None,
    batch_size: int,
    num_workers: int,
    data_dir: Path,
    out_json: Path,
    env: dict[str, str],
    use_mean_alignment: bool = False,
) -> int:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_exe),
        str(STANDARD_EVAL),
        "--ckpt",
        str(ckpt),
        "--dataset",
        datasets,
        "--gpu",
        gpu,
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--data_dir",
        str(data_dir),
        "--output",
        str(out_json),
        "--skip_errors",
    ]
    if limit_batches is not None:
        cmd.extend(["--limit_batches", str(limit_batches)])
    if use_mean_alignment:
        cmd.append("--use_mean_alignment")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.returncode


def json_to_rows(
    family: str,
    chapter: str,
    exp: str,
    ckpt_abs: str,
    json_path: Path,
    limit_batches: str,
    status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        data = json.loads(json_path.read_text())
    except Exception as e:
        return [
            {
                "timestamp_utc": ts,
                "family": family,
                "chapter": chapter,
                "experiment": exp,
                "checkpoint": ckpt_abs,
                "json_path": str(json_path),
                "dataset": "",
                "MPJPE_mm": "",
                "PA_MPJPE_mm": "",
                "KPL2": "",
                "limit_batches": limit_batches,
                "status": f"parse_error:{e}",
            }
        ]
    results = data.get("results") or {}
    for ds_name, m in results.items():
        rows.append(
            {
                "timestamp_utc": ts,
                "family": family,
                "chapter": chapter,
                "experiment": exp,
                "checkpoint": ckpt_abs,
                "json_path": str(json_path),
                "dataset": ds_name,
                "MPJPE_mm": m.get("mode_mpjpe", ""),
                "PA_MPJPE_mm": m.get("mode_re", ""),
                "KPL2": m.get("mode_kpl2", ""),
                "limit_batches": limit_batches,
                "status": status,
            }
        )
    if not rows:
        rows.append(
            {
                "timestamp_utc": ts,
                "family": family,
                "chapter": chapter,
                "experiment": exp,
                "checkpoint": ckpt_abs,
                "json_path": str(json_path),
                "dataset": "",
                "MPJPE_mm": "",
                "PA_MPJPE_mm": "",
                "KPL2": "",
                "limit_batches": limit_batches,
                "status": "no_results_in_json",
            }
        )
    return rows


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

# Ch5 ablation summary (parallel-safe append; one row per group).
CH5_SUMMARY_FIELDS = [
    "group",
    "primary_dataset",
    "best_composite_step",
    "best_composite_checkpoint",
    "rank_sum",
    "mean_rank",
    "best_primary_step",
    "best_primary_checkpoint",
    "aggregate_json",
]


def ingest_smpler_json(
    master_csv: Path,
    label: str,
    json_path: Path,
    chapter: str = "ch6",
) -> None:
    """Append rows from an existing SMPLer eval JSON (same schema as NViT)."""
    raw = json_path.read_text()
    data = json.loads(raw)
    args = data.get("args") or {}
    ckpt_hint = args.get("ckpt", str(json_path))
    lim = args.get("limit_batches", "")
    rows = json_to_rows(
        family="SMPLer",
        chapter=chapter,
        exp=label,
        ckpt_abs=str(ckpt_hint),
        json_path=json_path,
        limit_batches=str(lim) if lim != "" else "",
        status="ingested",
    )
    for r in rows:
        append_csv_row(master_csv, r, FIELDNAMES)


def run_ch5_ablation_all_steps(
    args: argparse.Namespace,
    project: Path,
    data_dir: Path,
    env: dict[str, str],
    master_csv: Path,
) -> None:
    ch5_root = project / "output" / "ch5_prior_compare"
    ab_root: Path = args.out_dir / "ablation" / "ch5"
    nvit_dir: Path = args.out_dir / "json" / "nvit"
    ab_root.mkdir(parents=True, exist_ok=True)
    nvit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    limit_str = "" if args.limit_batches is None else str(args.limit_batches)
    groups = [x.strip() for x in args.ablation_groups.split(",") if x.strip()]
    ds_list = parse_dataset_list(args.datasets)
    primary_ds = ds_list[0] if ds_list else ""
    sum_csv = ab_root / "summary_best_composite.csv"
    if not args.ch5_summary_append:
        sum_csv.unlink(missing_ok=True)

    for gname in groups:
        exp_dir = ch5_root / gname
        run = latest_run_dir_with_step_ckpts(exp_dir)
        if run is None:
            print(f"SKIP {gname}: no train/runs", file=sys.stderr)
            continue
        ckdir = run / "checkpoints"
        if not ckdir.is_dir():
            print(f"SKIP {gname}: no checkpoints", file=sys.stderr)
            continue
        ckpts = all_step_checkpoints(ckdir)
        if not ckpts:
            print(f"SKIP {gname}: no step_step=*.ckpt", file=sys.stderr)
            continue
        if getattr(args, "ch5_ablation_max_step_only", False):
            best = max(ckpts, key=lambda p: _step_from_ckpt_name(p.name) or -1)
            ckpts = [best]

        if args.dry_run:
            print(f"[dry-run] {gname}: {len(ckpts)} checkpoints in {ckdir}")
            continue

        records: list[dict[str, Any]] = []
        for ckpt in ckpts:
            step = _step_from_ckpt_name(ckpt.name)
            raw_json = nvit_dir / nvit_json_basename("ch5", ckpt)
            results: dict[str, Any] = {}
            status = "ok"
            if args.skip_existing_json and raw_json.is_file() and raw_json.stat().st_size > 10:
                print(f"SKIP (exists): {raw_json.name}")
                try:
                    data = json.loads(raw_json.read_text())
                    results = data.get("results") or {}
                except Exception as e:
                    status = f"parse_error:{e}"
            else:
                print(f"EVAL ch5-ablation {gname} step={step} -> {raw_json.name}")
                rc = run_standard_eval(
                    args.python,
                    ckpt,
                    datasets=args.datasets,
                    gpu=args.gpu,
                    limit_batches=args.limit_batches,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    data_dir=data_dir,
                    out_json=raw_json,
                    env=env,
                    use_mean_alignment=args.use_mean_alignment,
                )
                status = "ok" if rc == 0 else f"exit_{rc}"
                try:
                    data = json.loads(raw_json.read_text())
                    results = data.get("results") or {}
                except Exception as e:
                    results = {}
                    status = f"parse_error:{e}"

            rec = {
                "step": step,
                "checkpoint": str(ckpt.resolve()),
                "raw_json": str(raw_json.resolve()),
                "status": status,
                "results": results,
            }
            records.append(rec)

            for ds_name, m in results.items():
                row = {
                    "timestamp_utc": ts,
                    "family": "NViT",
                    "chapter": "ch5",
                    "experiment": f"ch5/{gname}/step_{step}",
                    "checkpoint": str(ckpt.resolve()),
                    "json_path": str(raw_json.resolve()),
                    "dataset": ds_name,
                    "MPJPE_mm": m.get("mode_mpjpe", ""),
                    "PA_MPJPE_mm": m.get("mode_re", ""),
                    "KPL2": m.get("mode_kpl2", ""),
                    "limit_batches": limit_str,
                    "status": status,
                }
                append_csv_row(master_csv, row, FIELDNAMES)

        composite = compute_composite_best(records, args.rank_metric_3d)
        best_primary: dict[str, Any] | None = None
        best_val = float("inf")
        if primary_ds:
            for rec in records:
                m = (rec.get("results") or {}).get(primary_ds) or {}
                v, _ = rank_value_for_dataset(m, primary_ds, args.rank_metric_3d)
                if v is not None and v < best_val:
                    best_val = v
                    best_primary = rec

        agg: dict[str, Any] = {
            "group": gname,
            "chapter": "ch5",
            "run_dir": str(run.resolve()),
            "datasets_arg": args.datasets,
            "rank_metric_3d": args.rank_metric_3d,
            "primary_dataset": primary_ds,
            "num_checkpoints": len(records),
            "records": records,
            "best_composite_rank_sum": composite,
            "best_on_primary_dataset": None
            if best_primary is None
            else {
                "step": best_primary.get("step"),
                "checkpoint": best_primary.get("checkpoint"),
                "dataset": primary_ds,
                "value": best_val,
                "metric_key": args.rank_metric_3d
                if primary_ds in ("3DPW-TEST", "H36M-VAL-P2")
                else "mode_kpl2",
                "results": best_primary.get("results"),
            },
        }
        out_agg = ab_root / f"{gname}.json"
        out_agg.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
        print(f"Wrote aggregate: {out_agg}")

        c = composite or {}
        sr: dict[str, Any] = {
            "group": gname,
            "primary_dataset": primary_ds,
            "best_composite_step": c.get("step", ""),
            "best_composite_checkpoint": c.get("checkpoint", ""),
            "rank_sum": c.get("rank_sum", ""),
            "mean_rank": c.get("mean_rank", ""),
            "best_primary_step": best_primary.get("step") if best_primary else "",
            "best_primary_checkpoint": best_primary.get("checkpoint") if best_primary else "",
            "aggregate_json": str(out_agg.resolve()),
        }
        if not args.dry_run:
            append_csv_row(sum_csv, sr, CH5_SUMMARY_FIELDS)
            print(f"Appended summary row -> {sum_csv}")


def _resolve_ch6_experiment_root(args: argparse.Namespace, project: Path) -> Path:
    raw = getattr(args, "ch6_experiment_dir", None)
    if raw is None:
        return (project / "output" / "ch6").resolve()
    p = Path(raw)
    if not p.is_absolute():
        p = (project / p).resolve()
    else:
        p = p.resolve()
    # Common mistake: `export CH6_EXPERIMENT_DIR="$ROOT/output/foo"` with unset $ROOT -> "/output/foo" (not NViT/output).
    out_root = Path("/output")
    if not p.is_dir() and p != out_root and p.is_relative_to(out_root):
        sub = p.relative_to(out_root)
        alt = (project / "output" / sub).resolve()
        if alt.is_dir():
            print(
                f"ch6: 路径 {p} 不存在，已改用项目下 {alt}（若未设置 ROOT，$ROOT/output/... 会变成 /output/...）",
                file=sys.stderr,
            )
            p = alt
    return p


def run_ch6_all_steps(
    args: argparse.Namespace,
    project: Path,
    data_dir: Path,
    env: dict[str, str],
    master_csv: Path,
) -> None:
    ch6_root = _resolve_ch6_experiment_root(args, project)
    print(f"ch6 experiment root: {ch6_root}", file=sys.stderr)
    ab_root: Path = args.out_dir / "ablation" / "ch6"
    nvit_dir: Path = args.out_dir / "json" / "nvit"
    ab_root.mkdir(parents=True, exist_ok=True)
    nvit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    limit_str = "" if args.limit_batches is None else str(args.limit_batches)

    run = latest_run_dir_with_step_ckpts(ch6_root)
    if run is None:
        print(
            f"SKIP ch6-all-steps: no train/runs with step_step=*.ckpt under {ch6_root}",
            file=sys.stderr,
        )
        kids = ch6_experiment_child_candidates(ch6_root)
        if kids:
            print(
                "ch6: 该路径下没有 <exp>/train/runs/.../checkpoints/step_step=*.ckpt，"
                "但下列**子目录**是合法实验根（请把 CH6_EXPERIMENT_DIR 指到其中一个，不要指父目录）：",
                file=sys.stderr,
            )
            for c in kids[:12]:
                print(f"  {c}", file=sys.stderr)
            if len(kids) > 12:
                print(f"  ... 共 {len(kids)} 个", file=sys.stderr)
        return
    ckdir = run / "checkpoints"
    ckpts = all_step_checkpoints(ckdir)
    if args.ch6_max_checkpoints and args.ch6_max_checkpoints > 0:
        ckpts = ckpts[: args.ch6_max_checkpoints]
    st = max(1, int(args.ch6_shard_total))
    si = int(args.ch6_shard_index)
    if si < 0 or si >= st:
        print(f"ERROR: --ch6-shard-index must satisfy 0 <= index < --ch6-shard-total (got {si}, {st})", file=sys.stderr)
        sys.exit(1)
    if st > 1:
        ckpts = [c for j, c in enumerate(ckpts) if j % st == si]
        print(f"Ch6 shard {si + 1}/{st}: {len(ckpts)} checkpoint(s) after split (round-robin).", file=sys.stderr)
    if not ckpts:
        print("SKIP ch6-all-steps: no step_step=*.ckpt (after shard filter)", file=sys.stderr)
        return

    if args.dry_run:
        print(f"[dry-run] ch6: {len(ckpts)} checkpoints in {ckdir}")
        return

    records: list[dict[str, Any]] = []
    for ckpt in ckpts:
        step = _step_from_ckpt_name(ckpt.name)
        raw_json = nvit_dir / nvit_json_basename("ch6", ckpt)
        results: dict[str, Any] = {}
        status = "ok"
        if args.skip_existing_json and raw_json.is_file() and raw_json.stat().st_size > 10:
            print(f"SKIP (exists): {raw_json.name}")
            try:
                data = json.loads(raw_json.read_text())
                results = data.get("results") or {}
            except Exception as e:
                status = f"parse_error:{e}"
        else:
            print(f"EVAL ch6-all-steps step={step} -> {raw_json.name}")
            rc = run_standard_eval(
                args.python,
                ckpt,
                datasets=args.datasets,
                gpu=args.gpu,
                limit_batches=args.limit_batches,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                data_dir=data_dir,
                out_json=raw_json,
                env=env,
                use_mean_alignment=args.use_mean_alignment,
            )
            status = "ok" if rc == 0 else f"exit_{rc}"
            try:
                data = json.loads(raw_json.read_text())
                results = data.get("results") or {}
            except Exception as e:
                results = {}
                status = f"parse_error:{e}"

        rec = {
            "step": step,
            "checkpoint": str(ckpt.resolve()),
            "raw_json": str(raw_json.resolve()),
            "status": status,
            "results": results,
        }
        records.append(rec)

        for ds_name, m in results.items():
            row = {
                "timestamp_utc": ts,
                "family": "NViT",
                "chapter": "ch6",
                "experiment": f"ch6/step_{step}",
                "checkpoint": str(ckpt.resolve()),
                "json_path": str(raw_json.resolve()),
                "dataset": ds_name,
                "MPJPE_mm": m.get("mode_mpjpe", ""),
                "PA_MPJPE_mm": m.get("mode_re", ""),
                "KPL2": m.get("mode_kpl2", ""),
                "limit_batches": limit_str,
                "status": status,
            }
            append_csv_row(master_csv, row, FIELDNAMES)

    composite = compute_composite_best(records, args.rank_metric_3d)
    ds_list = parse_dataset_list(args.datasets)
    primary_ds = ds_list[0] if ds_list else ""
    best_primary: dict[str, Any] | None = None
    best_val = float("inf")
    if primary_ds:
        for rec in records:
            m = (rec.get("results") or {}).get(primary_ds) or {}
            v, _ = rank_value_for_dataset(m, primary_ds, args.rank_metric_3d)
            if v is not None and v < best_val:
                best_val = v
                best_primary = rec

    agg = {
        "group": "ch6",
        "chapter": "ch6",
        "run_dir": str(run.resolve()),
        "datasets_arg": args.datasets,
        "rank_metric_3d": args.rank_metric_3d,
        "primary_dataset": primary_ds,
        "num_checkpoints": len(records),
        "records": records,
        "best_composite_rank_sum": composite,
        "best_on_primary_dataset": None
        if best_primary is None
        else {
            "step": best_primary.get("step"),
            "checkpoint": best_primary.get("checkpoint"),
            "dataset": primary_ds,
            "value": best_val,
            "results": best_primary.get("results"),
        },
        "shard": None
        if st <= 1
        else {"index": si, "total": st, "note": "partial shard — composite best is for this shard only"},
    }
    out_agg = ab_root / "ch6_latest_run.json"
    if st > 1:
        out_agg = ab_root / f"ch6_latest_run_shard{si + 1}of{st}.json"
        out_agg.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
        print(f"Wrote shard aggregate {out_agg} (merge all shards for global best; metrics_master.csv has all rows).")
    else:
        out_agg.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
        print(f"Wrote {out_agg}")

    sum_csv = ab_root / "summary_best_composite.csv"
    c6 = composite or {}
    row = {
        "group": "ch6",
        "primary_dataset": primary_ds,
        "best_composite_step": c6.get("step", ""),
        "best_composite_checkpoint": c6.get("checkpoint", ""),
        "rank_sum": c6.get("rank_sum", ""),
        "mean_rank": c6.get("mean_rank", ""),
        "best_primary_step": best_primary.get("step") if best_primary else "",
        "best_primary_checkpoint": best_primary.get("checkpoint") if best_primary else "",
        "aggregate_json": str(out_agg.resolve()),
    }
    if st > 1:
        sum_csv = ab_root / f"summary_best_composite_shard{si + 1}of{st}.csv"
        with sum_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"Wrote {sum_csv} (shard-local summary).")
    else:
        with sum_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        print(f"Wrote {sum_csv}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--chapters", type=str, default="ch5,ch6", help="Comma: ch5,ch6")
    p.add_argument(
        "--checkpoint-mode",
        choices=["max-step-per-run", "all"],
        default="max-step-per-run",
        help="max-step-per-run: one ckpt per .../checkpoints/ dir — highest step in filename; "
        "never uses last.ckpt/best.ckpt. all: every .ckpt file (heavy).",
    )
    p.add_argument(
        "--ckpt-glob",
        type=str,
        default="",
        help="If set, filter checkpoint paths containing this substring (e.g. M4_).",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="3DPW-TEST",
        help="Comma-separated, or ALL for the 5 HMR2 eval benchmarks (see standard_eval).",
    )
    p.add_argument(
        "--limit-batches",
        "--limit_batches",
        type=int,
        default=None,
        dest="limit_batches",
        help="Max batches per dataset (omit for full eval).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--gpu", type=str, default="0", help="Logical GPU index after CUDA_VISIBLE_DEVICES.")
    p.add_argument("--cuda-visible-devices", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    p.add_argument(
        "--python",
        type=Path,
        default=Path(os.environ.get("PYTHON", sys.executable)),
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="hmr2_evaluation_data parent; default: HUMANS_ROOT/hmr2_evaluation_data",
    )
    p.add_argument("--humans-root", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="List checkpoints only.")
    p.add_argument(
        "--skip-nvit",
        action="store_true",
        help="Do not evaluate NViT checkpoints (SMPLer ingest / dry-run only).",
    )
    p.add_argument("--max-checkpoints", type=int, default=0, help="0 = no limit.")
    p.add_argument(
        "--skip-existing-json",
        action="store_true",
        help="If json/nvit/<id>.json exists and non-empty, skip re-running eval.",
    )
    p.add_argument(
        "--ingest-smpler-json",
        action="store_true",
        help="Append SMPLer rows from --smpler-3dpw-json / --smpler-h36m-json into CSV without running.",
    )
    p.add_argument(
        "--smpler-3dpw-json",
        type=Path,
        default=None,
        help="Default: artifacts/external_baselines/SMPLer/smpler_3dpw.json if present.",
    )
    p.add_argument("--smpler-h36m-json", type=Path, default=None)
    p.add_argument(
        "--smpler-chapter",
        type=str,
        default="ch6",
        help="Chapter tag for ingested SMPLer rows (ch6 = compare against ch6 NViT).",
    )
    p.add_argument(
        "--ch5-ablation-all-steps",
        action="store_true",
        help="Evaluate every step_step=*.ckpt in each group's latest run (M0–M5); composite-best CSV.",
    )
    p.add_argument(
        "--ch5-ablation-max-step-only",
        action="store_true",
        help="With --ch5-ablation-all-steps: only evaluate the highest-step checkpoint per group; "
        "still writes ablation/ch5/<Group>.json (one step per group).",
    )
    p.add_argument(
        "--ch6-experiment-dir",
        type=Path,
        default=None,
        help="Ch6 training output root (contains train/runs/<date>/checkpoints/step_step=*.ckpt). "
        "Default: <project>/output/ch6. Example: output/ch6_phase2_unfreeze5 or an absolute path.",
    )
    p.add_argument(
        "--ch6-all-steps",
        action="store_true",
        help="Evaluate every step checkpoint in the ch6 experiment dir (see --ch6-experiment-dir) latest run with step_step=*.ckpt (for vs SMPLer on ch6).",
    )
    p.add_argument(
        "--ch6-max-checkpoints",
        type=int,
        default=0,
        help="With --ch6-all-steps: only evaluate this many checkpoints (highest steps first). 0 = all.",
    )
    p.add_argument(
        "--ch6-shard-index",
        type=int,
        default=0,
        help="With --ch6-all-steps: worker id in [0, --ch6-shard-total). Checkpoints are split round-robin.",
    )
    p.add_argument(
        "--ch6-shard-total",
        type=int,
        default=1,
        help="With --ch6-all-steps: number of parallel ch6 workers (1 = single GPU / no sharding).",
    )
    p.add_argument(
        "--ablation-groups",
        type=str,
        default=",".join(DEFAULT_CH5_GROUPS),
        help="Comma-separated folder names under output/ch5_prior_compare/.",
    )
    p.add_argument(
        "--rank-metric-3d",
        dest="rank_metric_3d",
        choices=["mode_re", "mode_mpjpe"],
        default="mode_re",
        help="Metric for ranking 3DPW/H36M in composite rank-sum (default PA-MPJPE).",
    )
    p.add_argument(
        "--use-mean-alignment",
        action="store_true",
        help="Forward --use_mean_alignment to standard_eval (recommended with H36M).",
    )
    p.add_argument(
        "--also-default-eval",
        action="store_true",
        help="After ablation modes, also run the normal --chapters checkpoint scan (max-step or all).",
    )
    p.add_argument(
        "--download-eval-metadata",
        action="store_true",
        help="If hmr2_evaluation_data has no NPZ, download the official 4D-Humans eval metadata tarball.",
    )
    p.add_argument(
        "--verify-eval-data",
        action="store_true",
        help="Check NPZ files for --datasets and exit 1 if any are missing (no eval).",
    )
    p.add_argument(
        "--ch5-summary-append",
        action="store_true",
        help="Do not reset ablation/ch5/summary_best_composite.csv at start; use when sharding "
        "one group per GPU (orchestrator should delete the CSV before launching workers).",
    )
    p.add_argument(
        "--report-eval-data",
        action="store_true",
        help="Print whether each eval NPZ's image paths exist (sample); no GPU eval.",
    )
    p.add_argument(
        "--prepare-eval-layout",
        action="store_true",
        help="Create .../data/{h36m,posetrack,hr-lspet/image} and write eval_data_status.txt.",
    )
    p.add_argument(
        "--fetch-hr-lspet",
        action="store_true",
        help="With --prepare-eval-layout, try to download and unzip MPI HR-LSPET images.",
    )
    p.add_argument(
        "--finish-hr-lspet",
        action="store_true",
        help="If HUMANS_ROOT/_hr-lspet.zip is complete (~2.7GiB), extract into HUMANS_ROOT/data/.",
    )
    args = p.parse_args()

    project = args.project_root
    out_dir: Path = args.out_dir
    json_dir = out_dir / "json" / "nvit"
    master_csv = out_dir / "metrics_master.csv"

    humans_root = args.humans_root
    if humans_root is None:
        hr = os.environ.get("HUMANS_ROOT", str(project.parent / "4D-Humans"))
        humans_root = Path(hr)
    data_dir = args.data_dir or (humans_root / "hmr2_evaluation_data")

    if args.report_eval_data:
        print(report_eval_image_status(humans_root, data_dir), end="")
        return
    if args.finish_hr_lspet:
        sys.exit(finish_hr_lspet_from_zip(humans_root))
    if args.prepare_eval_layout:
        prepare_eval_layout(
            humans_root,
            data_dir,
            args.out_dir / "eval_data_status.txt",
            fetch_hr_lspet=args.fetch_hr_lspet,
        )
        return

    chapters = {c.strip() for c in args.chapters.split(",") if c.strip()}

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project}:{humans_root}" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.verify_eval_data:
        missing = verify_eval_npz(data_dir, args.datasets)
        if missing:
            print(
                f"Missing NPZ under {data_dir}: {missing}. "
                f"Eval metadata tarball: {EVAL_METADATA_TAR_URL}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"OK: required NPZ present under {data_dir} for {args.datasets}.")
        return

    if args.download_eval_metadata:
        if not data_dir.is_dir() or not any(data_dir.glob("*.npz")):
            if not download_eval_metadata_tar(humans_root):
                sys.exit(1)
        else:
            print(f"Eval NPZ already present under {data_dir}; skip download.")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.ch5_ablation_max_step_only and not args.ch5_ablation_all_steps:
        print("ERROR: --ch5-ablation-max-step-only requires --ch5-ablation-all-steps.", file=sys.stderr)
        sys.exit(1)

    default_smpler_3dpw = project / "artifacts" / "external_baselines" / "SMPLer" / "smpler_3dpw.json"
    smpler_3 = args.smpler_3dpw_json or (default_smpler_3dpw if default_smpler_3dpw.is_file() else None)

    # Ingest SMPLer JSON files (baseline rows)
    if args.ingest_smpler_json:
        if smpler_3 and smpler_3.is_file():
            ingest_smpler_json(master_csv, "SMPLer_3dpw_ckpt", smpler_3, chapter=args.smpler_chapter)
            print(f"Ingested SMPLer 3DPW from {smpler_3} (chapter={args.smpler_chapter})")
        if args.smpler_h36m_json and args.smpler_h36m_json.is_file():
            ingest_smpler_json(
                master_csv, "SMPLer_h36m_ckpt", args.smpler_h36m_json, chapter=args.smpler_chapter
            )
            print(f"Ingested SMPLer H36M from {args.smpler_h36m_json} (chapter={args.smpler_chapter})")
        if not (smpler_3 and smpler_3.is_file()) and not (
            args.smpler_h36m_json and args.smpler_h36m_json.is_file()
        ):
            print("Warning: --ingest-smpler-json but no usable SMPLer JSON paths.")

    ablation_any = args.ch5_ablation_all_steps or args.ch6_all_steps
    run_default = (not ablation_any or args.also_default_eval) and not args.skip_nvit

    if args.ch5_ablation_all_steps:
        run_ch5_ablation_all_steps(args, project, data_dir, env, master_csv)

    if args.ch6_all_steps:
        run_ch6_all_steps(args, project, data_dir, env, master_csv)

    pairs: list[tuple[str, Path]] = []
    if run_default:
        if args.checkpoint_mode == "max-step-per-run":
            pairs = discover_ckpts_max_step_per_run(project, chapters)
        else:
            pairs = discover_ckpts(project, chapters)
        if args.ckpt_glob:
            g = args.ckpt_glob
            pairs = [(c, p) for c, p in pairs if g in str(p)]
        if args.max_checkpoints > 0:
            pairs = pairs[: args.max_checkpoints]

    if run_default:
        print(f"Discovered {len(pairs)} checkpoints for default scan (chapters {chapters}).")
    else:
        print("Default checkpoint scan skipped (ablation-only). Pass --also-default-eval to include it.")
    if args.dry_run:
        if run_default:
            for ch, path in pairs[:50]:
                print(f"  [{ch}] {path}")
            if len(pairs) > 50:
                print(f"  ... and {len(pairs) - 50} more")
        return

    limit_str = "" if args.limit_batches is None else str(args.limit_batches)

    for chapter, ckpt in pairs:
        exp = exp_label(chapter, ckpt)
        out_json = json_dir / nvit_json_basename(chapter, ckpt)
        if args.skip_existing_json and out_json.is_file() and out_json.stat().st_size > 10:
            print(f"SKIP (exists, no CSV duplicate): {out_json.name}")
            continue

        print(f"EVAL [{chapter}] {ckpt.name} -> {out_json.name}")
        rc = run_standard_eval(
            args.python,
            ckpt,
            datasets=args.datasets,
            gpu=args.gpu,
            limit_batches=args.limit_batches,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_dir=data_dir,
            out_json=out_json,
            env=env,
            use_mean_alignment=args.use_mean_alignment,
        )
        status = "ok" if rc == 0 else f"exit_{rc}"
        rows = json_to_rows(
            family="NViT",
            chapter=chapter,
            exp=exp,
            ckpt_abs=str(ckpt.resolve()),
            json_path=out_json,
            limit_batches=limit_str,
            status=status,
        )
        for r in rows:
            append_csv_row(master_csv, r, FIELDNAMES)

    print(f"Done. Master table: {master_csv}")
    print(f"Per-run JSON: {json_dir}")


if __name__ == "__main__":
    main()
