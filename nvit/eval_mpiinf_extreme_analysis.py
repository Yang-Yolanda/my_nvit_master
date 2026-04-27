"""
MPI-INF-3DHP-TEST：逐样本 MPJPE / PA-MPJPE、全量与「极端姿态」子集统计，并生成中文大图（无总标题）。

极端样本定义（与模型预测无关，仅由 npz 中 GT 3D 关键点导出）：
  对评估关节子集做均值根对齐后，取各关节到根的最大欧氏距离作为「姿态伸展度」；
  按该分数取 top ``--extreme-top-pct`` 百分比的帧为极端子集。

依赖与 standard_eval 相同（hmr2 / yacs / 数据路径）；图像根目录见 ``resolve_eval_img_dir``。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hmr2.configs import dataset_eval_config
from hmr2.datasets import create_dataset
from hmr2.utils import Evaluator, recursive_to

from nvit.utils.model_io import load_model_from_ckpt
from nvit.utils.path_utils import resolve_eval_img_dir


def _zh_font_path() -> Path:
    return _REPO / "artifacts" / "fonts" / "NotoSansSC-Regular.ttf"


def _setup_matplotlib_zh() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    fp = _zh_font_path()
    if fp.is_file():
        fm.fontManager.addfont(str(fp))
        prop = fm.FontProperties(fname=str(fp))
        plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )


def _build_keypoints_3d_from_npz(data: np.lib.npyio.NpzFile) -> np.ndarray:
    """与 ``ImageDataset`` 一致的 44×4 拼接与置信度掩码。"""
    n = len(data["imgname"])
    try:
        body_keypoints_3d = data["body_keypoints_3d"].astype(np.float32)
    except KeyError:
        body_keypoints_3d = np.zeros((n, 25, 4), dtype=np.float32)
    try:
        extra_keypoints_3d = data["extra_keypoints_3d"].astype(np.float32)
    except KeyError:
        extra_keypoints_3d = np.zeros((n, 19, 4), dtype=np.float32)
    body_keypoints_3d = body_keypoints_3d.copy()
    body_keypoints_3d[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], -1] = 0
    return np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1).astype(np.float32)


def _pose_extent_scores(keypoints_3d: np.ndarray, keypoint_list: list[int]) -> np.ndarray:
    """每帧一个标量：评估关节相对质心的最大距离（毫米量级与 GT 坐标一致）。"""
    scores = np.zeros(len(keypoints_3d), dtype=np.float64)
    for i in range(len(keypoints_3d)):
        k = keypoints_3d[i, keypoint_list, :3]
        root = k.mean(axis=0, keepdims=True)
        rr = k - root
        scores[i] = float(np.linalg.norm(rr, axis=1).max())
    return scores


def _percentiles(x: np.ndarray, ps: list[float]) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    out: dict[str, float] = {}
    for p in ps:
        out[f"p{int(p)}"] = float(np.percentile(x, p))
    return out


def _summarize(name: str, mpjpe: np.ndarray, pampjpe: np.ndarray) -> dict[str, Any]:
    mpjpe = np.asarray(mpjpe, dtype=np.float64).ravel()
    pa = np.asarray(pampjpe, dtype=np.float64).ravel()
    n = int(mpjpe.size)
    if n == 0:
        return {
            "name": name,
            "n": 0,
            "mpjpe_mean_mm": float("nan"),
            "mpjpe_std_mm": float("nan"),
            "mpjpe_median_mm": float("nan"),
            "pampjpe_mean_mm": float("nan"),
            "pampjpe_std_mm": float("nan"),
            "pampjpe_median_mm": float("nan"),
            "mpjpe_percentiles_mm": {},
            "pampjpe_percentiles_mm": {},
            "stability": {
                "mpjpe_iqr_mm": float("nan"),
                "pampjpe_iqr_mm": float("nan"),
                "mpjpe_mad_mm": float("nan"),
                "pampjpe_mad_mm": float("nan"),
            },
        }
    return {
        "name": name,
        "n": n,
        "mpjpe_mean_mm": float(mpjpe.mean()),
        "mpjpe_std_mm": float(mpjpe.std()),
        "mpjpe_median_mm": float(np.median(mpjpe)),
        "pampjpe_mean_mm": float(pa.mean()),
        "pampjpe_std_mm": float(pa.std()),
        "pampjpe_median_mm": float(np.median(pa)),
        "mpjpe_percentiles_mm": _percentiles(mpjpe, [50, 75, 90, 95, 99]),
        "pampjpe_percentiles_mm": _percentiles(pa, [50, 75, 90, 95, 99]),
        "stability": {
            "mpjpe_iqr_mm": float(np.percentile(mpjpe, 75) - np.percentile(mpjpe, 25)),
            "pampjpe_iqr_mm": float(np.percentile(pa, 75) - np.percentile(pa, 25)),
            "mpjpe_mad_mm": float(np.median(np.abs(mpjpe - np.median(mpjpe)))),
            "pampjpe_mad_mm": float(np.median(np.abs(pa - np.median(pa)))),
        },
    }


def _plots(
    out_dir: Path,
    mpjpe: np.ndarray,
    pampjpe: np.ndarray,
    extreme_mask: np.ndarray,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    _setup_matplotlib_zh()
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = extreme_mask.astype(bool)
    full_pa = pampjpe
    ex_pa = pampjpe[ex]

    # 1) PA-MPJPE 直方图（全量 + 极端叠加）
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(full_pa, bins=45, density=True, alpha=0.55, color="steelblue", label="全部样本")
    if ex_pa.size > 0:
        ax.hist(ex_pa, bins=30, density=True, alpha=0.55, color="darkorange", label="极端姿态子集")
    ax.axvline(float(np.mean(full_pa)), color="navy", ls="--", lw=2.0, label=f"全部均值 {float(np.mean(full_pa)):.1f} mm")
    if ex_pa.size > 0:
        ax.axvline(float(np.mean(ex_pa)), color="darkred", ls=":", lw=2.0, label=f"极端均值 {float(np.mean(ex_pa)):.1f} mm")
    ax.set_xlabel("PA-MPJPE（毫米）")
    ax.set_ylabel("密度")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_dir / "mpiinf_pampjpe_hist_zh.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 2) 箱线图：全量 vs 极端（MPJPE / PA）
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8))
    labels = ["全部", "极端"]
    axes[0].boxplot(
        [mpjpe[~ex], mpjpe[ex]],
        labels=labels,
        showmeans=True,
        meanline=True,
    )
    axes[0].set_ylabel("MPJPE（毫米）")
    axes[0].grid(True, axis="y", alpha=0.35, linestyle="--")
    axes[1].boxplot(
        [pampjpe[~ex], pampjpe[ex]],
        labels=labels,
        showmeans=True,
        meanline=True,
    )
    axes[1].set_ylabel("PA-MPJPE（毫米）")
    axes[1].grid(True, axis="y", alpha=0.35, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "mpiinf_box_full_vs_extreme_zh.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 3) PA-MPJPE 经验 CDF（稳定性 / 尾部）
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xs = np.sort(full_pa)
    ys = np.linspace(0, 1, len(xs), endpoint=False)
    ax.plot(xs, ys, lw=2.2, color="teal", label="全部样本 CDF")
    for p, lab in [(50, "中位数"), (90, "90% 分位"), (95, "95% 分位")]:
        v = float(np.percentile(full_pa, p))
        ax.axvline(v, ls="--", alpha=0.85, label=f"{lab} {v:.1f} mm")
    ax.set_xlabel("PA-MPJPE（毫米）")
    ax.set_ylabel("累积比例")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(loc="lower right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_dir / "mpiinf_pampjpe_cdf_zh.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def ch5_latest_step_ckpt(ch5_base: Path, group: str = "M5_8PlusHard") -> Path:
    runs_root = ch5_base / group / "train" / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(f"未找到 Ch5 训练目录: {runs_root}")
    cands = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    run = None
    for r in reversed(cands):
        if (r / "checkpoints").is_dir():
            run = r
            break
    if run is None:
        raise FileNotFoundError(f"{runs_root} 下无含 checkpoints 的 run")
    ck_dir = run / "checkpoints"
    steps: list[Path] = []
    for p in ck_dir.glob("step_step=*.ckpt"):
        m = re.search(r"step_step=(\d+)\.ckpt$", p.name)
        if m:
            steps.append(p)
    if not steps:
        last = ck_dir / "last.ckpt"
        if last.is_file():
            return last
        raise FileNotFoundError(f"{ck_dir} 下无 step_step=*.ckpt")
    return max(steps, key=lambda p: int(re.search(r"step_step=(\d+)\.ckpt$", p.name).group(1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=None, help="checkpoint；若与 --auto-ch5-latest-step 同用则忽略")
    ap.add_argument(
        "--auto-ch5-latest-step",
        action="store_true",
        help=f"使用 {{repo}}/output/ch5_prior_compare/{{--ch5-group}} 下最新 run 的最大 step_step=*.ckpt",
    )
    ap.add_argument("--ch5-base", type=Path, default=None, help="Ch5 消融根目录，默认 <repo>/output/ch5_prior_compare")
    ap.add_argument("--ch5-group", type=str, default="M5_8PlusHard")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--data-dir", type=Path, default=None, help="hmr2_evaluation_data，未设则用 HUMANS_ROOT/hmr2_evaluation_data")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--extreme-top-pct", type=float, default=10.0, help="按 GT 姿态伸展度取前若干 %% 为极端子集")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--no-mean-alignment", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--skip-errors",
        action="store_true",
        default=False,
        help="跳过失败 batch（会导致与 npz 索引不对齐，仅调试 smoke 使用）",
    )
    args = ap.parse_args()

    repo = _REPO
    if args.auto_ch5_latest_step:
        ch5_base = args.ch5_base or (repo / "output" / "ch5_prior_compare")
        ckpt = ch5_latest_step_ckpt(ch5_base, args.ch5_group)
    else:
        if args.ckpt is None:
            print("请指定 --ckpt 或使用 --auto-ch5-latest-step", file=sys.stderr)
            return 2
        ckpt = Path(args.ckpt).resolve()

    if not ckpt.is_file():
        print(f"checkpoint 不存在: {ckpt}", file=sys.stderr)
        return 1

    humans = Path(os.environ.get("HUMANS_ROOT", str(repo.parent / "4D-Humans")))
    data_dir = args.data_dir or (humans / "hmr2_evaluation_data")
    data_dir = Path(data_dir)

    cfg_eval = dataset_eval_config()
    ds_name = "MPI-INF-3DHP-TEST"
    if ds_name not in cfg_eval:
        print(f"配置中无 {ds_name}", file=sys.stderr)
        return 1
    dataset_cfg = cfg_eval[ds_name]
    dataset_cfg.defrost()
    dataset_cfg.DATASET_FILE = str(data_dir / Path(dataset_cfg.DATASET_FILE).name)
    if hasattr(dataset_cfg, "IMG_DIR"):
        dataset_cfg.IMG_DIR = resolve_eval_img_dir(ds_name, dataset_cfg.IMG_DIR)
    dataset_cfg.freeze()
    npz_path = Path(dataset_cfg.DATASET_FILE)
    if not npz_path.is_file():
        print(
            f"未找到评测 npz: {npz_path}\n"
            f"请将 mpi_inf_3dhp_test.npz 放入 {data_dir}，并设置 HMR2_EVAL_IMG_DIR_MPIINF 指向图像根。",
            file=sys.stderr,
        )
        return 1

    raw = np.load(npz_path, allow_pickle=True)
    kp3d_all = _build_keypoints_3d_from_npz(raw)
    keypoint_list = list(dataset_cfg.KEYPOINT_LIST)
    scores = _pose_extent_scores(kp3d_all, keypoint_list)
    thr = float(np.percentile(scores, 100.0 - args.extreme_top_pct))
    extreme_mask = scores >= thr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_ckpt(str(ckpt), device=device)
    model.eval()

    dataset = create_dataset(model.cfg, dataset_cfg, train=False)
    n_total = len(dataset)
    if len(scores) != n_total:
        print(
            f"警告: npz 样本数 {len(scores)} 与 ImageDataset 长度 {n_total} 不一致，"
            "极端掩码将按较短长度截断对齐。",
            file=sys.stderr,
        )
    n_eff = min(len(scores), n_total)
    scores = scores[:n_eff]
    extreme_mask = extreme_mask[:n_eff]

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    hmr2_evaluator = Evaluator(
        dataset_length=n_total,
        keypoint_list=keypoint_list,
        pelvis_ind=0,
        metrics=["mode_mpjpe", "mode_re"],
        pck_thresholds=None,
    )

    lab = ckpt.stem.replace("/", "_")
    out_dir = args.out_dir or (repo / "outputs" / "eval_global" / "Ch5" / f"mpiinf_extreme__{lab}")
    out_dir = Path(out_dir)
    out_json = args.out_json or (out_dir / "mpiinf_extreme_summary.json")

    use_mean_alignment = not args.no_mean_alignment
    pbar = tqdm(total=len(dl), desc=ds_name)
    for bi, batch in enumerate(dl):
        if args.limit_batches is not None and bi >= args.limit_batches:
            break
        try:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
                if use_mean_alignment:
                    pred_eval_kps = out["pred_keypoints_3d"][:, keypoint_list]
                    gt_eval_kps = batch["keypoints_3d"][:, keypoint_list, :-1]
                    pred_root = pred_eval_kps.mean(dim=1, keepdim=True)
                    gt_root = gt_eval_kps.mean(dim=1, keepdim=True)
                    out["pred_keypoints_3d"] = out["pred_keypoints_3d"] - pred_root
                    batch["keypoints_3d"][:, :, :-1] = batch["keypoints_3d"][:, :, :-1] - gt_root
                    max_idx = out["pred_keypoints_3d"].shape[1]
                    if max_idx > 0:
                        out["pred_keypoints_3d"][:, 0] = 0
                        batch["keypoints_3d"][:, 0, :-1] = 0
                        hmr2_evaluator.pelvis_ind = 0
                hmr2_evaluator(out, batch)
        except Exception as e:
            if args.skip_errors:
                print(f"跳过 batch {bi}: {e}", file=sys.stderr)
            else:
                raise
        pbar.update(1)
    pbar.close()

    c = hmr2_evaluator.counter
    mpjpe = np.asarray(hmr2_evaluator.mode_mpjpe[:c], dtype=np.float64)
    pampjpe = np.asarray(hmr2_evaluator.mode_re[:c], dtype=np.float64)
    mask = extreme_mask[:c]

    summary: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "dataset": ds_name,
        "npz": str(npz_path),
        "n_evaluated": int(c),
        "extreme_top_pct": float(args.extreme_top_pct),
        "extreme_threshold_pose_extent": float(thr),
        "extreme_count": int(mask.sum()),
        "full": _summarize("full", mpjpe, pampjpe),
        "extreme": _summarize("extreme", mpjpe[mask], pampjpe[mask]),
        "non_extreme": _summarize("non_extreme", mpjpe[~mask], pampjpe[~mask]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.no_plots:
        if not _zh_font_path().is_file():
            print(
                f"提示: 未找到中文字体 {_zh_font_path()}，图仍保存但中文可能缺字；"
                "可下载 NotoSansSC-Regular.ttf 到 artifacts/fonts/。",
                file=sys.stderr,
            )
        _plots(out_dir, mpjpe, pampjpe, mask, args.dpi)
        print(f"图已写入: {out_dir}")

    print(f"JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
