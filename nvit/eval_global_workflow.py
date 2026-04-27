"""
全局评测工作流编排（单一入口描述「模型 / 检测数据流 / 内外检测角色」）。

设计目标
--------
- **变量 1 — 模型文件**：任意 ``.ckpt``，由 ``standard_eval`` / ``global_evaluator`` / 遮挡脚本消费。
- **变量 2 — 检测数据流**：人体在输入张量里如何出现；当前仓库两条主路径见 ``DetectionDataFlow``。
- **变量 3 — 内部 vs 外部检测**：指 **人体定位/裁剪信息来源**（与「模型内部表征诊断」不同）：
    - ``INTERNAL``：裁剪与相机框来自 **评测 npz（ImageDataset）**，即当前 ``nvit/skills/evaluate_model/standard_eval.py`` 主线；
    - ``EXTERNAL``：应由 **独立检测器**（如 ViTDet + ``ViTDetDataset``，见 ``nvit/model_manager.py``）产框再送 HMR；**尚未**与 ``standard_eval`` 拼成一条 CLI，本模块在选 ``EXTERNAL`` 时会跳过 standard 并打印说明。

工作流阶段（可多选）
--------------------
- ``standard``：多数据集 MPJPE / PA-MPJPE（及 2D 集的 KPL2）→ ``standard_eval.py``。
- ``internal_diag``：人类学指标 + 层诊断（熵/KTI 等）→ ``python -m nvit.global_evaluator``。
- ``external_occlusion``：固定随机块遮挡曲线 → ``nvit.eval_ch5_external_occlusion``（当前绑定 3DPW-TEST 与 Evaluator）。
- ``mpiinf_extreme``：MPI-INF-3DHP-TEST 逐样本误差 + 基于 GT 的极端姿态子集 + 中文统计图 → ``python -m nvit.eval_mpiinf_extreme_analysis``（需自备 ``mpi_inf_3dhp_test.npz`` 与图像目录）。

用法示例
--------
  # 只打印将执行的子命令（不运行）
  python -m nvit.eval_global_workflow plan \\
      --ckpt /path/to/step.ckpt \\
      --localization internal \\
      --data-flow hmr2_image_dataset \\
      --stages standard,internal_diag \\
      --datasets 3DPW-TEST,H36M-VAL-P2 \\
      --chapter Ch6B --run-label my_run

  # 顺序执行（需已配置 PYTHONPATH / 4D-Humans）
  python -m nvit.eval_global_workflow run --ckpt ...（同上）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class DetectionDataFlow(str, Enum):
    """检测/人体相关的数据从哪条管线来（与具体数据集名字正交）。"""

    HMR2_IMAGE_DATASET = "hmr2_image_dataset"  # npz + ImageDataset（当前 leaderboard 主路径）
    VITDET_VIDEO = "vitdet_video"  # 预留：ModelManager + ViTDetDataset，未与 standard_eval 合并


class DetectorRole(str, Enum):
    """人体框/裁剪来源：内部 = 数据集元数据；外部 = 独立检测器。"""

    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass
class GlobalEvalWorkflow:
    """一次完整评测编排的可变参数容器。"""

    ckpt: Path
    localization: DetectorRole
    data_flow: DetectionDataFlow
    stages: tuple[str, ...] = ("standard",)
    datasets: str = "3DPW-TEST,H36M-VAL-P2"
    chapter: str = "Ch6B"
    run_label: str | None = None
    gpu: str = "0"
    python_exe: Path = field(default_factory=lambda: Path(sys.executable))
    repo_root: Path | None = None
    data_dir: Path | None = None
    out_json: Path | None = None
    diag_batches: int = 20
    use_mean_alignment: bool = True
    batch_size: int = 16
    num_workers: int = 4
    limit_batches: int | None = None
    occlusion_out_json: Path | None = None
    occlusion_group: str = "WORKFLOW"
    extreme_top_pct: float = 10.0
    mpiinf_analysis_out_dir: Path | None = None

    def __post_init__(self) -> None:
        self.ckpt = Path(self.ckpt).resolve()
        if self.repo_root is None:
            self.repo_root = Path(__file__).resolve().parents[1]
        if self.data_dir is None:
            self.data_dir = Path(
                os.environ.get("HUMANS_ROOT", str(self.repo_root.parent / "4D-Humans"))
            ) / "hmr2_evaluation_data"
        if self.out_json is None:
            lab = self.run_label or self.ckpt.stem
            self.out_json = (
                self.repo_root / "outputs" / "eval_global_workflow" / lab / "standard_eval.json"
            )
        if self.occlusion_out_json is None:
            lab = self.run_label or self.ckpt.stem
            self.occlusion_out_json = (
                self.repo_root / "outputs" / "eval_global_workflow" / lab / "external_occlusion.json"
            )


def _env_for_subprocess(repo: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": f"{repo}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }


def build_steps(spec: GlobalEvalWorkflow) -> list[dict[str, Any]]:
    """返回若干 {id, argv, skip_reason?}，便于 dry-run 与审计。"""
    repo = spec.repo_root
    assert repo is not None
    steps: list[dict[str, Any]] = []

    if "standard" in spec.stages:
        if spec.localization == DetectorRole.EXTERNAL:
            steps.append(
                {
                    "id": "standard",
                    "argv": None,
                    "skip_reason": "EXTERNAL 检测尚未与 standard_eval 打通；请用 model_manager / ViTDetDataset 单独脚本或改为 INTERNAL。",
                }
            )
        elif spec.data_flow != DetectionDataFlow.HMR2_IMAGE_DATASET:
            steps.append(
                {
                    "id": "standard",
                    "argv": None,
                    "skip_reason": f"data_flow={spec.data_flow} 暂无对应 standard_eval 实现。",
                }
            )
        else:
            se = repo / "nvit" / "skills" / "evaluate_model" / "standard_eval.py"
            argv = [
                str(spec.python_exe),
                str(se),
                "--ckpt",
                str(spec.ckpt),
                "--dataset",
                spec.datasets,
                "--gpu",
                spec.gpu,
                "--batch_size",
                str(spec.batch_size),
                "--num_workers",
                str(spec.num_workers),
                "--data_dir",
                str(spec.data_dir),
                "--output",
                str(spec.out_json),
                "--skip_errors",
            ]
            if spec.limit_batches is not None:
                argv.extend(["--limit_batches", str(spec.limit_batches)])
            if spec.use_mean_alignment:
                argv.append("--use_mean_alignment")
            steps.append({"id": "standard", "argv": argv})

    if "internal_diag" in spec.stages:
        label = spec.run_label or spec.ckpt.stem.replace("/", "_")
        argv = [
            str(spec.python_exe),
            "-m",
            "nvit.global_evaluator",
            "--chapter",
            spec.chapter,
            "--checkpoint_path",
            str(spec.ckpt),
            "--run_label",
            label,
            "--gpu",
            spec.gpu,
            "--diag_batches",
            str(spec.diag_batches),
            "--datasets",
            spec.datasets,
        ]
        if spec.limit_batches is not None:
            argv.extend(["--limit_batches", str(spec.limit_batches)])
        steps.append({"id": "internal_diag", "argv": argv})

    if "external_occlusion" in spec.stages:
        eo = repo / "nvit" / "eval_ch5_external_occlusion.py"
        argv = [
            str(spec.python_exe),
            str(eo),
            "--ckpt",
            str(spec.ckpt),
            "--group",
            spec.occlusion_group,
            "--output_json",
            str(spec.occlusion_out_json),
            "--gpu",
            str(int(spec.gpu) if spec.gpu.isdigit() else 0),
        ]
        if spec.limit_batches is not None:
            argv.extend(["--limit_batches", str(spec.limit_batches)])
        steps.append({"id": "external_occlusion", "argv": argv})

    if "mpiinf_extreme" in spec.stages:
        if spec.localization != DetectorRole.INTERNAL:
            steps.append(
                {
                    "id": "mpiinf_extreme",
                    "argv": None,
                    "skip_reason": "mpiinf_extreme 仅支持 INTERNAL（npz 框裁剪）；请改用 --localization internal。",
                }
            )
        elif spec.data_flow != DetectionDataFlow.HMR2_IMAGE_DATASET:
            steps.append(
                {
                    "id": "mpiinf_extreme",
                    "argv": None,
                    "skip_reason": f"mpiinf_extreme 需 data_flow=hmr2_image_dataset，当前为 {spec.data_flow}。",
                }
            )
        else:
            lab = spec.run_label or spec.ckpt.stem.replace("/", "_")
            out_dir = spec.mpiinf_analysis_out_dir or (
                spec.repo_root / "outputs" / "eval_global" / spec.chapter / f"mpiinf_extreme__{lab}"
            )
            argv = [
                str(spec.python_exe),
                "-m",
                "nvit.eval_mpiinf_extreme_analysis",
                "--ckpt",
                str(spec.ckpt),
                "--gpu",
                spec.gpu,
                "--data-dir",
                str(spec.data_dir),
                "--batch-size",
                str(spec.batch_size),
                "--num-workers",
                str(spec.num_workers),
                "--extreme-top-pct",
                str(spec.extreme_top_pct),
                "--out-dir",
                str(out_dir),
            ]
            if spec.limit_batches is not None:
                argv.extend(["--limit-batches", str(spec.limit_batches)])
            if not spec.use_mean_alignment:
                argv.append("--no-mean-alignment")
            steps.append({"id": "mpiinf_extreme", "argv": argv})

    return steps


def run_steps(spec: GlobalEvalWorkflow, dry_run: bool = False) -> int:
    repo = spec.repo_root
    assert repo is not None
    env = _env_for_subprocess(repo)
    for step in build_steps(spec):
        sid = step["id"]
        if step.get("skip_reason"):
            print(f"[{sid}] SKIP: {step['skip_reason']}", file=sys.stderr)
            continue
        argv = step["argv"]
        print(f"[{sid}] " + " ".join(str(x) for x in argv))
        if dry_run:
            continue
        if sid == "standard" and spec.out_json is not None:
            spec.out_json.parent.mkdir(parents=True, exist_ok=True)
        if sid == "external_occlusion" and spec.occlusion_out_json is not None:
            spec.occlusion_out_json.parent.mkdir(parents=True, exist_ok=True)
        if sid == "mpiinf_extreme" and argv is not None:
            try:
                i = argv.index("--out-dir")
                Path(argv[i + 1]).mkdir(parents=True, exist_ok=True)
            except (ValueError, IndexError):
                pass
        rc = subprocess.run(argv, cwd=str(repo), env=env).returncode
        if rc != 0:
            print(f"[{sid}] exit {rc}", file=sys.stderr)
            return rc
    return 0


def _parse_stages(s: str) -> tuple[str, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    allowed = {"standard", "internal_diag", "external_occlusion", "mpiinf_extreme"}
    for p in parts:
        if p not in allowed:
            raise SystemExit(f"未知 stage: {p}，允许: {sorted(allowed)}")
    return tuple(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--ckpt", type=Path, required=True)
        p.add_argument(
            "--localization",
            choices=[x.value for x in DetectorRole],
            default=DetectorRole.INTERNAL.value,
        )
        p.add_argument(
            "--data-flow",
            choices=[x.value for x in DetectionDataFlow],
            default=DetectionDataFlow.HMR2_IMAGE_DATASET.value,
        )
        p.add_argument(
            "--stages",
            type=str,
            default="standard",
            help="逗号分隔: standard,internal_diag,external_occlusion,mpiinf_extreme",
        )
        p.add_argument("--datasets", type=str, default="3DPW-TEST,H36M-VAL-P2")
        p.add_argument("--chapter", type=str, default="Ch6B", help="internal_diag 用: Ch5 | Ch6A | Ch6B")
        p.add_argument("--run-label", type=str, default=None)
        p.add_argument("--gpu", type=str, default="0")
        p.add_argument("--python", type=Path, default=None, help="默认当前解释器")
        p.add_argument("--repo", type=Path, default=None, help="NViT-master 根，默认自动推断")
        p.add_argument("--data-dir", type=Path, default=None, help="hmr2_evaluation_data 目录")
        p.add_argument("--out-json", type=Path, default=None)
        p.add_argument("--diag-batches", type=int, default=20)
        p.add_argument("--no-mean-alignment", action="store_true")
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--limit-batches", type=int, default=None)
        p.add_argument("--occlusion-out-json", type=Path, default=None)
        p.add_argument("--occlusion-group", type=str, default="WORKFLOW")
        p.add_argument(
            "--extreme-top-pct",
            type=float,
            default=10.0,
            help="mpiinf_extreme：按 GT 姿态伸展度取前若干 %% 为极端子集",
        )
        p.add_argument(
            "--mpiinf-out-dir",
            type=Path,
            default=None,
            help="mpiinf_extreme 输出目录（默认 outputs/eval_global/<chapter>/mpiinf_extreme__<run_label>）",
        )

    p_plan = sub.add_parser("plan", help="只打印 JSON 步骤")
    _shared(p_plan)

    p_run = sub.add_parser("run", help="执行各阶段子进程")
    _shared(p_run)

    args = ap.parse_args()
    py = args.python or Path(sys.executable)
    spec = GlobalEvalWorkflow(
        ckpt=args.ckpt,
        localization=DetectorRole(args.localization),
        data_flow=DetectionDataFlow(args.data_flow),
        stages=_parse_stages(args.stages),
        datasets=args.datasets,
        chapter=args.chapter,
        run_label=args.run_label,
        gpu=args.gpu,
        python_exe=py,
        repo_root=args.repo,
        data_dir=args.data_dir,
        out_json=args.out_json,
        diag_batches=args.diag_batches,
        use_mean_alignment=not args.no_mean_alignment,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_batches=args.limit_batches,
        occlusion_out_json=args.occlusion_out_json,
        occlusion_group=args.occlusion_group,
        extreme_top_pct=args.extreme_top_pct,
        mpiinf_analysis_out_dir=args.mpiinf_out_dir,
    )

    if args.cmd == "plan":
        print(json.dumps([{k: v for k, v in s.items() if k != "argv" or v is not None} for s in build_steps(spec)], indent=2))
        for s in build_steps(spec):
            if s.get("argv"):
                print("\n#", s["id"], "\n", " ".join(str(x) for x in s["argv"]))
        return 0
    return run_steps(spec, dry_run=False)


if __name__ == "__main__":
    raise SystemExit(main())
