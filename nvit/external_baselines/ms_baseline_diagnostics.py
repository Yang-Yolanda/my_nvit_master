#!/usr/bin/env python3
"""
METRO / Mesh Graphormer — local diagnostics bridge for NViT.

What this does
--------------
* **demo** (default): tiny multi-head attention stack; always runs offline. Computes
  entropy, effective rank (token features), lightweight KTI-style scalars, and CUDA
  latency. For **paper-identical** KTI to NViT's ViTDiagnosticLab, use ``--mode graphormer``
  with a working env so the lazy probe can import ``scientific_diagnostics`` (hmr2 on PYTHONPATH).
* **graphormer**: if ``MESHGRAPHORMER_ROOT`` (or ``--graphormer-root``) contains a
  *complete* install (real ``modeling_bert.py``, SMPL assets, HRNet weights), builds
  Graphormer, runs N random or HMR2-crop forwards, aggregates attention metrics.

Official repos use **git submodules** for BERT code; run::

    bash scripts/sync_microsoft_hmr_baselines.sh

then set ``MESHGRAPHORMER_ROOT`` to ``NViT-master/third_party/MeshGraphormer``.

Full MPJPE on 3DPW/H36M in *your* CH5 evaluator is not wired here (different mesh
pipeline); use their ``run_gphmer_bodymesh.py`` / METRO scripts for paper numbers,
or add an adapter later.

Example
-------
  # Offline sanity check
  python nvit/external_baselines/ms_baseline_diagnostics.py --mode demo --warmup 20 --iters 100

  # Graphormer (needs complete repo + checkpoint + SMPL files under repo)
  export MESHGRAPHORMER_ROOT=/path/to/MeshGraphormer
  python nvit/external_baselines/ms_baseline_diagnostics.py --mode graphormer \\
    --checkpoint /path/to/graphormer_state_dict.bin \\
    --num-forwards 16 --warmup 10 --iters 50
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# NViT root on PYTHONPATH
_NVIT_ROOT = Path(__file__).resolve().parents[2]
if str(_NVIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_NVIT_ROOT))


def _entropy(attn: torch.Tensor) -> float:
    """attn: (B, H, N, N) softmax probabilities."""
    if torch.isnan(attn).any():
        return 0.0
    eps = 1e-9
    p = attn + eps
    return float((-(p * torch.log(p)).sum(dim=-1)).mean().cpu())


def _effective_rank_tokens(feat: torch.Tensor) -> float:
    """feat: (B, N, D). Effective rank = exp(entropy of normalized singular values), mean over batch."""
    if feat.dim() != 3:
        return 0.0
    ranks: List[float] = []
    for b in range(feat.shape[0]):
        x = feat[b].float()
        try:
            _, s, _ = torch.linalg.svd(x)
            s = s[s > 1e-8]
            if s.numel() == 0:
                ranks.append(0.0)
                continue
            p = s / (s.sum() + 1e-9)
            h = -(p * torch.log(p + 1e-9)).sum()
            ranks.append(float(torch.exp(h).cpu()))
        except Exception:
            ranks.append(0.0)
    return float(np.mean(ranks)) if ranks else 0.0


def _kti_edge_ratio(attn: torch.Tensor, reduce: str = "mean") -> Any:
    """
    Lightweight KTI proxy matching ViTDiagnosticLab edge_ratio path when no keypoints:
    mean over heads of (attn * adj).sum / attn.sum for binary adjacency from uniform topology.
    Here we use identity adjacency N x N for a sanity metric (not physically grounded).
    """
    # attn: B, H, N, N
    b, h, n, _ = attn.shape
    adj = torch.eye(n, device=attn.device, dtype=attn.dtype).view(1, 1, n, n)
    num = (attn * adj).sum(dim=(-2, -1))
    den = attn.sum(dim=(-2, -1)) + 1e-9
    per = (num / den).view(-1)
    if reduce == "mean":
        return float(per.mean().cpu())
    return per.cpu().numpy()


def _kti_dist_corr_proxy(attn: torch.Tensor) -> float:
    """Cheap proxy: correlation between attn row-marginal and uniform (not full ViTDiagnosticLab)."""
    b, h, n, _ = attn.shape
    u = torch.full((n,), 1.0 / n, device=attn.device, dtype=attn.dtype)
    row = attn.mean(dim=-1).mean(dim=1)  # B, N
    scores = []
    for bi in range(b):
        r = row[bi]
        r = r / (r.sum() + 1e-9)
        scores.append(float((r * u).sum().cpu()))
    return float(np.mean(scores))


def _aggregate_attn_metrics_local(weights: List[torch.Tensor]) -> Dict[str, Any]:
    entropies = [_entropy(w) for w in weights]
    ktis = [_kti_edge_ratio(w) for w in weights]
    kt2 = [_kti_dist_corr_proxy(w) for w in weights]
    return {
        "per_layer_entropy": entropies,
        "per_layer_kti_edge_ratio": ktis,
        "per_layer_kti_dist_corr": kt2,
        "mean_entropy": float(np.mean(entropies)) if entropies else None,
        "mean_kti_edge_ratio": float(np.mean(ktis)) if ktis else None,
        "mean_kti_dist_corr": float(np.mean(kt2)) if kt2 else None,
    }


def _metric_probe_lazy(kti_mode: str = "edge_ratio"):
    """Optional: full ViTDiagnosticLab (pulls hmr2). Only for graphormer if needed."""
    from nvit.skills.evaluate_model.scientific_diagnostics import ModelWrapper, ViTDiagnosticLab

    class _BB:
        blocks: list = []

    class _W(ModelWrapper):
        def __init__(self) -> None:
            super().__init__(nn.Identity())

        def get_backbone(self):
            return _BB()

    td = tempfile.mkdtemp(prefix="nvit_ms_probe_")
    return ViTDiagnosticLab(_W(), model_name="probe", output_root=td, kti_mode=kti_mode)


def _repo_bert_ready(root: Path) -> Tuple[bool, str]:
    mb = root / "src/modeling/bert/modeling_bert.py"
    if not mb.is_file():
        return False, f"missing {mb}"
    try:
        if mb.is_symlink():
            tgt = mb.readlink()
            if not mb.resolve().is_file():
                return False, f"{mb} is broken symlink -> {tgt} (run git submodule update --init --recursive)"
        sz = mb.stat().st_size
        if sz < 500:
            return False, f"{mb} is only {sz} bytes (submodule not checked out?)"
    except OSError as e:
        return False, str(e)
    return True, "ok"


def _try_init_submodules(root: Path) -> None:
    gitdir = root / ".git"
    if not gitdir.exists():
        return
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=str(root),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class _AttnBlock(nn.Module):
    """Minimal block with MultiheadAttention; returns attention weights."""

    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.ln1(x)
        y, aw = self.mha(h, h, h, need_weights=True, average_attn_weights=False)
        x = x + y
        x = x + self.ff(self.ln2(x))
        return x, aw


class DemoTransformer(nn.Module):
    def __init__(self, depth: int = 4, d_model: int = 128, nhead: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_AttnBlock(d_model, nhead) for _ in range(depth)]
        )
        self.embed = nn.Linear(3, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.embed(x)
        weights: List[torch.Tensor] = []
        for blk in self.blocks:
            x, aw = blk(x)
            weights.append(aw)
        return x, weights


def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = DemoTransformer(depth=args.depth, d_model=args.d_model, nhead=args.nhead).to(
        device
    )
    model.eval()
    B, N, C = 2, 49, 3
    x = torch.randn(B, N, C, device=device)

    # latency
    if device.type == "cuda":
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            with torch.no_grad():
                out, w = model(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms = (t1 - t0) / args.iters * 1000.0
    else:
        t0 = time.perf_counter()
        for _ in range(args.iters):
            with torch.no_grad():
                out, w = model(x)
        ms = (time.perf_counter() - t0) / args.iters * 1000.0

    with torch.no_grad():
        out, w = model(x)
    attn_stats = _aggregate_attn_metrics_local(w)
    er = _effective_rank_tokens(out)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    return {
        "mode": "demo",
        "device": str(device),
        "params_m": params_m,
        "latency_ms_per_forward": ms,
        "effective_rank_output_tokens": er,
        "attention": attn_stats,
        "kti_mode": args.kti_mode,
    }


def _ensure_graphormer_smpl_assets(
    graphormer_root: Path, humans_data: Optional[Path]
) -> None:
    """Best-effort: copy neutral SMPL pkl into Graphormer expected path."""
    data_dir = graphormer_root / "src/modeling/data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dst = data_dir / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    if dst.is_file():
        return
    candidates = []
    if humans_data:
        candidates.append(
            humans_data / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
        )
    candidates.append(
        Path("/cpfs_infra/shared/yangz/4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
    )
    for src in candidates:
        if src and src.is_file():
            import shutil

            shutil.copy2(src, dst)
            return
    raise FileNotFoundError(
        "Graphormer expects SMPL pkl under src/modeling/data/. "
        "Copy basicModel_neutral_lbs_10_207_0_v1.0.0.pkl there or set --humans-data."
    )


def run_graphormer(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(
        args.graphormer_root
        or os.environ.get("MESHGRAPHORMER_ROOT", "")
        or _NVIT_ROOT / "third_party" / "MeshGraphormer"
    ).resolve()
    _try_init_submodules(root)
    ok, msg = _repo_bert_ready(root)
    if not ok:
        raise RuntimeError(
            f"MeshGraphormer repo incomplete: {msg}\n"
            f"Run: bash {_NVIT_ROOT}/scripts/sync_microsoft_hmr_baselines.sh\n"
            f"Then: cd {root} && git submodule update --init --recursive"
        )

    prev_cwd = os.getcwd()
    os.chdir(str(root))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))

    try:
        _ensure_graphormer_smpl_assets(root, Path(args.humans_data) if args.humans_data else None)
    except FileNotFoundError as e:
        os.chdir(prev_cwd)
        raise RuntimeError(str(e)) from e

    from src.modeling._smpl import SMPL, Mesh  # type: ignore
    from src.modeling.bert import BertConfig, Graphormer  # type: ignore
    from src.modeling.bert.e2e_body_network import Graphormer_Body_Network  # type: ignore
    from src.modeling.hrnet.config import config as hrnet_config  # type: ignore
    from src.modeling.hrnet.config import update_config as hrnet_update_config  # type: ignore
    from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat  # type: ignore

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    class _Args:
        # Graphormer code calls ``.cuda(self.config.device)``; use int GPU id or "cpu".
        device = int(args.gpu) if device.type == "cuda" else -1
        arch = args.arch
        drop_out = 0.1
        hidden_size = -1
        num_attention_heads = 4
        intermediate_size = -1
        interm_size_scale = 2
        mesh_type = "body"
        num_hidden_layers = 4

    a = _Args()
    smpl = SMPL().to(device)
    mesh_sampler = Mesh()

    input_feat_dim = [int(x) for x in "2051,512,128".split(",")]
    hidden_feat_dim = [int(x) for x in "1024,256,64".split(",")]
    output_feat_dim = input_feat_dim[1:] + [3]
    which_blk_graph = [int(x) for x in "0,0,1".split(",")]
    trans_encoder: List[torch.nn.Module] = []
    config = None
    for i in range(len(output_feat_dim)):
        config = BertConfig.from_pretrained(
            args.model_name_or_path
            if args.model_name_or_path
            else str(root / "src/modeling/bert/bert-base-uncased")
        )
        config.output_attentions = True
        config.output_hidden_states = True
        config.hidden_dropout_prob = a.drop_out
        config.img_feature_dim = input_feat_dim[i]
        config.output_feature_dim = output_feat_dim[i]
        a.hidden_size = hidden_feat_dim[i]
        a.intermediate_size = int(a.hidden_size * a.interm_size_scale)
        if which_blk_graph[i] == 1:
            config.graph_conv = True
        else:
            config.graph_conv = False
        config.mesh_type = a.mesh_type
        for param in ["num_hidden_layers", "hidden_size", "num_attention_heads", "intermediate_size"]:
            arg_v = getattr(a, param)
            if arg_v > 0 and getattr(config, param) != arg_v:
                setattr(config, param, arg_v)
        trans_encoder.append(Graphormer(config=config))

    hrnet_yaml = root / "models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
    hrnet_ckpt = root / "models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
    if args.arch == "hrnet":
        hrnet_yaml = root / "models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
        hrnet_ckpt = root / "models/hrnet/hrnetv2_w40_imagenet_pretrained.pth"
    if not hrnet_yaml.is_file() or not hrnet_ckpt.is_file():
        os.chdir(prev_cwd)
        raise FileNotFoundError(
            f"HRNet config/checkpoint missing under {root}/models/hrnet. "
            "Follow MeshGraphormer docs/DOWNLOAD.md."
        )

    hrnet_update_config(hrnet_config, str(hrnet_yaml))
    backbone = get_cls_net_gridfeat(hrnet_config, pretrained=str(hrnet_ckpt))
    trans_seq = torch.nn.Sequential(*trans_encoder)
    if device.type == "cuda":
        setattr(config, "device", int(args.gpu))
    _model = Graphormer_Body_Network(a, config, backbone, trans_seq, mesh_sampler).to(device)

    if args.checkpoint and Path(args.checkpoint).is_file():
        states = torch.load(args.checkpoint, map_location="cpu")
        if hasattr(states, "state_dict"):
            states = states.state_dict()
        _model.load_state_dict(states, strict=False)

    for m in trans_seq:
        m.config.output_attentions = True
        m.config.output_hidden_states = True
        m.bert.encoder.output_attentions = True
        m.bert.encoder.output_hidden_states = True

    _model.eval()
    try:
        probe = _metric_probe_lazy(args.kti_mode)
    except Exception:
        probe = None
    hooks_att: List[Tuple[int, torch.Tensor]] = []
    hooks_feat: List[Tuple[int, torch.Tensor]] = []

    def reg_hooks() -> List[Any]:
        handles = []
        li = 0
        for enc in trans_seq:
            for bert_layer in enc.bert.encoder.layer:
                attn_self = bert_layer.attention.self

                def _mk_attn(li_local: int):
                    def _h(_m, _inp, out):
                        if isinstance(out, tuple) and len(out) > 1:
                            hooks_att.append((li_local, out[1].detach()))

                    return _h

                def _mk_feat(li_local: int):
                    def _h(_m, _inp, out):
                        if isinstance(out, tuple):
                            hooks_feat.append((li_local, out[0].detach()))
                        else:
                            hooks_feat.append((li_local, out.detach()))

                    return _h

                handles.append(attn_self.register_forward_hook(_mk_attn(li)))
                handles.append(bert_layer.register_forward_hook(_mk_feat(li)))
                li += 1
        return handles

    imgs = torch.randn(args.batch_size, 3, 224, 224, device=device)

    if device.type == "cuda":
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = _model(imgs, smpl, mesh_sampler)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            with torch.no_grad():
                _ = _model(imgs, smpl, mesh_sampler)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / args.iters * 1000.0
    else:
        t0 = time.perf_counter()
        for _ in range(args.iters):
            with torch.no_grad():
                _ = _model(imgs, smpl, mesh_sampler)
        latency_ms = (time.perf_counter() - t0) / args.iters * 1000.0

    hooks_att.clear()
    hooks_feat.clear()
    handles = reg_hooks()
    with torch.no_grad():
        _ = _model(imgs, smpl, mesh_sampler)
    for h in handles:
        h.remove()

    os.chdir(prev_cwd)

    # last forward's hooks
    attn_by_l: Dict[int, List[torch.Tensor]] = {}
    for li, t in hooks_att:
        attn_by_l.setdefault(li, []).append(t)
    feat_by_l: Dict[int, List[torch.Tensor]] = {}
    for li, t in hooks_feat:
        feat_by_l.setdefault(li, []).append(t)

    layer_summaries = []
    for li in sorted(attn_by_l.keys()):
        aw = torch.stack(attn_by_l[li], dim=0).mean(0)
        if probe is not None:
            entropies = [float(probe.calculate_entropy(aw))]
            ktis = [
                float(
                    probe.calculate_kti(
                        aw, probe.smpl_adj, reduce="mean", mode_override="edge_ratio"
                    )
                )
            ]
            kt2 = [
                float(
                    probe.calculate_kti(
                        aw, probe.smpl_adj, reduce="mean", mode_override="dist_corr"
                    )
                )
            ]
            m = {
                "mean_entropy": float(np.mean(entropies)),
                "mean_kti_edge_ratio": float(np.mean(ktis)),
                "mean_kti_dist_corr": float(np.mean(kt2)),
            }
        else:
            m = _aggregate_attn_metrics_local([aw])
        er = None
        if li in feat_by_l:
            ft = torch.stack(feat_by_l[li], dim=0).mean(0)
            er = (
                float(probe.calculate_effective_rank(ft))
                if probe is not None
                else _effective_rank_tokens(ft)
            )
        layer_summaries.append(
            {
                "layer_index": li,
                "mean_entropy": m["mean_entropy"],
                "mean_kti_edge_ratio": m["mean_kti_edge_ratio"],
                "mean_kti_dist_corr": m["mean_kti_dist_corr"],
                "effective_rank_tokens": er,
            }
        )

    params_m = sum(p.numel() for p in _model.parameters()) / 1e6
    return {
        "mode": "graphormer",
        "graphormer_root": str(root),
        "checkpoint": args.checkpoint,
        "device": str(device),
        "params_m": params_m,
        "latency_ms_per_forward": latency_ms,
        "batch_size": args.batch_size,
        "layers": layer_summaries,
        "kti_mode": args.kti_mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["demo", "graphormer", "metro"],
        default="demo",
        help="metro: not wired in this script; use third_party/MeshTransformer + submodule init (same as Graphormer).",
    )
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--kti_mode", default="edge_ratio")
    ap.add_argument("--out", type=Path, default=None)
    # demo
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    # graphormer
    ap.add_argument("--graphormer-root", type=str, default="")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--arch", type=str, default="hrnet-w64")
    ap.add_argument("--model_name_or_path", type=str, default="")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-forwards", type=int, default=1, help="unused; single forward for hooks")
    ap.add_argument(
        "--humans-data",
        type=str,
        default="",
        help="Parent folder with basicModel_neutral_lbs_10_207_0_v1.0.0.pkl (e.g. 4D-Humans/data)",
    )
    args = ap.parse_args()
    if args.mode == "demo":
        out = run_demo(args)
    elif args.mode == "metro":
        raise SystemExit(
            "METRO uses the same BERT submodule layout as MeshGraphormer. "
            "Clone with: bash scripts/sync_microsoft_hmr_baselines.sh\n"
            "Then run tools under third_party/MeshTransformer/metro/tools/ "
            "or extend this script mirroring --mode graphormer for METRO_Body_Network."
        )
    else:
        out = run_graphormer(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.out}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
