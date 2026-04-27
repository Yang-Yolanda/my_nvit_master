"""
HMR2 ViT 剪枝权重（.pth 全量 state_dict）：按 backbone 各层 mlp.fc1 形状重载 Mlp，再 strict 装载。

与 OSS 上 `hmr2_mid_heavy_model_config.yaml` 描述的 mid_heavy 结构一致时，
可配合 4D-Humans 官方多轮 HMR2 Lightning 断点作为 `ref_ckpt`（同 cfg / 同 smpl_head）。
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

import torch
import torch.nn as nn

from .model_io import load_model_from_ckpt


def mlp_hiddens_from_pth_state_dict(
    state_dict: dict, backbone_prefix: str = "backbone"
) -> List[int]:
    """从 state_dict 解析每块 ViT Block 的 MLP 隐层维（fc1 输出维）。"""
    pat = re.compile(
        re.escape(backbone_prefix) + r"\.blocks\.(\d+)\.mlp\.fc1\.weight"
    )
    found: dict[int, int] = {}
    for k, v in state_dict.items():
        m = pat.match(k)
        if m:
            found[int(m.group(1))] = int(v.shape[0])
    if not found:
        raise ValueError("state_dict 中无 backbone*.blocks.*.mlp.fc1.weight，不是预期 HMR2 剪枝格式")
    n = max(found) + 1
    for i in range(n):
        if i not in found:
            raise ValueError(f"缺少 block {i} 的 mlp.fc1")
    return [found[i] for i in range(n)]


def load_hmr2_from_pruned_pth(
    pth_path: str,
    ref_hmr2_ckpt: str,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    用参考 Lightning 断点实例化 HMR2，再按 pth 中 MLP 宽度替换各层 Mlp 并 load_state_dict(strict=True)。
    pth 须为与参考图同构的**完整** HMR2 权重（含 smpl_head 等），仅 ViT MLP 通道有剪枝。
    """
    print(
        f"[hmr2_pruned] (1/4) torch.load 剪枝 pth（体积大，可能数分钟无输出）: {pth_path}",
        flush=True,
    )
    sd = torch.load(pth_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise ValueError("pth 需为可索引的 state_dict 字典")
    mlp_h = mlp_hiddens_from_pth_state_dict(sd)
    print(
        f"[hmr2_pruned] (2/4) 已读 pth，共 {len(mlp_h)} 个 block 的 MLP 隐层已解析",
        flush=True,
    )

    print(
        f"[hmr2_pruned] (3/4) 加载 ref Lightning 以建网: {ref_hmr2_ckpt}",
        flush=True,
    )
    model = load_model_from_ckpt(ref_hmr2_ckpt, device="cpu")
    n_blocks = len(model.backbone.blocks)
    if n_blocks != len(mlp_h):
        raise ValueError(f"参数量: ref backbone depth={n_blocks} vs pth mlp 层数={len(mlp_h)}")

    from hmr2.models.backbones.vit import Mlp

    D = int(model.backbone.embed_dim)
    for i, h in enumerate(mlp_h):
        model.backbone.blocks[i].mlp = Mlp(
            in_features=D,
            hidden_features=h,
            out_features=D,
            act_layer=nn.GELU,
            drop=0.0,
        )
    print("[hmr2_pruned] 已按 pth 重配各层 Mlp，开始 strict load_state_dict …", flush=True)
    m = model.load_state_dict(sd, strict=True)
    if m.missing_keys or m.unexpected_keys:
        raise RuntimeError(f"strict 装载仍异常: missing={m.missing_keys!r} unexp={m.unexpected_keys!r}")
    model.eval()
    model.to(device)
    print(
        f"[hmr2_pruned] (4/4) 完成，已放到 {device}，可开始逐数据集评测",
        flush=True,
    )
    return model


def load_model_hmr2_pth_or_ckpt(
    ckpt_path: str, device: str, pth_ref_ckpt: Optional[str]
) -> torch.nn.Module:
    """给评测/bench 用：若提供 pth_ref 且为剪枝 pth 则走重载 MLP 分支，否则走 Lightning 装载。"""
    p = str(ckpt_path)
    if pth_ref_ckpt and is_hmr2_pruned_pth_file(p):
        return load_hmr2_from_pruned_pth(p, pth_ref_ckpt, device=device)
    return load_model_from_ckpt(ckpt_path, device=device)


def looks_like_hmr2_pth_path(path: str) -> bool:
    p = (path or "").lower()
    return p.endswith((".pth", ".pt")) and "guided" not in p and "mamba" not in p


def is_hmr2_pruned_pth_file(path: str) -> bool:
    """
    判定是否应按剪枝 MLP 分支装载。优先用文件名（免重复读大文件）；
    未命中时检查 state_dict 第 0 层 MLP 是否非默认 4×embed 宽（5120@ViT-H）。
    """
    b = os.path.basename((path or "").lower())
    if b.endswith((".pth", ".pt")) and (
        "mid_heavy" in b or "pruned" in b
    ):
        return True
    try:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(sd, dict) or "backbone.blocks.0.mlp.fc1.weight" not in sd:
        return False
    t = sd["backbone.blocks.0.mlp.fc1.weight"]
    in_dim = t.shape[1]
    out_dim = t.shape[0]
    default_mlp = int(4 * in_dim)
    return int(out_dim) != default_mlp
