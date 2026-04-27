"""Model: cfg → GuidedHMR2Module, optional freeze / attention masking, optional weight load."""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from nvit2_models.guided_hmr2 import GuidedHMR2Module
from nvit.masking_utils import MaskingPatcher
from hmr2.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def build_guided_lightning_module(cfg: DictConfig):
    """
    1) Instantiate GuidedHMR2Module from cfg (architecture + loss heads).
    2) Optionally freeze early backbone layers (FREEZE_DEPTH).
    3) Optionally patch attention masks (MASK_CONFIG).
    """
    model = GuidedHMR2Module(cfg)

    freeze_depth = cfg.get("FREEZE_DEPTH", 0)
    if freeze_depth > 0:
        log.info(f"Surgically freezing first {freeze_depth} layers of backbone (ViT stage)...")
        bb = getattr(model, "nvit_backbone", getattr(model, "backbone", None))

        if bb is not None:
            if hasattr(bb, "surgical_freeze"):
                bb.surgical_freeze(freeze_depth=freeze_depth)
            elif hasattr(bb, "blocks"):
                log.info(f"Applying universal block-level freeze to {type(bb).__name__}")
                for p_name, p in bb.named_parameters():
                    if any(x in p_name for x in ["patch_embed", "pos_embed", "cls_token"]):
                        p.requires_grad = False
                for i, blk in enumerate(bb.blocks):
                    if i < freeze_depth:
                        for p in blk.parameters():
                            p.requires_grad = False
                        log.info(f"❄️  [Manual] Layer {i} FROZEN")
            else:
                log.warning(
                    f"Backbone found but has no '.blocks' or 'surgical_freeze'. Unsafe to freeze depth {freeze_depth}."
                )
        else:
            log.error("CRITICAL: No backbone found in model. Freezing failed.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"Model Parameters: Total={total_params:,} | Trainable={trainable_params:,} "
        f"({trainable_params/total_params:.1%})"
    )

    mask_config = cfg.get("MASK_CONFIG", None)
    if mask_config is not None:
        log.info(f"Applying Attention Masking (Mode: {mask_config.mode}) to backbone...")
        patcher = MaskingPatcher(model, mask_config)
        if hasattr(model, "nvit_backbone"):
            patcher.att_modules = []
            for i, blk in enumerate(model.nvit_backbone.blocks):
                if hasattr(blk, "attn"):
                    patcher.att_modules.append((i, blk.attn))
                elif hasattr(blk, "block") and hasattr(blk.block, "attn"):
                    patcher.att_modules.append((i, blk.block.attn))
        patcher.apply()
        model.mask_patcher = patcher

    return model


def load_finetune_weights(cfg, model) -> None:
    """Load partial weights from FINETUNE_FROM (after Trainer is built; before fit)."""
    if "FINETUNE_FROM" not in cfg or cfg.FINETUNE_FROM is None:
        return

    log.info(f"Finetuning from checkpoint: {cfg.FINETUNE_FROM}")
    ckpt = torch.load(cfg.FINETUNE_FROM, map_location="cpu")
    state_dict = ckpt["state_dict"]

    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    is_hybrid = hasattr(model, "nvit_backbone")

    for k, v in state_dict.items():
        k_new = k
        if is_hybrid and k.startswith("backbone."):
            k_new = k.replace("backbone.", "nvit_backbone.", 1)
            if "blocks." in k_new and ".block." not in k_new:
                parts = k_new.split(".")
                try:
                    idx = parts.index("blocks")
                    if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                        parts.insert(idx + 2, "block")
                        k_new = ".".join(parts)
                except (ValueError, IndexError):
                    pass

        if k_new in model_state_dict:
            if v.shape == model_state_dict[k_new].shape:
                filtered_state_dict[k_new] = v
            else:
                log.warning(
                    f"Shape mismatch for {k_new}: CKPT {v.shape} vs MODEL {model_state_dict[k_new].shape}. Skipping."
                )
        else:
            log.debug(f"Skipping {k} (not in model)")

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    total_model_keys = len(model.state_dict())
    matched_keys = total_model_keys - len(missing)
    match_rate = matched_keys / total_model_keys if total_model_keys > 0 else 0
    log.info(f"Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}, Match Rate: {match_rate:.2%}")
    if match_rate < 0.20:
        raise RuntimeError(
            f"Weight mapping failed: {match_rate:.2%} Match Rate is below 20% threshold! Missing: {len(missing)}."
        )
