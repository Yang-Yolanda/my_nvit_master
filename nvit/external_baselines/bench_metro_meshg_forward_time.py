# -*- coding: utf-8 -*-
"""
Microsoft METRO / Mesh Graphormer 前向计时（224×224 随机图）。

支持两类权重:
1) 整网 torch.save 的 pickle（路径名不含 state_dict 且非 .bin）
2) 官方 release 的 *_state_dict.bin（文件名含 state_dict 或以 .bin 结尾）— 脚本内按官方
   inference 代码组网并 load_state_dict。

用法:
  python bench_metro_meshg_forward_time.py --which metro --code-root .../MeshTransformer \\
    --ckpt models/metro_release/metro_h36m_state_dict.bin --label METRO --gpu 0

依赖: 已完成 scripts/vendor_pytorch_transformers_baselines.sh、HRNet 权重、SMPL pkl、
      J_regressor *.npy（见 scripts/fetch_ms_baseline_aux_data.sh 与 download_ms_azure_pretrained.sh）。
"""
from __future__ import print_function

import argparse
import os
import sys
import time
import types


def _is_state_dict_checkpoint(ckpt_path):
    b = os.path.basename(ckpt_path).lower()
    return "state_dict" in b or b.endswith(".bin")


def _build_metro_from_state_dict(device, smpl, mesh_sampler, ckpt_path, gpu_index):
    import torch
    import torchvision.models as models  # noqa: F401  # resnet path in upstream

    from metro.modeling.bert import BertConfig, METRO
    from metro.modeling.bert import METRO_Body_Network as METRO_Network
    from metro.modeling.hrnet.config import config as hrnet_config
    from metro.modeling.hrnet.config import update_config as hrnet_update_config
    from metro.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net

    args = types.SimpleNamespace(
        arch="hrnet-w64",
        num_hidden_layers=4,
        hidden_size=-1,
        num_attention_heads=4,
        intermediate_size=-1,
        input_feat_dim="2051,512,128",
        hidden_feat_dim="1024,256,128",
        legacy_setting=True,
        model_name_or_path="metro/modeling/bert/bert-base-uncased/",
        device=device,
    )

    trans_encoder = []
    input_feat_dim = [int(x) for x in args.input_feat_dim.split(",")]
    hidden_feat_dim = [int(x) for x in args.hidden_feat_dim.split(",")]
    output_feat_dim = input_feat_dim[1:] + [3]
    config = None
    for i in range(len(output_feat_dim)):
        config = BertConfig.from_pretrained(args.model_name_or_path)
        config.output_attentions = False
        config.img_feature_dim = input_feat_dim[i]
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim[i]
        if args.legacy_setting:
            args.intermediate_size = -1
        else:
            args.intermediate_size = int(args.hidden_size * 4)
        for param in (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "intermediate_size",
        ):
            arg_v = getattr(args, param)
            cfg_v = getattr(config, param)
            if arg_v > 0 and arg_v != cfg_v:
                setattr(config, param, arg_v)
        assert config.hidden_size % config.num_attention_heads == 0
        trans_encoder.append(METRO(config=config))

    hrnet_yaml = "models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
    hrnet_checkpoint = "models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
    hrnet_update_config(hrnet_config, hrnet_yaml)
    backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
    trans_encoder = torch.nn.Sequential(*trans_encoder)
    net = METRO_Network(args, config, backbone, trans_encoder, mesh_sampler)

    cpu = torch.device("cpu")
    states = torch.load(ckpt_path, map_location=cpu)
    net.load_state_dict(states, strict=False)
    del states

    net.to(device)
    if device.type == "cuda":
        net.config.device = int(gpu_index)
    return net


def _build_graphormer_from_state_dict(device, smpl, mesh_sampler, ckpt_path, gpu_index):
    import gc

    import torch
    import torchvision.models as models

    from src.modeling.bert import BertConfig, Graphormer
    from src.modeling.bert.e2e_body_network import Graphormer_Body_Network as Graphormer_Network
    from src.modeling.hrnet.config import config as hrnet_config
    from src.modeling.hrnet.config import update_config as hrnet_update_config
    from src.modeling.hrnet.hrnet_cls_net_gridfeat import get_cls_net_gridfeat

    args = types.SimpleNamespace(
        arch="hrnet-w64",
        num_hidden_layers=4,
        hidden_size=-1,
        num_attention_heads=4,
        intermediate_size=-1,
        input_feat_dim="2051,512,128",
        hidden_feat_dim="1024,256,64",
        which_gcn="0,0,1",
        mesh_type="body",
        interm_size_scale=2,
        model_name_or_path="src/modeling/bert/bert-base-uncased/",
        config_name="",
        device=device,
    )

    trans_encoder = []
    input_feat_dim = [int(x) for x in args.input_feat_dim.split(",")]
    hidden_feat_dim = [int(x) for x in args.hidden_feat_dim.split(",")]
    output_feat_dim = input_feat_dim[1:] + [3]
    which_blk_graph = [int(x) for x in args.which_gcn.split(",")]
    config = None
    for i in range(len(output_feat_dim)):
        config = BertConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path
        )
        config.output_attentions = False
        config.img_feature_dim = input_feat_dim[i]
        config.output_feature_dim = output_feat_dim[i]
        args.hidden_size = hidden_feat_dim[i]
        args.intermediate_size = int(args.hidden_size * args.interm_size_scale)
        if which_blk_graph[i] == 1:
            config.graph_conv = True
        else:
            config.graph_conv = False
        config.mesh_type = args.mesh_type
        for param in (
            "num_hidden_layers",
            "hidden_size",
            "num_attention_heads",
            "intermediate_size",
        ):
            arg_v = getattr(args, param)
            cfg_v = getattr(config, param)
            if arg_v > 0 and arg_v != cfg_v:
                setattr(config, param, arg_v)
        assert config.hidden_size % config.num_attention_heads == 0
        trans_encoder.append(Graphormer(config=config))

    hrnet_yaml = "models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
    hrnet_checkpoint = "models/hrnet/hrnetv2_w64_imagenet_pretrained.pth"
    hrnet_update_config(hrnet_config, hrnet_yaml)
    backbone = get_cls_net_gridfeat(hrnet_config, pretrained=hrnet_checkpoint)
    trans_encoder = torch.nn.Sequential(*trans_encoder)
    net = Graphormer_Network(args, config, backbone, trans_encoder, mesh_sampler)

    states = torch.load(ckpt_path, map_location="cpu")
    for k, v in list(states.items()):
        states[k] = v.cpu()
    net.load_state_dict(states, strict=False)
    del states
    gc.collect()

    net.to(device)
    if device.type == "cuda":
        net.config.device = int(gpu_index)
    return net


def _load_network(which, ckpt_path, device, smpl, mesh_sampler, gpu_index):
    import torch

    if not _is_state_dict_checkpoint(ckpt_path):
        net = torch.load(ckpt_path, map_location=device)
        return net
    if which == "metro":
        return _build_metro_from_state_dict(device, smpl, mesh_sampler, ckpt_path, gpu_index)
    return _build_graphormer_from_state_dict(
        device, smpl, mesh_sampler, ckpt_path, gpu_index
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("metro", "meshg"), required=True)
    ap.add_argument("--code-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    root = os.path.abspath(args.code_root)
    if not os.path.isdir(root):
        print("ERROR: code-root not dir: %s" % root, file=sys.stderr)
        sys.exit(2)
    ckpt_abs = os.path.abspath(args.ckpt)
    if not os.path.isfile(ckpt_abs):
        print("ERROR: ckpt not found: %s" % ckpt_abs, file=sys.stderr)
        sys.exit(2)

    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    if args.which == "meshg" and os.path.join(root, "src") not in sys.path:
        sys.path.insert(0, os.path.join(root, "src"))

    import torch

    if args.which == "metro":
        from metro.modeling._smpl import SMPL, Mesh
    else:
        from src.modeling._smpl import SMPL, Mesh

    device = torch.device(
        "cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu"
    )
    smpl = SMPL().to(device)
    mesh_sampler = Mesh()
    try:
        net = _load_network(
            args.which, ckpt_abs, device, smpl, mesh_sampler, int(args.gpu)
        )
    except Exception as e:
        print("ERROR: load network failed: %s" % e, file=sys.stderr)
        sys.exit(3)
    net.eval()
    smpl.eval()

    x = torch.randn(1, 3, 224, 224, device=device)

    def _step():
        with torch.no_grad():
            net(x, smpl, mesh_sampler)

    for _ in range(args.warmup):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    total = t1 - t0
    ms_step = 1000.0 * total / args.iters
    ms_img = ms_step
    ips = 1000.0 / ms_img if ms_img > 0 else 0.0

    print(
        "%s,%s,1,%d,%.6f,%.6f,%.6f"
        % (args.label, ckpt_abs, args.iters, ms_step, ms_img, ips)
    )


if __name__ == "__main__":
    main()
