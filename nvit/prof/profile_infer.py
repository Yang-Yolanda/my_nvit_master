import argparse
import json
import os
import time
import torch
import hashlib
import sys

# Append standard paths
sys.path.append('/home/yangz/4D-Humans')
sys.path.append('/home/yangz/NViT-master')

from nvit2_models.guided_hmr2 import GuidedHMR2Module
from hmr2.datasets import create_dataset
from hmr2.configs import dataset_eval_config
from hmr2.utils import recursive_to
from yacs.config import CfgNode as CN

EXPECTED_IMAGE_SIZE = 256
REQUIRED_KEYS = {"img", "box_center", "box_size", "img_size"}

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="3dpw_test")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--throughput_bs", type=int, default=32)
    args = ap.parse_args()

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    print(f"Loading {args.ckpt} for profiling...")
    from nvit.utils.model_io import load_model_from_ckpt
    model = load_model_from_ckpt(args.ckpt, device="cuda")
    model.cuda().eval()

    actual_img_size = int(model.cfg.MODEL.IMAGE_SIZE)
    assert actual_img_size == EXPECTED_IMAGE_SIZE, f"IMG_SIZE mismatch: {actual_img_size}"

    # Load real batch topology
    cfg_idx = "3DPW-TEST" if "3dpw" in args.dataset.lower() else args.dataset
    cfg_eval = dataset_eval_config()[cfg_idx]
    cfg_eval.defrost()
    cfg_eval.DATASET_FILE = os.path.join(args.data_root, os.path.basename(cfg_eval.DATASET_FILE)) if not os.path.isfile(args.data_root) else args.data_root
    cfg_eval.freeze()

    dummy_cfg = CN()
    dummy_cfg.MODEL = CN()
    dummy_cfg.MODEL.IMAGE_SIZE = EXPECTED_IMAGE_SIZE
    dummy_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
    dummy_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
    dummy_cfg.SMPL = CN()
    dummy_cfg.SMPL.NUM_BODY_JOINTS = 23
    dummy_cfg.DATASETS = CN()
    dummy_cfg.DATASETS.CONFIG = CN()

    ds = create_dataset(dummy_cfg, cfg_eval, train=False)

    def get_real_batch(bs):
        loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, drop_last=False, pin_memory=False)
        batch = next(iter(loader))
        batch = recursive_to(batch, "cuda")
        return batch

    x_lat = get_real_batch(1)
    batch_keys_sorted = sorted(list(x_lat.keys()))
    missing = REQUIRED_KEYS - set(batch_keys_sorted)
    if missing:
        raise AssertionError(f"[Contract Violation] profile batch missing keys: {sorted(missing)}")
    required_keys_ok = True

    def measure_latency(x, warmup, iters):
        for _ in range(warmup): _ = model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters): _ = model(x)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    print("Measuring Latency (BS=1)...")
    lat_ms = measure_latency(x_lat, warmup=10, iters=50)

    print("Measuring Throughput...")
    x_thr = get_real_batch(args.throughput_bs)
    thr_lat_ms = measure_latency(x_thr, warmup=5, iters=20)
    fps = (args.throughput_bs) / (thr_lat_ms / 1000.0)

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_mb = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 * 1024)

    prof = {
        "profile_mode": "real_batch_topology",
        "phase": "profile_infer",
        "batch_keys_sorted": batch_keys_sorted,
        "required_keys_ok": required_keys_ok,
        "dataset": args.dataset,
        "image_size": EXPECTED_IMAGE_SIZE,
        "latency_ms_bs1": lat_ms,
        "throughput_fps_bs{}".format(args.throughput_bs): fps,
        "params": params,
        "model_mb": model_mb,
        "flops": None,
        "trainable_params_current": trainable_params,
        "ckpt": args.ckpt,
        "model_class": model.__class__.__name__
    }
    write_json(os.path.join(log_dir, "prof_infer.json"), prof)
    print(f"Profiling complete. Saved to {log_dir}")

if __name__ == "__main__":
    main()
