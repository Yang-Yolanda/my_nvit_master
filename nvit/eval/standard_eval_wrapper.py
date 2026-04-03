import os
import sys
sys.path.append("/home/yangz/4D-Humans")
sys.path.append("/home/yangz/NViT-master")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import json
import traceback
import argparse
import time
import platform
import math
from pathlib import Path
import torch

from nvit.skills.evaluate_model.standard_eval import EvaluatorSkill
from types import SimpleNamespace

EXPECTED_IMAGE_SIZE = 256
DATASET_DEFAULT_ALIGNMENT = {
    "3dpw_test": True,
    "3DPW-TEST": True
}
MAX_REASONABLE_MPJPE_MM = 2000.0

def write_text(path, s):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: f.write(s)

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def infer_mm(raw_value):
    if raw_value is None:
        return None, "unknown"
    v = float(raw_value)
    if 0 <= v < 10:
        return v * 1000.0, "m->mm"
    return v, "assume_mm"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--dataset", default="3dpw_test")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--use_mean_alignment", type=int, default=None)
    ap.add_argument("--allow_alignment_off", action="store_true")
    args = ap.parse_args()

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)

    cmd = " ".join(os.sys.argv)
    write_text(os.path.join(log_dir, "cmd.txt"), cmd)

    status = {"status": "FAIL", "phase": "eval", "error": None, "error_short": "", "cmd": cmd}
    t0 = time.time()

    try:
        skill = EvaluatorSkill(gpu="0")
        
        # 1. Assert Image Size
        model = skill.load_model(args.ckpt)
        actual_img_size = int(model.cfg.MODEL.IMAGE_SIZE)
        if actual_img_size != EXPECTED_IMAGE_SIZE:
            raise AssertionError(f"[Contract Violation] Expected IMAGE_SIZE={EXPECTED_IMAGE_SIZE}, got {actual_img_size}")

        # 2. Alignment Logic
        if args.use_mean_alignment is None:
            use_align = DATASET_DEFAULT_ALIGNMENT.get(args.dataset.lower(), False)
        else:
            use_align = bool(args.use_mean_alignment)

        if "3dpw" in args.dataset.lower() and (not use_align) and (not args.allow_alignment_off):
            raise AssertionError("[Contract Violation] 3DPW must use mean alignment unless --allow_alignment_off")

        # 3. Call Underlying Script safely via Mock Args
        ds_token = "3DPW-TEST" if "3dpw" in args.dataset.lower() else args.dataset
        eval_json_out = os.path.join(log_dir, "eval_raw_out.json")
        mock_args = SimpleNamespace(
            ckpt=args.ckpt,
            dataset=ds_token,
            gpu="0",
            batch_size=32,
            num_workers=0, # Must be 0 for anti-hang
            limit_batches=None,
            skip_errors=False,
            diagnostics=False,
            dense_gt=True,
            output=eval_json_out,
            data_dir=os.path.dirname(args.data_root) if os.path.isfile(args.data_root) else args.data_root,
            use_mean_alignment=use_align
        )
        
        skill.run_eval(mock_args)
        
        # 4. Parse Output JSON
        with open(eval_json_out, "r") as f:
            eval_data = json.load(f)
        
        raw = eval_data.get("results", {}).get(ds_token, {})
        
        mode_mpjpe = raw.get("mode_mpjpe")
        mode_re = raw.get("mode_re")

        if mode_mpjpe is None or mode_re is None:
            raise ValueError("[Contract Violation] METRIC_KEY_MISSING")

        mpjpe_mm = float(mode_mpjpe)
        pa_mpjpe_mm = float(mode_re)

        # 5. Assert Guardrails
        for name, v in [("mpjpe_mm", mpjpe_mm), ("pa_mpjpe_mm", pa_mpjpe_mm)]:
            if (v < 0) or (v > MAX_REASONABLE_MPJPE_MM):
                raise AssertionError(f"[Contract Violation] {name} out of range: {v} mm. Possible alignment or unit issue.")

        metrics = {
            "dataset": args.dataset,
            "alignment_requested": bool(use_align),
            "alignment_effective": bool(use_align),
            "unit": "mm",
            "unit_source": "hmr2_evaluator_returns_mm",
            "mpjpe_mm": mpjpe_mm,
            "pa_mpjpe_mm": pa_mpjpe_mm,
            "raw": raw,
            "image_size": actual_img_size,
            "model_class": model.__class__.__name__
        }
        write_json(os.path.join(log_dir, "metrics.json"), metrics)
        status["status"] = "OK"

    except Exception as e:
        status["error"] = "".join(traceback.format_exc())
        write_text(os.path.join(log_dir, "stderr.log"), status["error"])
        
        err = status["error"]
        if "FileNotFoundError" in err or "No such file or directory" in err:
            if "best.ckpt" in err or ".ckpt" in err:
                status["error_short"] = "CKPT_NOT_FOUND"
            elif "3dpw_test.npz" in err or ".npz" in err:
                status["error_short"] = "DATASET_NOT_FOUND"
            else:
                status["error_short"] = "FILE_NOT_FOUND"
        elif "CUDA out of memory" in err:
            status["error_short"] = "OOM"
        elif "[Contract Violation]" in err:
            status["error_short"] = "CONTRACT_VIOLATION"
        else:
            status["error_short"] = "RUNTIME_ERROR"

    status["wall_time_s"] = time.time() - t0
    status["env"] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "None",
    }
    write_json(os.path.join(log_dir, "status.json"), status)

if __name__ == "__main__":
    main()
