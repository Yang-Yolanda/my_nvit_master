import os
import sys
import time
sys.path.append("/home/yangz/4D-Humans")
sys.path.append("/home/yangz/NViT-master")
import glob
import json
import math
import subprocess
import argparse
from pathlib import Path

EXPECTED_GRID = [
  ("G0","T0"),
  ("G1","T0"),
  ("G2","T0"), 
  ("G3","T0"), ("G3","T1"),
]
DATASET="3dpw_test"
IMG=256
SEED=0

from nvit.eval.standard_eval_wrapper import infer_mm

def get_sha1(filepath):
    import hashlib
    with open(filepath, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:8]

def read_json(p):
    with open(p,"r") as f: return json.load(f)

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def resolve_latest(exp_dir):
    latest = Path(exp_dir) / "latest"
    if latest.exists():
        return latest.resolve()
    runs = sorted((Path(exp_dir)/"runs").glob("*"))
    return runs[-1] if runs else None

CSV_COLS = [
    "Model", "TrainPolicy", "MPJPE", "PA-MPJPE", "Params", "FLOPs", "MB",
    "Latency", "Throughput", "TrainableParams", "PeakVRAM", "GPUHours",
    "Status", "ErrorShort", "ExpID", "RunID", "AuditOK", "dataset_sha1_8"
]

def write_csv(path, rows):
    import csv
    if not rows: return
    with open(path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction='ignore')
        dict_writer.writeheader()
        dict_writer.writerows(rows)

COLS = ["Model","TrainPolicy","MPJPE","PA-MPJPE","Params","FLOPs","MB",
        "Latency","Throughput","TrainableParams","PeakVRAM","GPUHours"]

def latex_escape(s: str) -> str:
    if s is None: return ""
    return (s.replace("\\","\\textbackslash{}")
             .replace("_","\\_").replace("%","\\%").replace("&","\\&")
             .replace("#","\\#").replace("{","\\{").replace("}","\\}")
             .replace("^","\\^{}").replace("~","\\~{}"))

def fmt_num(x, digits=2):
    if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))):
        return "\\textbf{--}"
    if isinstance(x, int):
        return f"{x}"
    return f"{x:.{digits}f}"

def fmt_fail_cell():
    return "\\textbf{--}"

def write_latex_table(path, rows, include_fail_rows=True):
    lines = []
    lines.append("\\begin{tabular}{llrrrrrrrrrr}")
    lines.append("\\toprule")
    lines.append(" & ".join(COLS) + " \\\\")
    lines.append("\\midrule")

    for r in rows:
        status = r.get("Status","FAIL")
        if (not include_fail_rows) and status != "OK":
            continue

        if status == "OK":
            gpu_hours = fmt_num(r.get("GPUHours"), digits=2)
            cells = [
                r["Model"], r["TrainPolicy"],
                fmt_num(r.get("MPJPE"),2), fmt_num(r.get("PA-MPJPE"),2),
                fmt_num(r.get("Params"),0), fmt_num(r.get("FLOPs"),0),
                fmt_num(r.get("MB"),1), fmt_num(r.get("Latency"),2),
                fmt_num(r.get("Throughput"),1),
                fmt_num(r.get("TrainableParams"),0),
                fmt_num(r.get("PeakVRAM"),2),
                gpu_hours,
            ]
        else:
            note = latex_escape(r.get("ErrorShort","FAIL"))
            cells = [
                r["Model"], r["TrainPolicy"],
                fmt_fail_cell(), fmt_fail_cell(), fmt_fail_cell(), fmt_fail_cell(),
                fmt_fail_cell(), fmt_fail_cell(), fmt_fail_cell(), fmt_fail_cell(),
                fmt_fail_cell(),
                f"\\textbf{{--}}\\,{{\\scriptsize({note})}}",
            ]

        cells[0] = latex_escape(cells[0])
        cells[1] = latex_escape(cells[1])
        lines.append(" & ".join(cells) + " \\\\")
        
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    with open(path, "w") as f:
        f.write("\n".join(lines))
        
import re
def parse_and_validate_exp_id(exp_id: str, meta: dict) -> dict:
    pattern = r"^(G[0-3])__(T[01])__([A-Za-z0-9_]+)__img(\d+)__seed(\d+)$"
    m = re.match(pattern, exp_id)
    if not m:
        raise ValueError(f"[Audit Failed] Bad exp_id: {exp_id}")
    p_model, p_policy, p_dataset, p_img, p_seed = m.groups()
    if p_dataset not in {"3dpw_test"}:
        raise ValueError(f"[Audit Failed] Unknown dataset token in exp_id: {p_dataset}")

    expected = {
        "model_group": p_model,
        "train_policy": p_policy,
        "dataset": p_dataset,
        "image_size": int(p_img),
        "seed": int(p_seed),
    }

    mismatches = []
    for k, v in expected.items():
        if k not in meta: mismatches.append(f"{k} missing in meta.json")
        elif meta[k] != v: mismatches.append(f"{k} (Dir '{v}' vs Meta '{meta[k]}')")

    if mismatches:
        raise ValueError(f"[Audit Failed] exp_id != meta.json for {exp_id}. " + "; ".join(mismatches))
    return expected

def collect(out_dir):
    logs_root = Path(out_dir)/"logs"
    results_root = Path(out_dir)/"results"
    results_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for G,T in EXPECTED_GRID:
        exp_id = f"{G}__{T}__{DATASET}__img{IMG}__seed{SEED}"
        exp_dir = logs_root/exp_id
        row = {"Model": G, "TrainPolicy": T, "Status":"FAIL", "ErrorShort":"MISSING_LOG", "ExpID": exp_id, "AuditOK": 0}

        run_dir = resolve_latest(exp_dir) if exp_dir.exists() else None
        if run_dir is None:
            rows.append(row); continue

        try:
            meta = read_json(run_dir/"meta.json")
            _ = parse_and_validate_exp_id(exp_id, meta)
            row["AuditOK"] = 1

            status = read_json(run_dir/"status.json")
            row["Status"] = status.get("status","FAIL")
            row["ErrorShort"] = status.get("error_short","")
            row["RunID"] = status.get("run_id","")

            if row["Status"] == "OK":
                metrics = read_json(run_dir/"metrics.json")
                row.update({
                  "MPJPE": metrics.get("mpjpe_mm"),
                  "PA-MPJPE": metrics.get("pa_mpjpe_mm"),
                })
                
                if (run_dir/"prof_infer.json").exists():
                    pi = read_json(run_dir/"prof_infer.json")
                    pt = read_json(run_dir/"prof_train.json") if (run_dir/"prof_train.json").exists() else {}
                    row.update({
                      "Params": pi.get("params"),
                      "FLOPs": pi.get("flops"),
                      "MB": pi.get("model_mb"),
                      "Latency": pi.get("latency_ms_bs1"),
                      "Throughput": pi.get("throughput_fps_bs32"),
                      "TrainableParams": pt.get("trainable_params"),
                      "PeakVRAM": pt.get("peak_vram_gb"),
                      "GPUHours": pt.get("gpu_hours"),
                    })
                else:
                    if not row["ErrorShort"]: row["ErrorShort"] = "PROFILE_FAIL"
        except Exception as e:
            if row["AuditOK"] == 0:
                row["Status"] = "FAIL"
                row["ErrorShort"] = "AUDIT_MISMATCH"
            else:
                row["Status"] = "FAIL"
                row["ErrorShort"] = "STATUS_MISSING"
                
            with open(results_root/"collect_errors.log", "a") as f:
                f.write(f"\\nError checking {exp_id}:\\n{str(e)}\\n")
        rows.append(row)

    write_csv(results_root/"efficiency_table.csv", rows)
    write_latex_table(results_root/"efficiency_table.tex", rows, include_fail_rows=True)
    print(f"Collected and generated results in {results_root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sanity", "grid", "collect", "dry_collect_test"])
    ap.add_argument("--dataset", default="3dpw_test")
    ap.add_argument("--data_root", default="")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--g0_ckpt", default="")
    ap.add_argument("--g3_ckpt", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--throughput_bs", type=int, default=32)
    ap.add_argument("--do_G0_T0", action="store_true")
    ap.add_argument("--do_G1_T0", action="store_true")
    ap.add_argument("--do_G2_T0", action="store_true")
    ap.add_argument("--do_G3_T0", action="store_true")
    ap.add_argument("--do_G3_T1", action="store_true")
    args = ap.parse_args()

    if args.mode in ["collect", "dry_collect_test"]:
        collect(args.out_dir)
        if args.mode == "dry_collect_test":
            print(f"Dry run collect complete. Output at: {args.out_dir}/results")
        return

    def run_eval_job(log_dir, ckpt):
        cmd = [
            "python", "-m", "nvit.eval.standard_eval_wrapper",
            "--log_dir", str(log_dir),
            "--dataset", args.dataset,
            "--data_root", args.data_root,
            "--ckpt", ckpt
        ]
        return subprocess.run(cmd)

    def run_prof_job(log_dir, ckpt):
        cmd = [
            "python", "-m", "nvit.prof.profile_infer",
            "--log_dir", str(log_dir),
            "--dataset", args.dataset,
            "--data_root", args.data_root,
            "--ckpt", ckpt,
            "--throughput_bs", str(args.throughput_bs)
        ]
        return subprocess.run(cmd)
        
    def write_run_meta(exp_id, G, T, ckpt):
        import uuid
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:8]
        exp_dir = Path(args.out_dir)/"logs"/exp_id
        run_dir = exp_dir/"runs"/run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        latest = exp_dir/"latest"
        if latest.exists() or latest.is_symlink(): latest.unlink()
        latest.symlink_to(f"runs/{run_id}")
        
        dataset_file = os.path.join(args.data_root, "3dpw_test.npz") if not args.data_root.endswith(".npz") else args.data_root
        d_sha = get_sha1(dataset_file) if os.path.exists(dataset_file) else "unknown"
        
        meta = {
          "model_group": G,
          "train_policy": T,
          "dataset": args.dataset,
          "image_size": args.image_size,
          "seed": 0,
          "alignment": "ON",
          "unit": "mm",
          "ckpt_path": ckpt,
          "dataset_file": dataset_file,
          "dataset_sha1_8": d_sha
        }
        write_json(run_dir/"meta.json", meta)
        return run_dir

    if args.mode == "sanity":
        exp_id = f"G0__T0__{args.dataset}__img{args.image_size}__seed0"
        run_dir = write_run_meta(exp_id, "G0", "T0", args.g0_ckpt)
        res_e = run_eval_job(run_dir, args.g0_ckpt)
        res_p = run_prof_job(run_dir, args.g0_ckpt)
        if res_e.returncode != 0 or res_p.returncode != 0:
            print("Sanity CHECK FAILED! Terminating.")
            exit(1)
        collect(args.out_dir)

    elif args.mode == "grid":
        all_jobs = []
        if args.do_G0_T0: all_jobs.append(("G0", "T0", args.g0_ckpt))
        if args.do_G1_T0: all_jobs.append(("G1", "T0", args.g0_ckpt)) 
        if args.do_G2_T0: all_jobs.append(("G2", "T0", args.g0_ckpt))
        if args.do_G3_T0: all_jobs.append(("G3", "T0", args.g3_ckpt))
        if args.do_G3_T1: all_jobs.append(("G3", "T1", args.g3_ckpt))
        
        for g, t, ckpt in all_jobs:
            exp_id = f"{g}__{t}__{args.dataset}__img{args.image_size}__seed0"
            run_dir = write_run_meta(exp_id, g, t, ckpt)
            run_eval_job(run_dir, ckpt)
            run_prof_job(run_dir, ckpt)
        
        collect(args.out_dir)

if __name__ == "__main__":
    main()
