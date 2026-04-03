import os
import time
import subprocess
import glob
import shutil
import json

# ==========================================
# SUPER-SAFE ISOLATED EVALUATION DAEMON
# ==========================================

# 1. Point directly to the Samsung SSD Vault
VAULT_DIR = "/mnt/ssd_samsung_1/home/nkd/yangz_data/nvit_output/Periodic_Checkpoints"
BEST_CKPT_PATH = os.path.join(VAULT_DIR, "best_phase2.ckpt")
BEST_METRICS_PATH = os.path.join(VAULT_DIR, "best_metrics.json")
EVAL_LOG = os.path.join(VAULT_DIR, "auto_eval.log")

# State
evaluated_ckpts = set()
current_best_mpjpe = float('inf')

def get_checkpoints():
    # Only pick up the periodic epoch-step checkpoints
    ckpts = glob.glob(os.path.join(VAULT_DIR, "nvit-*.ckpt"))
    ckpts = [c for c in ckpts if "best_phase2" not in c and "eval_temp" not in c]
    return sorted(ckpts, key=os.path.getmtime)

def evaluate_ckpt(ckpt_path):
    global current_best_mpjpe
    print(f"\n[{time.ctime()}] 🔍 Discovered new checkpoint: {os.path.basename(ckpt_path)}. Evaluating on GPU 2...")
    
    # Safely duplicate it in case PyTorch Lightning overwrites / deletes it during max-save cap
    safe_copy = os.path.join(VAULT_DIR, "eval_temp.ckpt")
    shutil.copy2(ckpt_path, safe_copy)
    
    cmd = [
        "/home/yangz/.conda/envs/4D-humans/bin/python", 
        "nvit/skills/evaluate_model/standard_eval.py", 
        "--ckpt", safe_copy, 
        "--dataset", "3DPW-TEST", 
        "--use_mean_alignment"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":/home/yangz/NViT-master:/home/yangz/4D-Humans"
    # [CRITICAL HARD ISOLATION]: Exclusively lock to GPU 2
    env["CUDA_VISIBLE_DEVICES"] = "2"
    
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    with open(EVAL_LOG, "a") as f:
        f.write(f"\n\n{'='*40}\n[{time.ctime()}] Evaluating: {os.path.basename(ckpt_path)}\n{'='*40}\n")
        f.write(result.stdout)
        
    # Parse 3DPW MPJPE
    mpjpe_3dpw = None
    for line in result.stdout.split('\n'):
        if "3DPW-TEST" in line and "MPJPE:" in line:
            try:
                mpjpe_3dpw = float(line.split("MPJPE=")[1].split(",")[0])
            except: pass

    if mpjpe_3dpw is not None:
        print(f"👉 3DPW Score for {os.path.basename(ckpt_path)}: {mpjpe_3dpw} mm")
        
        if mpjpe_3dpw < current_best_mpjpe or not os.path.exists(BEST_CKPT_PATH):
            print(f"🏆 NEW BEST ACHIEVED! {mpjpe_3dpw} < {current_best_mpjpe}. Promoting to best_phase2.ckpt...")
            current_best_mpjpe = mpjpe_3dpw
            shutil.copy2(safe_copy, BEST_CKPT_PATH)
            
            with open(BEST_METRICS_PATH, "w") as f:
                json.dump({"best_mpjpe_3dpw": current_best_mpjpe, "ckpt": os.path.basename(ckpt_path), "timestamp": time.ctime()}, f)
    else:
        print("⚠️ Failed to parse valid MPJPE. The script might have crashed.")

def main():
    global current_best_mpjpe
    print("🚀 Starting GPU-2 Isolated Batch Evaluator...")
    print(f"📁 Watching Directory: {VAULT_DIR}")
    
    # Reload historical best so we don't start from scratch
    if os.path.exists(BEST_METRICS_PATH):
        try:
            with open(BEST_METRICS_PATH, "r") as f:
                data = json.load(f)
                current_best_mpjpe = data.get("best_mpjpe_3dpw", float('inf'))
                print(f"📊 Restored existing historical best: {current_best_mpjpe} mm")
        except: pass

    # In case we restart, mark all existing ones as evaluated (optional, but let's evaluate them all if they haven't been)
    while True:
        try:
            ckpts = get_checkpoints()
            for ckpt in ckpts:
                if ckpt not in evaluated_ckpts:
                    evaluate_ckpt(ckpt)
                    evaluated_ckpts.add(ckpt)
        except Exception as e:
            print(f"Daemon Error: {e}")
            
        time.sleep(60) # Poll every 60 seconds

if __name__ == "__main__":
    main()
