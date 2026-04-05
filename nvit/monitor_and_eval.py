import os
import time
import subprocess
import glob
import shutil
import json

# Configuration
LOGS_BASE = "/home/yangz/NViT-master/outputs/phase2_full_run/checkpoints"
OUTPUT_DIR = "/home/yangz/NViT-master/outputs"
LOG_FILE = os.path.join(OUTPUT_DIR, "auto_eval.log")
INTERACTION_LOG = "/home/yangz/NViT-master/nvit/INTERACTION_LOG.md"
BEST_CKPT_PATH = "/home/yangz/NViT-master/outputs/phase2_full_run/checkpoints/best_phase2.ckpt"
BEST_METRICS_PATH = "/home/yangz/NViT-master/outputs/phase2_full_run/best_metrics.json"

# State
evaluated_ckpts = set()
current_best_mpjpe = float('inf')

def get_checkpoints():
    ckpts = glob.glob(os.path.join(LOGS_BASE, "epoch=*-step=*.ckpt"))
    # Filter out best_phase2.ckpt if it exists in the same dir
    ckpts = [c for c in ckpts if "best_phase2" not in c]
    return sorted(ckpts, key=os.path.getmtime)

def evaluate_ckpt(ckpt_path):
    global current_best_mpjpe
    print(f"[{time.ctime()}] Found {ckpt_path}. Evaluating...")
    
    with open(LOG_FILE, "a") as f:
        f.write(f"\n\n{'='*40}\nEvaluating: {ckpt_path}\n{'='*40}\n")
    
    # Run standard_eval on ALL 6 datasets
    cmd = [
        "/home/yangz/.conda/envs/4D-humans/bin/python", 
        "nvit/skills/evaluate_model/standard_eval.py", 
        "--ckpt", ckpt_path, 
        "--dataset", "ALL", 
        "--use_mean_alignment"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":/home/yangz/NViT-master:/home/yangz/4D-Humans"
    
    result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    with open(LOG_FILE, "a") as f:
        f.write(result.stdout)
    
    # Simple extraction of 3DPW as the primary "best" metric (can be averaged if needed)
    mpjpe_3dpw = None
    all_metrics = {}
    
    for line in result.stdout.split('\n'):
        if "3DPW-TEST" in line and "MPJPE:" in line:
            try:
                mpjpe_3dpw = float(line.split("MPJPE:")[1].split()[0])
            except: pass
    
    if mpjpe_3dpw is not None:
        print(f"3DPW MPJPE: {mpjpe_3dpw}")
        
        # Promotion logic
        if mpjpe_3dpw < current_best_mpjpe:
            print(f"🏆 NEW BEST! {mpjpe_3dpw} < {current_best_mpjpe}. Promoting...")
            current_best_mpjpe = mpjpe_3dpw
            shutil.copy2(ckpt_path, BEST_CKPT_PATH)
            with open(BEST_METRICS_PATH, "w") as f:
                json.dump({"best_mpjpe_3dpw": current_best_mpjpe, "ckpt": ckpt_path, "timestamp": time.ctime()}, f)
            
            # Log to interaction log
            with open(INTERACTION_LOG, "a") as f:
                f.write(f"\n- **Auto-Eval [NEW BEST]**: {os.path.basename(ckpt_path)} -> 3DPW: {mpjpe_3dpw}mm. Promoted to `best_phase2.ckpt`.")
        else:
            with open(INTERACTION_LOG, "a") as f:
                f.write(f"\n- **Auto-Eval [Update]**: {os.path.basename(ckpt_path)} -> 3DPW: {mpjpe_3dpw}mm. (Best: {current_best_mpjpe}mm)")

def main():
    global current_best_mpjpe
    # Load current best if exists
    if os.path.exists(BEST_METRICS_PATH):
        try:
            with open(BEST_METRICS_PATH, "r") as f:
                data = json.load(f)
                current_best_mpjpe = data.get("best_mpjpe_3dpw", float('inf'))
                print(f"Loaded existing best: {current_best_mpjpe}")
        except: pass

    while True:
        ckpts = get_checkpoints()
        for ckpt in ckpts:
            if ckpt not in evaluated_ckpts:
                # If we have too many checkpoints (e.g. from a restart),
                # just evaluate the latest ones to catch up faster
                if len(ckpts) - list(ckpts).index(ckpt) > 3:
                     print(f"Skipping old checkpoint {ckpt} to catch up...")
                     evaluated_ckpts.add(ckpt)
                     continue
                
                evaluate_ckpt(ckpt)
                evaluated_ckpts.add(ckpt)
        time.sleep(300) # Increased to 5 min to avoid redundant checks

if __name__ == "__main__":
    main()
