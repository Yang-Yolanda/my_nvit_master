import subprocess
import time
from pathlib import Path
import sys
import threading

# Define combinations
# Define combinations
mamba_variants = ['spiral', 'bi', 'seq'] # spiral=Center-Out, bi=Bidirectional, seq=Sequential
gcn_variants = ['grid', 'random'] # grid=Spatial, random=Random-Graph

jobs = []
for m in mamba_variants:
    for g in gcn_variants:
        jobs.append((m, g))

print(f"🚀 Starting Parallel Surgical Fine-tuning Sweep ({len(jobs)} configurations) on 3 GPUs")
print(f"⚠️ Strategy: Maximize Utilization. Running all 8 jobs in parallel.")

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Path to train.py
train_script = "train.py"

# Allocation Strategy
# Total Jobs: 6
# GPU 0: 2 jobs
# GPU 1: 2 jobs
# GPU 2: 2 jobs
allocations = [
    (0, jobs[0]), (0, jobs[1]),
    (1, jobs[2]), (1, jobs[3]),
    (2, jobs[4]), (2, jobs[5])
]

# Function to run a job
def run_job(gpu_id, config):
    mamba_v, gcn_v = config
    log_file = log_dir / f"train_{mamba_v}_{gcn_v}.log"
    
    cmd = [
        "python", "-u", train_script,
        "--mamba", mamba_v,
        "--gcn", gcn_v,
        "--epochs", "3",
        "--batch_size", "16" 
    ]
    
    env = sys.modules['os'].environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] Launching: {mamba_v}+{gcn_v} > {log_file}")
    
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            process.wait()
            
        if process.returncode == 0:
            print(f"✅ [GPU {gpu_id}] Success: {mamba_v}+{gcn_v}")
        else:
            print(f"❌ [GPU {gpu_id}] Failed: {mamba_v}+{gcn_v} (Exit: {process.returncode})")
            
    except Exception as e:
        print(f"❌ [GPU {gpu_id}] Error: {config} -> {e}")

# Launch threads
threads = []
for gpu_id, config in allocations:
    t = threading.Thread(target=run_job, args=(gpu_id, config))
    t.start()
    threads.append(t)
    time.sleep(2) # Stagger start to avoid initial CPU/IO spike

for t in threads:
    t.join()

print("\n🎉 All Parallel Jobs Completed!")
