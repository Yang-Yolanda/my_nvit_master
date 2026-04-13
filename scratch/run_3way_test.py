import subprocess
import os
import time

def run_ablation(name, flags):
    print(f"\n🔥 Starting Ablation: {name}")
    log_file = f"scratch/ablation_{name.lower().replace(' ', '_')}.log"
    cmd = [
        "python", "nvit/train_guided.py",
        "experiment=hmr_vit_transformer",
        "data=mix_all",
        "++trainer.max_steps=1000",
        "++trainer.devices=1",
        "++GENERAL.LOG_STEPS=100",
        "++TRAIN.BATCH_SIZE=8",
    ] + flags
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=f)
        print(f"  - Log: {log_file}")
        # We'll wait for it to finish
        process.wait()
    print(f"✅ Finished {name}")

if __name__ == "__main__":
    os.makedirs("scratch", exist_ok=True)
    
    # 1. Teacher Forcing (Oracle)
    run_ablation("GT Guidance", ["++TRAIN.TEACHER_FORCING=True", "++TRAIN.BYPASS_GUIDANCE=False"])
    
    # 2. Predicted Guidance (Standard)
    run_ablation("Pred Guidance", ["++TRAIN.TEACHER_FORCING=False", "++TRAIN.BYPASS_GUIDANCE=False"])
    
    # 3. No Guidance (Bypass)
    run_ablation("No Guidance", ["++TRAIN.BYPASS_GUIDANCE=True"])
    
    print("\n--- 3-Way Comparative Test Complete ---")
