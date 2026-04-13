import subprocess
import os
import re

def run_test(name, extra_args):
    print(f"\n🚀 Running {name} test...")
    cmd = [
        "python", "nvit/train_guided.py",
        "experiment=hmr_vit_transformer",
        "data=mix_all",
        "++trainer.max_steps=5",
        "++trainer.devices=1",
        "++GENERAL.LOG_STEPS=1",
        "++TRAIN.BATCH_SIZE=8"
    ] + extra_args
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract coordinates and loss from logs
    log = result.stderr
    coords_log = re.findall(r"🔍 \[Diag\] Step 0 \| Coords: Range=\[(.*?)\]", log)
    sampled_log = re.findall(r"🔍 \[Diag\] Sampled Feats: Mean=(.*?), Std=(.*)", log)
    loss_log = re.findall(r"train/loss_step=(.*?)[,\]]", log)
    
    print(f"  - Coords: {coords_log[0] if coords_log else 'N/A'}")
    print(f"  - Sampled Feats: {sampled_log[0] if sampled_log else 'N/A'}")
    print(f"  - Final Loss: {loss_log[-1] if loss_log else 'N/A'}")
    return loss_log[-1] if loss_log else None

if __name__ == "__main__":
    # Case 1: GT Guidance (Teacher Forcing)
    run_test("Teacher Forcing (GT)", ["++TRAIN.TEACHER_FORCING=True"])
    
    # Case 2: Predicted Guidance
    run_test("Predicted Guidance", ["++TRAIN.TEACHER_FORCING=False"])
    
    # Case 3: Identity Mock (Check logic stability)
    # Just run Predicted again or mock if possible
    print("\n--- Go/No-Go Check Complete ---")
