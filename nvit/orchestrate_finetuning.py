#!/home/yangz/.conda/envs/4D-humans/bin/python
import argparse
import subprocess
import sys
import os
from pathlib import Path
from masking_utils import get_experimental_groups



def run_campaign(args):
    """
    Orchestrate the 14-group finetuning campaign.
    """
    # 1. Get Groups
    groups = get_experimental_groups()
    
    # 2. Filter Groups
    if args.target_group:
        if args.target_group not in groups:
            print(f"Error: Target group '{args.target_group}' not found in {list(groups.keys())}")
            return
        target_groups = {args.target_group: groups[args.target_group]}
    else:
        target_groups = groups

    # 3. Setup Paths
    base_output_dir = Path("output/finetune_campaign")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = "/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
    
    # 4. GPU Setup
    env = os.environ.copy()
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        gpu_list = args.gpus.split(',')
        num_gpus = len(gpu_list)
        print(f"🖥️  Using GPUs: {args.gpus} (Count: {num_gpus})")
    else:
        num_gpus = 1
        print("🖥️  Using Default GPU Configuration")

    # 4. Iterate and Run
    for name, config in target_groups.items():
        print(f"\n{'='*60}")
        print(f"🚀 Launching Finetuning for Group: {name}")
        print(f"{'='*60}")
        
        group_output_dir = base_output_dir / name
        
        # Build Command
        cmd = []
        
        if num_gpus > 1:
            # Use torchrun for DDP
            cmd += [
                "/home/yangz/.conda/envs/4D-humans/bin/torchrun",
                f"--nproc_per_node={num_gpus}",
                "--master_port=29505" # Use a non-standard port to avoid conflicts
            ]
        else:
            # Single Process
            cmd += ["/home/yangz/.conda/envs/4D-humans/bin/python"]

        cmd += [
            "nvit/finetune_dense.py",
            "--dense",
            "--use-mixed", # Enable Mixed Dataset
            "--checkpoint-path", checkpoint,
            "--mask-group", name,
            "--output_dir", str(group_output_dir),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--model-ema", # Enable EMA
            "--device", "cuda"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        if not args.dry_run:
            try:
                # We use check_call to run sequentially and stop on error
                subprocess.check_call(cmd, env=env)
                print(f"✅ Group {name} Completed Successfully.")
            except subprocess.CalledProcessError as e:
                print(f"❌ Group {name} FAILED with exit code {e.returncode}")
                if not args.ignore_errors:
                    sys.exit(1)
        else:
            print("Suggest running (Dry Run): " + " ".join(cmd))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate Masked Finetuning Campaign")
    parser.add_argument("--target_group", type=str, default=None, help="Run specific group (default: run all)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per group")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (Per GPU)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--ignore_errors", action="store_true", help="Continue to next group on failure")
    parser.add_argument("--gpus", type=str, default="0", help="Comma separated list of GPU IDs to use (e.g. 0,1)")
    
    args = parser.parse_args()
    run_campaign(args)
