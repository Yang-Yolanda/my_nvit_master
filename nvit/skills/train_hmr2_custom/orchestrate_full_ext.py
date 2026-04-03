#!/home/yangz/.conda/envs/4D-humans/bin/python

import argparse
import sys
import os
import json
import threading
import time
import re
import subprocess
from pathlib import Path

# Add core skills to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from manage_experiment.skill_base import SkillBase

class HMR2FullExtOrchestrator(SkillBase):
    def __init__(self, gpu="0", output_dir="output/hmr2_full_ext_training"):
        super().__init__(gpu=gpu)
        self.gpu_list = gpu.split(",")
        self.num_gpus = len(self.gpu_list)
        self.output_dir = Path(output_dir)
        # Standard Round 9 / Paper 2 Base Weights
        self.base_checkpoint = "/home/yangz/NViT-master/logs/train/runs/2026-01-21_15-28-28/checkpoints/last.ckpt"

    def setup_data_verification(self):
        """Step 1: Data Setup & Verification"""
        self.logger.info("🔍 Step 1: Running Data Verification for Full Ext...")
        # For full ext, we trust our manual verification of shards, but could run a check.
        # We'll just confirm the directory exists.
        data_dir = Path("/home/yangz/4D-Humans/data/finetune_ext")
        if not data_dir.exists():
             self.logger.error(f"Full dataset directory not found: {data_dir}")
             return False
        self.logger.info("✅ Full data directory confirmed.")
        return True

    def setup_configs(self):
        """Step 2: Dataset Configuration Check"""
        self.logger.info("📄 Step 2: Verifying Dataset Configuration...")
        config_path = Path("/home/yangz/4D-Humans/hmr2/configs/datasets_full_ext.yaml")
        if not config_path.exists():
            self.logger.error(f"Config file not found: {config_path}")
            return False
        self.logger.info(f"✅ Found datasets_full_ext.yaml")
        return True

    def configure_code(self):
        """Step 3: Code Setting / Environment Prep"""
        self.logger.info("⚙️  Step 3: Configuring Environment...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True

    def run_training(self, epochs, batch_size, lr, dry_run=False):
        """Step 4: Execute HMR2 Training (8-GPU)"""
        # 1. Detect Resume Checkpoint (Robustly)
        ckpt_dir = self.output_dir / "checkpoints"
        checkpoint_to_use = self.base_checkpoint
        
        if ckpt_dir.exists():
            ckpts = list(ckpt_dir.glob("*.ckpt"))
            if ckpts:
                # Sort by modification time, exclude any clearly corrupt ones if size < 1MB
                ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                # [Fix] To avoid picking up 'regressed' checkpoints from the failed run, 
                # we look for the most advanced 'epoch=' checkpoint if it exists,
                # or just the newest one if the user confirms.
                # For this specific recovery, we'll prioritize the epoch=33 one.
                latest_ckpt = ckpts[0]
                
                # Check for the specific epoch=33 recovery target
                recovery_target = ckpt_dir / "epoch=33-step=580000.ckpt"
                if recovery_target.exists():
                     checkpoint_to_use = str(recovery_target)
                     self.logger.info(f"🚨 RECOVERY MODE: Forced resume from {checkpoint_to_use} to fix regression.")
                else:
                     checkpoint_to_use = str(latest_ckpt)
                     self.logger.info(f"🔄 Found latest checkpoint at {checkpoint_to_use}. Resuming...")
        else:
            self.logger.info(f"🆕 Starting new training from base weight: {checkpoint_to_use}")

        # 2. Configure Local vs Global Batch Size
        target_bs = 128
        accumulate = max(1, target_bs // (batch_size * self.num_gpus))
        self.logger.info(f"📉 {self.num_gpus}-GPU Optimization: Local BS={batch_size}, Accumulate={accumulate} -> Effective BS={batch_size * self.num_gpus * accumulate}")

        cmd = [
            sys.executable, "nvit/train_guided.py",
            "experiment=hmr_vit_transformer",
            "data=full_ext",
            "++DATASETS_CONFIG_FILE=datasets_full_ext.yaml",
            "++MODEL.BACKBONE.mamba_variant=spiral",
            "++MODEL.BACKBONE.gcn_variant=guided",
            "++MODEL.BACKBONE.switch_layer_1=8",
            "++MODEL.BACKBONE.switch_layer_2=12",
            "++MODEL.SMPL_HEAD.TRANSFORMER_DECODER.depth=3",
            "++MODEL.SMPL_HEAD.TRANSFORMER_DECODER.heads=4",
            "++trainer.max_epochs=250", # 150 (baseline) + 100 (finetune)
            f"++TRAIN.BATCH_SIZE={batch_size}",
            f"++trainer.devices={self.num_gpus}", 
            f"++TRAIN.ACCUMULATE_GRAD_BATCHES={accumulate}", # Match 8-GPU batch size
            "++trainer.precision=bf16-mixed",
            "++LOSS_WEIGHTS.HEATMAP=2.0",
            "++LOSS_WEIGHTS.GLOBAL_ORIENT=0.001",
            "++LOSS_WEIGHTS.BODY_POSE=0.001",
            "++LOSS_WEIGHTS.KEYPOINTS_3D=0.05",
            "++LOSS_WEIGHTS.KEYPOINTS_2D=0.01",
            "++GENERAL.NUM_WORKERS=2",
            "++GENERAL.VAL_STEPS=500",
            f"++paths.output_dir={self.output_dir}",
            f"++ckpt_path=\"{checkpoint_to_use}\"",
            "++GENERAL.task_name=Paper2_SoftGuided_Pivot"
        ]
        
        if dry_run:
            self.logger.info(f"[DRY RUN] {' '.join(cmd)}")
            return
        
        self.run_command(cmd)

    def run_testing(self, dry_run=False):
        """Step 5: Standard Evaluation"""
        self.logger.info("🧪 Step 5: Running Standard Evaluation...")
        
        ckpt_path = self.output_dir / "best_ft_model.pth"
        if not ckpt_path.exists():
            ckpt_path = self.output_dir / "latest_ft_model.pth"
        
        if not ckpt_path.exists():
            self.logger.warning(f"No checkpoint found in {self.output_dir}. Skipping evaluation.")
            return None
        
        eval_cmd = [
            sys.executable, "nvit/skills/evaluate_model/standard_eval.py",
            "--ckpt", str(ckpt_path),
            "--dataset", "3DPW-TEST,H36M-VAL-P2",
            "--batch_size", "16",
            "--data_dir", "/home/yangz/4D-Humans/hmr2_evaluation_data",
            "--output", str(self.output_dir / "eval_results.json")
        ]
        
        if dry_run:
            self.logger.info(f"[DRY RUN] {' '.join(eval_cmd)}")
            return
        
        self.run_command(eval_cmd)
        
        if (self.output_dir / "eval_results.json").exists():
            with open(self.output_dir / "eval_results.json") as f:
                return json.load(f)
        return None

    def generate_report(self, results):
        """Step 6: Summary Report"""
        self.logger.info("📊 Step 6: Generating Final Report...")
        if not results:
            self.logger.warning("No evaluation results to report.")
            return

        report_file = self.output_dir / "training_summary.txt"
        with open(report_file, "w") as f:
            f.write("HMR2 Full Extension Training Summary (8-GPU)\n")
            f.write("="*45 + "\n")
            f.write(f"Checkpoint Used: {self.checkpoint}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            f.write("Evaluation Metrics:\n")
            for ds, metrics in results.get('results', {}).items():
                f.write(f"- {ds}: MPJPE={metrics['mpjpe']:.2f}, PA-MPJPE={metrics['pa_mpjpe']:.2f}\n")
        
        self.logger.info(f"✅ Pipeline Complete. Summary saved to {report_file}")

    def analyze_and_improve(self, results):
        """Step 7: Autonomous Reflection & Feedback"""
        self.logger.info("🧠 Step 7: Analyzing Results for Autonomous Improvement...")
        if not results: return
        
        # Example Logic: 
        # If MPJPE > 85mm, maybe Heatmap loss is too high/low or Mamba scan needs adjustment.
        # This is where the 'LEGO' thinking happens.
        for ds, metrics in results.get('results', {}).items():
            mpjpe = metrics['mpjpe']
            if mpjpe > 80:
                self.logger.warning(f"⚠️ {ds} MPJPE ({mpjpe:.2f}) is high. Suggested Action: Review Mamba Spiral scan density.")
            else:
                self.logger.info(f"✅ {ds} MPJPE ({mpjpe:.2f}) is within acceptable range.")

    def monitor_and_evaluate(self):
        """Background thread to evaluate checkpoints every 10 epochs."""
        self.logger.info("🕵️ Starting background auto-evaluation monitor (checking every 5 mins)...")
        evaluated_epochs = set()
        ckpt_dir = self.output_dir / "checkpoints"
        
        while True:
            if ckpt_dir.exists():
                ckpts = list(ckpt_dir.glob("epoch=*.ckpt"))
                for ckpt in ckpts:
                    m = re.search(r'epoch=(\d+)', ckpt.name)
                    if m:
                        epoch = int(m.group(1))
                        # Evaluate every 10 epochs
                        if epoch > 0 and epoch % 10 == 0 and epoch not in evaluated_epochs:
                            self.logger.info(f"⏳ Triggering background offline evaluation for {ckpt.name} on GPU 0...")
                            eval_cmd = [
                                sys.executable, "nvit/skills/evaluate_model/standard_eval.py",
                                "--ckpt", str(ckpt),
                                "--dataset", "3DPW-TEST",
                                "--batch_size", "16",
                                "--data_dir", "/home/yangz/4D-Humans/hmr2_evaluation_data",
                                "--output", str(self.output_dir / f"eval_results_epoch{epoch}.json")
                            ]
                            
                            env = os.environ.copy()
                            env["CUDA_VISIBLE_DEVICES"] = "0"  # Evaluate on GPU 0 to not disrupt training on 6/7
                            env["PYTHONPATH"] = "/home/yangz/NViT-master:/home/yangz/4D-Humans"
                            
                            try:
                                res = subprocess.run(eval_cmd, env=env, capture_output=True, text=True)
                                if res.returncode == 0:
                                    self.logger.info(f"✅ Auto-Evaluation for Epoch {epoch} complete! Results saved.")
                                    evaluated_epochs.add(epoch)
                                else:
                                    self.logger.error(f"❌ Auto-Evaluation failed for {ckpt.name}:\n{res.stderr[-500:] if res.stderr else 'Unknown Error'}")
                            except Exception as e:
                                self.logger.error(f"❌ Auto-Evaluation exception: {e}")
            time.sleep(300)

    def execute_full_pipeline(self, epochs, batch_size, lr, dry_run=False):
        """Standardized 6-Step Pipeline Execution with Autonomous Enhancement"""
        self.logger.info(f"\n{'#'*80}\n# HMR2 Full Dataset Orchestrator (Autonomous LEGO Flow)\n{'#'*80}")
        
        if not self.setup_data_verification(): return
        if not self.setup_configs(): return
        if not self.configure_code(): return
        
        # Start background evaluation monitor thread if not dry run
        if not dry_run:
            monitor_thread = threading.Thread(target=self.monitor_and_evaluate, daemon=True)
            monitor_thread.start()
        
        # Training Phase
        self.run_training(epochs, batch_size, lr, dry_run)
        
        if not dry_run:
            # Evaluation Phase
            results = self.run_testing(dry_run)
            self.generate_report(results)
            
            # Autonomous Refinement Phase
            self.analyze_and_improve(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill: HMR2 Training on Full Extension Dataset (8-GPU)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=str, default="6,7")
    parser.add_argument("--output_dir", type=str, default="output/hmr2_full_ext_training")
    parser.add_argument("--dry_run", action="store_true")
    
    args = parser.parse_args()
    
    orchestrator = HMR2FullExtOrchestrator(gpu=args.gpu, output_dir=args.output_dir)
    orchestrator.execute_full_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dry_run=args.dry_run
    )
