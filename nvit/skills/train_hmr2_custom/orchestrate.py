#!/home/yangz/.conda/envs/4D-humans/bin/python

import argparse
import sys
import os
import json
from pathlib import Path

# Add core skills to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from manage_experiment.skill_base import SkillBase

class HMR2CustomTrainingOrchestrator(SkillBase):
    def __init__(self, gpu="0", output_dir="output/hmr2_custom_training"):
        super().__init__(gpu=gpu)
        self.output_dir = Path(output_dir)
        self.checkpoint = "/home/yangz/NViT-master/logs/train/runs/2026-01-21_15-28-28/checkpoints/last.ckpt"

    def setup_data_verification(self):
        """Step 1: Data Setup & Verification (Reusing existing wheel)"""
        self.logger.info("� Step 1: Running Standard Data Verification...")
        verify_script = self.resolve_path("nvit/skills/verify_datasets.py")
        
        try:
            self.run_command([sys.executable, str(verify_script)])
            self.logger.info("✅ Data verification passed.")
            return True
        except Exception as e:
            self.logger.error(f"❌ Data verification failed: {e}")
            return False

    def setup_configs(self):
        """Step 2: Dataset Configuration Check"""
        self.logger.info("📄 Step 2: Verifying Dataset Configuration...")
        config_path = Path("/home/yangz/4D-Humans/hmr2/configs/datasets_tar.yaml")
        if not config_path.exists():
            self.logger.error(f"Config file not found: {config_path}")
            return False
        self.logger.info(f"✅ Found datasets_tar.yaml")
        return True

    def configure_code(self):
        """Step 3: Code Setting / Environment Prep"""
        self.logger.info("⚙️  Step 3: Configuring Environment...")
        # SkillBase already handles PYTHONPATH, but we ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True

    def run_training(self, epochs, batch_size, lr, dry_run=False):
        """Step 4: Execute HMR2 Training"""
        self.logger.info(f"🔥 Step 4: Starting HMR2 Training (Run 10: Multi-GPU BF16)...")
        
        cmd = [
            sys.executable, "nvit/train_guided.py",
            "experiment=hmr_vit_transformer",
            "data=mix_all",
            "++MODEL.BACKBONE.mamba_variant=spiral",
            "++MODEL.BACKBONE.gcn_variant=grid",
            f"++trainer.max_epochs={epochs}",
            f"++TRAIN.BATCH_SIZE={batch_size}",
            "++trainer.devices=-1",
            "++trainer.strategy=ddp",
            "++trainer.precision=bf16-mixed",
            f"++paths.output_dir={self.output_dir}",
            f"++FINETUNE_FROM=\"{self.checkpoint}\"",
            "++GENERAL.task_name=Run10_Custom_MultiGPU"
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
            f.write("HMR2 Custom Training Summary\n")
            f.write("="*30 + "\n")
            f.write(f"Checkpoint Used: {self.checkpoint}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            f.write("Evaluation Metrics:\n")
            for ds, metrics in results.get('results', {}).items():
                f.write(f"- {ds}: MPJPE={metrics['mpjpe']:.2f}, PA-MPJPE={metrics['pa_mpjpe']:.2f}\n")
        
        self.logger.info(f"✅ Pipeline Complete. Summary saved to {report_file}")

    def execute_full_pipeline(self, epochs, batch_size, lr, dry_run=False):
        """Standardized 6-Step Pipeline Execution"""
        self.logger.info(f"\n{'#'*80}\n# HMR2 Custom Training Orchestrator (Standard Flow)\n{'#'*80}")
        
        if not self.setup_data_verification(): return
        if not self.setup_configs(): return
        if not self.configure_code(): return
        
        self.run_training(epochs, batch_size, lr, dry_run)
        
        if not dry_run:
            results = self.run_testing(dry_run)
            self.generate_report(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill: HMR2 Custom Training with Custom Datasets")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="output/hmr2_custom_training")
    parser.add_argument("--dry_run", action="store_true")
    
    args = parser.parse_args()
    
    orchestrator = HMR2CustomTrainingOrchestrator(gpu=args.gpu, output_dir=args.output_dir)
    orchestrator.execute_full_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dry_run=args.dry_run
    )
