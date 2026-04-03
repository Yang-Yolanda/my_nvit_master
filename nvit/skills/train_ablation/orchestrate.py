#!/home/yangz/.conda/envs/4D-humans/bin/python

import argparse
import sys
import os
import json
import csv
from pathlib import Path

# Add core skills to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from manage_experiment.skill_base import SkillBase
from nvit.masking_utils import get_experimental_groups

def parallel_worker(item, gpu, output_dir, checkpoint, epochs, batch_size, lr, dry_run):
    """ Top-level worker for parallel execution to avoid pickling errors. """
    name, config = item
    from orchestrate import AblationOrchestrator
    # Create a local instance for each worker to avoid thread/process safety issues with logger
    orch = AblationOrchestrator(gpu=gpu, output_dir=output_dir)
    orch.checkpoint = checkpoint 
    try:
        orch.execute_single_group(name, config, epochs, batch_size, lr, dry_run)
        return (name, "Success")
    except Exception as e:
        return (name, f"Failed: {str(e)}")

class AblationOrchestrator(SkillBase):
    def __init__(self, gpu="0", output_dir="output/ablation_results"):
        super().__init__(gpu=gpu)
        self.output_dir = Path(output_dir)
        self.checkpoint = "/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"

    def identify_paper(self, group_name):
        """ Identifies if the task belongs to Paper 1 or Paper 2. """
        p2_keywords = ['mamba', 'gcn', 'spiral', 'grid', 'skeleton']
        if any(k in group_name.lower() for k in p2_keywords):
            return "Paper 2 (Hybrid Architecture)"
        return "Paper 1 (Soft Coding / Masking)"

    def setup_data_and_configs(self, name, config):
        """ Step 1 & 2: Data/Dataset Configuration. """
        self.logger.info(f"📁 Setup Data for {name}...")
        # Verify dataset paths
        data_path = "/home/yangz/4D-Humans/data"
        if not os.path.exists(data_path):
            self.logger.warning(f"Data path {data_path} not found. Check local mounts.")
        
        # In HMR2/NViT, configs are often passed via hydra or args
        return {
            "batch_size": 16,
            "lr": 1e-5,
            "use_mixed": True
        }

    def configure_code(self, name, config):
        """ Step 3: Code Setting (Masking, Backbone, etc). """
        paper = self.identify_paper(name)
        self.logger.info(f"⚙️  Configure Code for {paper} Group: {name}")
        return True

    def run_training(self, name, cmd):
        """ Step 4: Training execution. """
        self.logger.info(f"🔥 Training Group: {name}")
        self.run_command(cmd)

    def run_testing(self, name, group_out):
        """ Step 5: Testing / Evaluation using standardized skill. """
        self.logger.info(f"🧪 Evaluation for {name}...")
        ckpt_path = group_out / "latest_ft_model.pth"
        if not ckpt_path.exists():
             ckpt_path = group_out / "checkpoint.pth"
             
        if not ckpt_path.exists():
            self.logger.warning(f"No checkpoint found for {name} in {group_out}")
            return None

        eval_cmd = [
            "python", "nvit/skills/evaluate_model/standard_eval.py",
            "--ckpt", str(ckpt_path),
            "--dataset", "3DPW-TEST",
            "--batch_size", "16"
        ]
        
        try:
            self.run_command(eval_cmd)
            metrics_file = group_out / "eval_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Evaluation failed for {name}: {str(e)}")
        return {"mpjpe": 0.0, "pa_mpjpe": 0.0}

    def generate_report(self, name, metrics):
        """ Step 6: Result / Report generation (CSV Summary). """
        report_file = self.output_dir / "ablation_summary.csv"
        paper = self.identify_paper(name)
        
        import csv
        file_exists = report_file.exists()
        
        with open(report_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Paper", "Group", "MPJPE", "PA-MPJPE", "Status"])
            
            writer.writerow([
                paper, 
                name, 
                metrics.get("mpjpe", "N/A"), 
                metrics.get("pa_mpjpe", "N/A"),
                "Success" if metrics.get("mpjpe", 0) > 0 else "Fail"
            ])
        self.logger.info(f"📊 Results appended to {report_file}")

    def run_campaign(self, tier=1, target_groups=None, epochs=10, batch_size=16, lr=1e-5, dry_run=False, ignore_errors=False, parallel=1):
        """
        Structured Orchestration: Data -> Dataset -> Code -> Train -> Test -> Result.
        """
        all_groups = get_experimental_groups(total_layers=32)
        
        if target_groups:
            targets_list = [g.strip() for g in target_groups.split(',')]
            groups = {g: all_groups[g] for g in targets_list if g in all_groups}
        elif tier:
            tier_groups = {k: v for k, v in all_groups.items() if k.startswith(f"Tier{tier}")}
            control_group = {k: v for k, v in all_groups.items() if k == 'Control'}
            # Merge: Tier groups first, then Control
            groups = {**tier_groups, **control_group}
        else:
            groups = all_groups

        if not groups:
            self.logger.error(f"No groups found for Tier={tier}")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if parallel > 1:
            self.logger.info(f"⚡ Parallel Mode Enabled: Running with {parallel} workers on GPU {self.gpu}.")
            from concurrent.futures import ProcessPoolExecutor
            import functools

            task_list = list(groups.items())
            
            with ProcessPoolExecutor(max_workers=parallel) as executor:
                # Use partial to pass complex args
                worker_call = functools.partial(
                    parallel_worker, 
                    gpu=self.gpu, 
                    output_dir=str(self.output_dir), 
                    checkpoint=self.checkpoint,
                    epochs=epochs, 
                    batch_size=batch_size, 
                    lr=lr, 
                    dry_run=dry_run
                )
                for res_name, status in executor.map(worker_call, task_list):
                    self.logger.info(f"🏁 Task finished: {res_name} -> {status}")
        else:
            for name, config in groups.items():
                self.execute_single_group(name, config, epochs, batch_size, lr, dry_run)
        
        self.logger.info(f"✅ Ablation Campaign Finished in {self.output_dir}")

    def execute_single_group(self, name, config, epochs, batch_size, lr, dry_run):
        """ Internal method to execute the 6-step flow for one group. """
        paper = self.identify_paper(name)
        self.logger.info(f"\n{'#'*80}\n# EXECUTION: {paper} -> {name}\n{'#'*80}")
        
        # Step 1 & 2: Data & Dataset Setup
        self.setup_data_and_configs(name, config)
        
        # Step 3: Code Setting
        self.configure_code(name, config)
        
        group_out = self.output_dir / name
        group_out.mkdir(parents=True, exist_ok=True)
        
        # Step 4: Training execution
        if config.get('mode') == 'architecture': # Tier 3
            script = "nvit/train_guided.py"
            # Hydra Syntax: key=value
            cmd = [
                sys.executable, script,
                "experiment=hmr_vit_transformer",
                "data=mix_all",
                f"++MODEL.BACKBONE.mamba_variant={config.get('mamba_variant', 'spiral')}",
                f"++MODEL.BACKBONE.gcn_variant={config.get('gcn_variant', 'grid')}",
                f"++MODEL.BACKBONE.switch_layer_1={config.get('sl1', 8)}",
                f"++MODEL.BACKBONE.switch_layer_2={config.get('sl2', 10)}",
                f"++trainer.max_epochs={epochs}",
                f"++TRAIN.BATCH_SIZE={batch_size}",
                "++trainer.devices=1", 
                "++trainer.precision=bf16-mixed",
                f"++paths.output_dir={group_out}",
                f"++FINETUNE_FROM=\"{self.checkpoint}\"",
                f"++GENERAL.task_name={name}"
            ]
        elif "Paper 2" in paper:
             # Legacy Fallback
            script = "nvit/Code_Paper2_Implementation/Variant_SpiralMamba_GridGCN/Expt2_Adaptive_NViT.py"
            cmd = ["python", script]
        else:
            script = "nvit/finetune_dense.py"
            cmd = [
                "python", script,
                "--dense", "--use-mixed", 
                "--checkpoint-path", self.checkpoint,
                "--mask-group", name,
                "--output_dir", str(group_out),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--lr", str(lr),
                "--model-ema", "--device", "cuda"
            ]
        
        if dry_run:
            self.logger.info(f"[DRY RUN] {' '.join(cmd)}")
            return
            
        # Execution Flow
        self.run_training(name, cmd)
        metrics = self.run_testing(name, group_out)
        self.generate_report(name, metrics or {})
        
        self.logger.info(f"✅ Ablation Campaign Finished in {self.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill: Sequential Paper 1 Ablation Orchestrator")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="output/paper1_ablation")
    parser.add_argument("--target_groups", type=str, default=None)
    parser.add_argument("--tier", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1, help="Number of groups to run in parallel")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--ignore_errors", action="store_true")
    
    args = parser.parse_args()
    
    orchestrator = AblationOrchestrator(gpu=args.gpu, output_dir=args.output_dir)
    orchestrator.run_campaign(
        tier=args.tier, 
        target_groups=args.target_groups,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        parallel=args.parallel,
        dry_run=args.dry_run,
        ignore_errors=args.ignore_errors
    )
