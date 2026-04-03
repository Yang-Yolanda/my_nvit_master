#!/home/yangz/.conda/envs/4D-humans/bin/python

import os
import subprocess
import sys
import logging
from pathlib import Path

class SkillBase:
    """
    Base class for NViT Skill engineering.
    Provides standardized methods for subprocess execution, environment setup, and logging.
    """
    def __init__(self, gpu="0", log_level=logging.INFO):
        self.gpu = gpu
        self.project_root = self._resolve_project_root()
        self.env = self._setup_environment()
        
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

    def _resolve_project_root(self):
        """ Resolves the absolute path to the NViT-master root. """
        # Assuming SkillBase is in nvit/skills/manage_experiment/
        root = Path(__file__).resolve().parent.parent.parent.parent
        return root

    def _setup_environment(self):
        """ Configures the environment for subprocess execution (GPU, PYTHONPATH). """
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu
        
        # Consistent PYTHONPATH for all skills
        paths = [
            str(self.project_root),
            str(self.project_root / "nvit"),
            str(self.project_root / "nvit" / "Code_Paper2_Implementation"),
            "/home/yangz/4D-Humans",
        ]
        env["PYTHONPATH"] = ":".join(paths) + ":" + env.get("PYTHONPATH", "")
        return env

    def run_command(self, cmd, cwd=None, check=True):
        """ Standardized subprocess runner with logging. """
        if cwd is None:
            cwd = self.project_root
            
        cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
        self.logger.info(f"🚀 Running: {cmd_str}")
        
        try:
            result = subprocess.run(
                cmd, 
                env=self.env, 
                cwd=cwd, 
                check=check,
                text=True,
                capture_output=False # Stream output to stdout/stderr
            )
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Command failed with exit code {e.returncode}")
            raise e
        except Exception as e:
            self.logger.error(f"⚠️ Unexpected error: {str(e)}")
            raise e

    def resolve_path(self, relative_path):
        """ Resolves a path relative to project root. """
        return self.project_root / relative_path

if __name__ == "__main__":
    # Smoke test
    skill = SkillBase(gpu="0")
    print(f"Project Root: {skill.project_root}")
    print(f"PYTHONPATH: {skill.env['PYTHONPATH']}")
