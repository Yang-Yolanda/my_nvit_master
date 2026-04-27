import os
from typing import Optional

from pathlib import Path
import pyrootutils

def get_project_root():
    """Dynamically detect the NViT-master root directory."""
    try:
        root = pyrootutils.find_root(search_from=__file__, indicator=[".git", "pyproject.toml"])
    except:
        # Fallback to grandparent of this file
        root = Path(__file__).resolve().parent.parent
    return root

def get_humans_root():
    """Dynamically detect the 4D-Humans sibling directory."""
    root = get_project_root()
    # Priority 1: Environment variable
    if "HUMANS_ROOT" in os.environ:
        return Path(os.environ["HUMANS_ROOT"])
    
    # Priority 2: Sibling directory
    humans_root = root.parent / "4D-Humans"
    if humans_root.exists():
        return humans_root
    
    # Priority 3: Common cluster mounts
    fallback = Path("/cpfs_infra/shared/yangz/4D-Humans")
    if fallback.exists():
        return fallback
        
    return humans_root # Default to sibling even if not exists yet

def rebase_humans_path(path_str: Optional[str]) -> Optional[str]:
    """
    Map a path that was recorded under any `.../4D-Humans/...` tree onto the
    current `get_humans_root()` so evaluation works without hardcoded `/home/yangz/...`.

    Example: `/home/yangz/4D-Humans/data/3DPW/foo.jpg` → `{HUMANS_ROOT}/data/3DPW/foo.jpg`
    """
    if not path_str or not isinstance(path_str, str):
        return path_str
    hr = get_humans_root()
    marker = "4D-Humans/"
    if marker in path_str:
        suffix = path_str.split(marker, 1)[1].lstrip("/")
        return str(hr / suffix)
    hs = str(hr.resolve())
    if path_str.startswith(hs):
        return path_str
    return path_str


def resolve_eval_img_dir(dataset_name: str, yaml_img_dir: str) -> str:
    """
    Resolve ImageDataset IMG_DIR for CH5 eval.

    Priority:
      1) Per-dataset env, e.g. ``HMR2_EVAL_IMG_DIR_3DPW``, ``HMR2_EVAL_IMG_DIR_H36M``
      2) ``HMR2_EVAL_IMG_DIR`` (single override for all)
      3) ``rebase_humans_path(yaml_img_dir)`` so paths follow ``HUMANS_ROOT`` + relative suffix
    """
    if not yaml_img_dir:
        return yaml_img_dir
    tag = {
        "3DPW-TEST": "3DPW",
        "3DPW-OCC-TEST": "3DPW",
        "H36M-VAL-P2": "H36M",
        "MPI-INF-3DHP-TEST": "MPIINF",
        "POSETRACK-VAL": "POSETRACK",
        "LSP-EXTENDED": "LSP",
        "COCO-VAL": "COCO",
    }.get(dataset_name, dataset_name.replace("-", "_").upper())
    for key in (f"HMR2_EVAL_IMG_DIR_{tag}", "HMR2_EVAL_IMG_DIR"):
        v = os.environ.get(key)
        if v:
            p = Path(v.rstrip("/"))
            return str(p) + "/"
    out = rebase_humans_path(yaml_img_dir) or yaml_img_dir
    return out if out.endswith("/") else out + "/"


def resolve_data_path(relative_path):
    """
    Resolve a path relative to the 4D-Humans/data directory.
    Usage: resolve_data_path('3dpw_test.npz')
    """
    humans_root = get_humans_root()
    data_root = humans_root / "data"
    
    # Handle both common 'data/...' and just '...'
    if str(relative_path).startswith("data/"):
        return humans_root / relative_path
    
    return data_root / relative_path

def get_hydra_runtime_output_dir(fallback: Optional[str] = None) -> Optional[str]:
    """
    Return Hydra's actual run directory (where .hydra/, main job log, etc. live).

    Training code should set `cfg.paths.output_dir` to this value so TensorBoard,
    checkpoints, and `model_config.yaml` live beside Hydra outputs instead of a
    separately overridden flat `paths.output_dir`.
    """
    try:
        from hydra.core.hydra_config import HydraConfig

        return HydraConfig.get().runtime.output_dir
    except Exception:
        return fallback


def sync_cfg_paths_output_to_hydra_run(cfg) -> None:
    """Set cfg.paths.output_dir to Hydra's runtime dir so artifacts share one tree."""
    from omegaconf import open_dict

    if not hasattr(cfg, "paths"):
        return
    fb = getattr(cfg.paths, "output_dir", None)
    run_root = get_hydra_runtime_output_dir(fallback=fb)
    if run_root:
        with open_dict(cfg):
            cfg.paths.output_dir = run_root


def get_checkpoint_path(path_str):
    """
    Converts a potentially absolute /home/yangz path to a dynamic relative path.
    """
    if not path_str or not isinstance(path_str, str):
        return path_str
        
    if str(get_project_root()) in path_str:
        rel = path_str.split("/home/yangz/NViT-master/")[-1]
        return str(get_project_root() / rel)
    
    if str(get_humans_root()) in path_str:
        rel = path_str.split("/home/yangz/4D-Humans/")[-1]
        return str(get_humans_root() / rel)
        
    return path_str
