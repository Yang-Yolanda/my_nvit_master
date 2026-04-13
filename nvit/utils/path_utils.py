import os

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
