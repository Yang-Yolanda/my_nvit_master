#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
from torch.utils.data import ConcatDataset, IterableDataset
from datasets_3dpw import ThreeDPWDataset
from hmr2.datasets.image_dataset import ImageDataset as HMR2ImageDataset
import glob
import os
from pathlib import Path
from yacs.config import CfgNode as CN

def create_mixed_dataset(args, split='train'):
    """
    Creates a composite dataset from multiple sources defined in the model config.
    Supports both Map-style (3DPW) and Iterable-style (WebDataset tars).
    """
    # 1. Load Model Config to get dataset list and weights
    # Note: In nvit, args often has the config object or we load it from a known path
    cfg = None
    if hasattr(args, 'cfg'):
        cfg = args.cfg
    else:
        # Fallback to standard HMR2 default if not passed
        from hmr2.configs import default_config as model_config
        cfg = model_config()

    datasets = []
    
    # 2. 3DPW - Load if present in config or by default
    print(f"Loading 3DPW ({split})...")
    
    # --- Monkey Patch HMR2 to avoid hardcoded path error ---
    try:
        import hmr2.datasets.image_dataset
        hmr2.datasets.image_dataset.load_amass_hist_smooth = lambda: None
        from hmr2.datasets import smplh_prob_filter
        smplh_prob_filter.load_amass_hist_smooth = lambda: None
    except ImportError:
        pass
    # -------------------------------------------------------

    try:
        # Use centralized path on HDD
        d_3dpw = ThreeDPWDataset(config=args, split=split)
        datasets.append(d_3dpw)
    except Exception as e:
        print(f"Warning: Failed to load 3DPW: {e}")

    # 3. WebDatasets (MPII, COCO, H36M, etc.) - Only for Training
    if split == 'train':
        # Load datasets_tar.yaml configuration
        import yaml
        config_path = Path("/home/yangz/4D-Humans/hmr2/configs/datasets_tar.yaml")
        
        if not config_path.exists():
            print(f"Warning: Config file {config_path} not found. Skipping WebDatasets.")
        else:
            with open(config_path) as f:
                datasets_config = yaml.safe_load(f)
            
            all_iterable_datasets = []
            # Convert 3DPW (Map) to Iterable
            class IterableWrapper(IterableDataset):
                def __init__(self, map_ds):
                    self.map_ds = map_ds
                def __iter__(self):
                    import torch.distributed as dist
                    total_len = len(self.map_ds)
                    indices = torch.randperm(total_len).tolist()
                    if dist.is_available() and dist.is_initialized():
                        indices = indices[dist.get_rank()::dist.get_world_size()]
                    worker_info = torch.utils.data.get_worker_info()
                    if worker_info is not None:
                        indices = indices[worker_info.id :: worker_info.num_workers]
                    for idx in indices:
                        yield self.map_ds[idx]
                def __len__(self):
                    return len(self.map_ds)

            for d in datasets:
                all_iterable_datasets.append(IterableWrapper(d))

            # Load each dataset from config
            for ds_name, ds_config in datasets_config.items():
                if ds_name.startswith('#'):  # Skip commented datasets
                    continue
                
                if 'URLS' not in ds_config:
                    continue
                
                # Extract tar file pattern from URLS
                url_pattern = ds_config['URLS']
                
                # Parse the pattern to get actual files
                # Pattern format: /path/to/dataset/{000000..000312}.tar
                import re
                match = re.match(r'(.+)/\{(\d+)\.\.(\d+)\}\.tar', url_pattern)
                
                if match:
                    base_path = match.group(1)
                    start_idx = int(match.group(2))
                    end_idx = int(match.group(3))
                    
                    # Generate tar file list
                    tar_files = []
                    for i in range(start_idx, end_idx + 1):
                        tar_file = f"{base_path}/{i:06d}.tar"
                        if os.path.exists(tar_file):
                            tar_files.append(tar_file)
                    
                    if tar_files:
                        epoch_size = ds_config.get('epoch_size', 100_000)
                        print(f"Loading {ds_name} ({len(tar_files)} shards, epoch_size={epoch_size})...")
                        
                        # Create internal HMR2 style config
                        local_cfg = CN()
                        local_cfg.MODEL = CN()
                        local_cfg.MODEL.IMAGE_SIZE = 256
                        local_cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
                        local_cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
                        local_cfg.DATASETS = CN()
                        local_cfg.DATASETS.CONFIG = CN()
                        local_cfg.DATASETS.CONFIG.SCALE_FACTOR = 0.25
                        local_cfg.DATASETS.CONFIG.ROT_FACTOR = 30
                        local_cfg.DATASETS.CONFIG.TRANS_FACTOR = 0.1
                        local_cfg.DATASETS.CONFIG.COLOR_SCALE = 0.2
                        local_cfg.DATASETS.CONFIG.ROT_AUG_RATE = 0.6
                        local_cfg.DATASETS.CONFIG.TRANS_AUG_RATE = 0.5
                        local_cfg.DATASETS.CONFIG.DO_FLIP = True
                        local_cfg.DATASETS.CONFIG.FLIP_AUG_RATE = 0.5
                        local_cfg.DATASETS.CONFIG.EXTREME_CROP_AUG_RATE = 0.10
                        
                        try:
                            d_raw = HMR2ImageDataset.load_tars_as_webdataset(
                                local_cfg, 
                                tar_files, 
                                train=True, 
                                resampled=True,
                                epoch_size=epoch_size
                            )
                            
                            class WDSWrapper(IterableDataset):
                                def __init__(self, ds, length):
                                    self.ds = ds
                                    self.length = length
                                def __iter__(self):
                                    return iter(self.ds)
                                def __len__(self):
                                    return self.length
                            
                            all_iterable_datasets.append(WDSWrapper(d_raw, epoch_size))
                        except Exception as e:
                            print(f"Error loading {ds_name}: {e}")
                    else:
                        print(f"Warning: No tar files found for {ds_name} at {base_path}")
                else:
                    print(f"Warning: Could not parse URL pattern for {ds_name}: {url_pattern}")

        print(f"Returning ChainDataset with {len(all_iterable_datasets)} sources.")
        return torch.utils.data.ChainDataset(all_iterable_datasets)
    
    if not datasets:
        raise ValueError("No datasets loaded!")
        
    if len(datasets) == 1:
        return datasets[0]
        
    return ConcatDataset(datasets)
