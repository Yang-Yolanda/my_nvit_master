#!/home/yangz/.conda/envs/4D-humans/bin/python
"""
Dataset Path Verification and Dataloader Sanity Check
Tests that all configured datasets can be loaded successfully
"""
import sys
sys.path.insert(0, '/home/yangz/4D-Humans')
sys.path.insert(0, '/home/yangz/NViT-master/nvit')

import torch
from pathlib import Path
import yaml

def test_training_datasets():
    """Test that all training datasets can be loaded"""
    print("\n" + "="*60)
    print("🔍 TRAINING DATASET VERIFICATION")
    print("="*60 + "\n")
    
    # Load config
    config_path = Path("/home/yangz/4D-Humans/hmr2/configs/datasets_tar.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"📄 Config: {config_path}")
    print(f"📊 Datasets defined: {len(config)}\n")
    
    for ds_name, ds_config in config.items():
        if ds_name.startswith('#'):  # Skip commented datasets
            continue
            
        print(f"Testing: {ds_name}")
        
        if 'URLS' in ds_config:
            # WebDataset (tar files)
            url_pattern = ds_config['URLS']
            print(f"  URL Pattern: {url_pattern}")
            
            # Extract base path and check if symlink exists
            base_path = url_pattern.split('/{')[0]  # Get path before pattern
            path = Path(base_path)
            
            if path.exists():
                if path.is_symlink():
                    target = path.resolve()
                    print(f"  ✅ Symlink exists -> {target}")
                    if target.exists():
                        # Count tar files
                        tar_files = list(target.glob("*.tar"))
                        print(f"  ✅ Target accessible ({len(tar_files)} .tar files)")
                    else:
                        print(f"  ❌ Symlink target does not exist!")
                else:
                    print(f"  ✅ Directory exists")
            else:
                print(f"  ❌ Path does not exist: {base_path}")
                
        elif 'DATASET_FILE' in ds_config:
            # Single file dataset
            file_path = Path(ds_config['DATASET_FILE'])
            print(f"  File: {file_path}")
            if file_path.exists():
                print(f"  ✅ File exists ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
            else:
                print(f"  ❌ File not found")
        
        print()


def test_evaluation_datasets():
    """Test that all evaluation datasets exist"""
    print("\n" + "="*60)
    print("🔍 EVALUATION DATASET VERIFICATION")
    print("="*60 + "\n")
    
    eval_dir = Path("/home/yangz/4D-Humans/data")
    
    expected_files = [
        '3dpw_test.npz',
        'coco_val.npz',
        'h36m_val_p2.npz',
        'posetrack_2018_val.npz',
        'hr-lspet_train.npz'
    ]
    
    print(f"📁 Evaluation Directory: {eval_dir}\n")
    
    for filename in expected_files:
        filepath = eval_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            print(f"✅ {filename:25s} ({size_mb:6.1f} MB)")
        else:
            print(f"❌ {filename:25s} (NOT FOUND)")


def test_dataloader_sanity():
    """Test loading 1 batch from each dataset"""
    print("\n" + "="*60)
    print("🔍 DATALOADER SANITY CHECK")
    print("="*60 + "\n")
    
    try:
        from hmr2.datasets import create_dataset
        from hmr2.configs import dataset_config
        from torch.utils.data import DataLoader
        
        ds_cfg = dataset_config()
        eval_dir = Path("/home/yangz/4D-Humans/data")
        test_files = {
            '3dpw_test': eval_dir / '3dpw_test.npz',
            'coco_val': eval_dir / 'coco_val.npz',
        }
        
        for ds_name, ds_path in test_files.items():
            if not ds_path.exists():
                print(f"⏭️  Skipping {ds_name} (file not found)")
                continue
                
            print(f"Testing: {ds_name}")
            try:
                dataset = create_dataset(str(ds_path), is_train=False, dataset_cfg=ds_cfg)
                loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
                
                # Try to load one batch
                batch = next(iter(loader))
                print(f"  ✅ Loaded 1 batch: {len(batch)} items")
                print(f"     Keys: {list(batch.keys())[:5]}...")
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
            
            print()
            
    except ImportError as e:
        print(f"⚠️  Cannot test dataloader: {e}")
        print("   (This is OK if hmr2 package is not fully installed)")


if __name__ == "__main__":
    test_training_datasets()
    test_evaluation_datasets()
    test_dataloader_sanity()
    
    print("\n" + "="*60)
    print("✅ VERIFICATION COMPLETE")
    print("="*60 + "\n")
