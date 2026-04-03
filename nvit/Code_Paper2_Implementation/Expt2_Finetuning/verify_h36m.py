#!/home/yangz/.conda/envs/4D-humans/bin/python
import sys
import os
import torch
from pathlib import Path

# Fix paths to allow importing hmr2
sys.path.append("/home/yangz/4D-Humans")

from hmr2.configs import dataset_eval_config
from hmr2.datasets import ImageDataset

def verify_h36m():
    print("🔍 Verifying Human3.6M (H3.6M-VAL-P2) Dataset...")
    
    # 1. Load Configuration
    dataset_cfg = dataset_eval_config()
    h36m_cfg = dataset_cfg.get('H36M-VAL-P2')
    
    if not h36m_cfg:
        print("❌ Error: H36M-VAL-P2 not found in dataset_eval_config")
        return

    print(f"配置文件 keys: {h36m_cfg.keys()}")
    
    # 2. Check File Paths
    npz_path = h36m_cfg['DATASET_FILE']
    img_dir = h36m_cfg['IMG_DIR']
    
    print(f"📄 Annotation File: {npz_path}")
    print(f"📂 Image Directory: {img_dir}")
    
    if not os.path.exists(npz_path):
        print("❌ Error: Annotation file not found!")
        return
    if not os.path.exists(img_dir):
        print("❌ Error: Image directory not found!")
        return
        
    # 3. Initialize Dataset
    print("\n⏳ Initializing ImageDataset...")
    try:
        # Convert keys to lowercase for __init__ unpacking if needed, 
        # but hmr2 might expect upper case in config? 
        # train.py had to lower() it. HMR2 ImageDataset __init__ takes (cfg, dataset_file, img_dir, ...)
        # Let's look at ImageDataset signature again.
        # It takes: dataset_file, img_dir, train=True, ...
        
        dataset = ImageDataset(
            dataset_file=npz_path,
            img_dir=img_dir,
            train=False,
            **{k.lower(): v for k, v in h36m_cfg.items() if k not in ['TYPE', 'DATASET_FILE', 'IMG_DIR']}
        )
        print(f"✅ Dataset Initialized (Length: {len(dataset)})")
        
        # 4. Try Loading Samples
        print("🔄 Loading first 5 samples...")
        for i in range(5):
            batch = dataset[i]
            img_name = batch['imgname']
            print(f"   [{i}] Loaded: {img_name} | Keypoints shape: {batch['keypoints_2d'].shape}")
            
        print("\n🎉 H3.6M Data Verification Successful! Data is ready.")
        
    except Exception as e:
        print(f"\n❌ Error during dataset loading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_h36m()
