#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import os
import argparse
import numpy as np

def load_structure(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint not found: {ckpt_path}")
        return None
    
    print(f"Loading {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading {ckpt_path}: {e}")
        return None
    
    if 'pruned_structure' not in checkpoint:
        print(f"❌ No 'pruned_structure' metadata in {ckpt_path}")
        return None
        
    return checkpoint['pruned_structure']

def compare_structures(structures):
    # structures: dict of {sparsity_name: structure_dict}
    print("\n" + "="*60)
    print("🔍 Structural Consistency Analysis")
    print("="*60)
    
    all_layers = set()
    for s in structures.values():
        if s: all_layers.update(s.keys())
        
    sorted_layers = sorted(list(all_layers))
    
    print(f"Found {len(sorted_layers)} pruned layers/modules.")
    
    # Simple overlap check for the first few layers
    for i, layer in enumerate(sorted_layers[:5]): # Show first 5 examples
        print(f"\nExample Layer: {layer}")
        for name, struct in structures.items():
            if struct and layer in struct:
                data = struct[layer]
                # Assuming data is a tensor/list of REMAINING indices
                if isinstance(data, torch.Tensor):
                    count = data.numel()
                elif isinstance(data, list):
                    count = len(data)
                else:
                    count = "Unknown"
                print(f"  - {name}: Kept {count} neurons/channels")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='NViT-master/nvit/output/hmr2_ddp_v1')
    args = parser.parse_args()
    
    sparsities = [40, 60, 80]
    structures = {}
    
    print(f"Checking for checkpoints in {args.base_dir}...")
    for s in sparsities:
        path = os.path.join(args.base_dir, f'hmr2_pruned_sparsity_{s}.pth')
        structures[f"{s}%"] = load_structure(path)
        
    valid_structs = {k:v for k,v in structures.items() if v is not None}
    
    if not valid_structs:
        print("❌ No valid checkpoints found yet. Please wait for Pruning V7 to complete.")
    else:
        compare_structures(valid_structs)
