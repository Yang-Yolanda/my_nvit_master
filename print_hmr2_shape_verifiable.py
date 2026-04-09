import torch
import sys
from pathlib import Path

# Path setup to include 4D-Humans
sys.path.append('/home/yangz/4D-Humans')

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else '/home/yangz/NViT-master/nvit/Paper1_Diagnostics/weights/hmr2_pretrained.ckpt'

try:
    print(f"Loading checkpoint: {checkpoint_path}")
    # Load state_dict directly
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    
    # Identify backbone keys
    backbone_keys = [k for k in sd.keys() if 'backbone' in k or 'model.backbone' in k]
    
    # 1. Blocks count
    block_indices = set()
    for k in backbone_keys:
        if '.blocks.' in k:
            parts = k.split('.blocks.')[1].split('.')
            idx = int(parts[0])
            block_indices.add(idx)
    blocks_count = max(block_indices) + 1 if block_indices else 0
    
    # 2. Patch Embed shape
    patch_size = ("unknown", "unknown")
    for k in backbone_keys:
        if '.patch_embed.proj.weight' in k:
            w = sd[k]
            patch_size = (w.shape[2], w.shape[3])
            break
        
    # 3. Pos Embed shape
    n_tokens = "unknown"
    for k in backbone_keys:
        if '.pos_embed' in k:
            n_tokens = sd[k].shape[1]
            break
        
    # 4. Check for CLS token
    has_cls_token_param = any('.cls_token' in k for k in sd.keys())

    print(f"HMR2 Patch Embed Shape (ph, pw): {patch_size}")
    print(f"HMR2 Grid (N_tokens): {n_tokens}")
    print(f"HMR2 blocks: {blocks_count}")
    print(f"Has CLS token: {has_cls_token_param}")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
