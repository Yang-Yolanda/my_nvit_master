#!/home/yangz/.conda/envs/4D-humans/bin/python

import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup Paths
NVIT_ROOT = '/home/yangz/NViT-master/nvit'
HUMANS_DIR = '/home/yangz/4D-Humans'
sys.path.append(NVIT_ROOT)
sys.path.append(HUMANS_DIR)
sys.path.append('/home/yangz/NViT-master/nvit/Code_Paper2_Implementation')

# Imports
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from datasets_3dpw import create_dataset
from smpl_topology import SMPL_PARENTS, SMPL_JOINT_NAMES

# ==============================================================================
# Helper Functions (Entropy & KTI)
# ==============================================================================

def calculate_entropy(attn_map):
    """
    Compute Shannon Entropy of attention scores.
    attn_map: (B, H, N, N)
    output: scalar entropy
    """
    if torch.isnan(attn_map).any():
        return 0.0
    epsilon = 1e-9
    # Average over heads for simplicity or keep separate? 
    # Usually we average over heads to get layer-wise entropy.
    # attn_map is Softmaxed.
    p = attn_map + epsilon
    entropy = -torch.sum(p * torch.log(p), dim=-1).mean()
    return entropy.item()

def transform_keypoints_to_crop(keypoints, center, scale, img_size=256):
    """
    Transform original 2D keypoints to the cropped image coordinate system.
    """
    scale_px = scale * 200.0
    ul = center - scale_px / 2.0
    kp_xy = keypoints[:, :2]
    kp_new = (kp_xy - ul) * (img_size / scale_px)
    return kp_new

def create_dynamic_mask(keypoints_crop, img_w=192, img_h=256, patch_size=16):
    """
    Creates the 'Physical/Kinematic Mask' based on joint locations in the crop.
    """
    grid_h = img_h // patch_size
    grid_w = img_w // patch_size
    N = grid_h * grid_w
    
    # Map joints to patch indices
    joint_to_patch = {}
    for i, (u, v) in enumerate(keypoints_crop):
        u = max(0, min(img_w-0.1, u))
        v = max(0, min(img_h-0.1, v))
        row = int(v // patch_size)
        col = int(u // patch_size)
        idx = row * grid_w + col
        joint_to_patch[i] = int(idx)
        
    mask = torch.zeros((N, N))
    parents = SMPL_PARENTS
    name_to_idx = {name: i for i, name in enumerate(SMPL_JOINT_NAMES)}
    
    for child, parent in parents.items():
        if parent is None: continue
        if child not in name_to_idx or parent not in name_to_idx: continue
        
        c_idx = name_to_idx[child]
        p_idx = name_to_idx[parent]
        
        if c_idx in joint_to_patch and p_idx in joint_to_patch:
            patch_c = joint_to_patch[c_idx]
            patch_p = joint_to_patch[p_idx]
            
            if patch_c < N and patch_p < N:
                mask[patch_p, patch_c] = 1.0
                mask[patch_c, patch_p] = 1.0
                
    mask = mask + torch.eye(N) # Self-loops
    return mask

def calculate_kmi_score(attn_map, mask):
    """
    Computes Kinematic Mutual Information (alignment score).
    attn_map: (N, N) Average Attention Matrix (CLS removed)
    mask: (N, N) Physical Adjacency
    """
    total_energy = attn_map.sum().item() + 1e-9
    aligned_energy = (attn_map * mask).sum().item()
    return aligned_energy / total_energy

# ==============================================================================
# Main Experiment
# ==============================================================================

def main():
    logger.info("Starting Expt1: Baseline Diagnostics (Entropy & KTI)...")
    
    # 1. Setup Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, _ = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()
    
    # 2. Register Hooks to Capture Attention
    captured_attn = {}
    def get_attn_hook(idx):
        def attn_hook(module, input, output):
            # output is typically the result of dropout(softmax(QK^T))
            # But in HMR2/timm implementation, we might be hooking the dropout layer AFTER softmax
            # So input[0] is likely the attention weights.
            if idx not in captured_attn: captured_attn[idx] = []
            captured_attn[idx] = input[0].detach().cpu() 
        return attn_hook

    hooks = []
    if hasattr(model.backbone, 'blocks'):
        for i, blk in enumerate(model.backbone.blocks):
            # Hook the 'attn_drop' which comes right after softmax
            if hasattr(blk.attn, 'attn_drop'):
                h = blk.attn.attn_drop.register_forward_hook(get_attn_hook(i))
                hooks.append(h)
    
    logger.info(f"Hooked {len(hooks)} ViT Layers.")
    
    # 3. Load Data
    class Args:
        batch_size = 1 # Keep 1 for clean per-sample KTI calculation
        num_workers = 0
        pin_mem = False
        data_path = '/home/yangz/4D-Humans/data' 
    
    dataset = create_dataset(Args(), split='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 4. Run Inference Loop
    metrics = {l: {'entropy': [], 'kmi': []} for l in range(len(hooks))}
    
    LIMIT = 100 # Analyze 100 samples for robust stats
    logger.info(f"Processing {LIMIT} samples...")
    
    for i, batch in tqdm(enumerate(dataloader), total=LIMIT):
        if i >= LIMIT: break
        
        # Prepare Metadata (Crop info)
        kp_2d_orig = batch['keypoints_2d'][0, :24, :2].numpy()
        # Simple crop logic reconstruction using bbox if available, or just keypoints bounds
        # Note: 4D-Humans loader creates the image crop, but metadata refers to original image
        # We need to recreate the crop transform to map Original Keypoints -> Cropped Patch Grid
        
        xmin, ymin = kp_2d_orig[:,0].min(), kp_2d_orig[:,1].min()
        xmax, ymax = kp_2d_orig[:,0].max(), kp_2d_orig[:,1].max()
        w, h = xmax - xmin, ymax - ymin
        center = np.array([xmin + w/2, ymin + h/2])
        scale = (max(w, h) * 1.2) / 200.0
        
        kp_crop = transform_keypoints_to_crop(kp_2d_orig, center, scale, img_size=256)
        phys_mask = create_dynamic_mask(kp_crop, img_w=192, img_h=256)
        
        # Forward Pass
        batch = recursive_to(batch, device)
        with torch.no_grad():
            model(batch)
            
        # Compute Metrics
        for layer_idx, attn_tensor in captured_attn.items():
            # attn_tensor: (1, H, N, N)
            
            # 1. Entropy (Global)
            # Use raw attention map (with CLS typically)
            ent = calculate_entropy(attn_tensor)
            metrics[layer_idx]['entropy'].append(ent)
            
            # 2. KTI (Structure)
            # Average heads -> (N, N)
            avg_attn = attn_tensor[0].mean(dim=0)
            
            # Handle CLS Token (Remove first row/col if N > mask size)
            # Mask is typically 192/16 * 256/16 = 12*16 = 192 tokens?
            # Check shape
            N_attn = avg_attn.shape[0]
            N_mask = phys_mask.shape[0]
            
            if N_attn > N_mask:
                # Remove CLS (index 0)
                avg_attn_spatial = avg_attn[1:, 1:]
            else:
                avg_attn_spatial = avg_attn
                
            if avg_attn_spatial.shape == phys_mask.shape:
                kmi = calculate_kmi_score(avg_attn_spatial, phys_mask)
                metrics[layer_idx]['kmi'].append(kmi)
            else:
                # Shape mismatch fallback
                metrics[layer_idx]['kmi'].append(0.0)
                
        captured_attn.clear()

    # 5. Aggregate & Analyze
    layers = sorted(metrics.keys())
    avg_entropy = [np.mean(metrics[l]['entropy']) for l in layers]
    avg_kmi = [np.mean(metrics[l]['kmi']) for l in layers]
    
    # Find Peaks
    max_ent_val = max(avg_entropy)
    max_ent_layer = layers[avg_entropy.index(max_ent_val)]
    
    max_kmi_val = max(avg_kmi)
    max_kmi_layer = layers[avg_kmi.index(max_kmi_val)]
    
    logger.info(f"Results Analysis:")
    logger.info(f"Max Entropy Layer (Switch 1 Candidate): Layer {max_ent_layer} (Val: {max_ent_val:.4f})")
    logger.info(f"Max KTI Layer (Switch 2 Candidate): Layer {max_kmi_layer} (Val: {max_kmi_val:.4f})")
    
    # 6. Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Transformer Layer')
    ax1.set_ylabel('Entropy (Global Search)', color=color)
    ax1.plot(layers, avg_entropy, color=color, linewidth=2, marker='o', label='Entropy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(x=max_ent_layer, color=color, linestyle='--', alpha=0.5, label=f'Peak Entropy (L{max_ent_layer})')
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('KTI (Structure Assembly)', color=color)
    ax2.plot(layers, avg_kmi, color=color, linewidth=2, marker='x', label='KTI')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axvline(x=max_kmi_layer, color=color, linestyle='--', alpha=0.5, label=f'Peak KTI (L{max_kmi_layer})')
    
    plt.title('NViT Paper 2: Diagnostic Analysis (Baseline ViT)')
    fig.tight_layout()
    plt.savefig('Expt1_Baseline_ViT.png')
    
    # 7. Logging to File
    with open('Expt1_Baseline_ViT.log', 'w') as f:
        f.write("Layer,Entropy,KTI\n")
        for l, e, k in zip(layers, avg_entropy, avg_kmi):
            f.write(f"{l},{e:.4f},{k:.4f}\n")
        f.write(f"# CONCLUSION\n")
        f.write(f"Layer_i (Entropy Peak) = {max_ent_layer}\n")
        f.write(f"Layer_j (KTI Peak) = {max_kmi_layer}\n")

if __name__ == "__main__":
    main()
