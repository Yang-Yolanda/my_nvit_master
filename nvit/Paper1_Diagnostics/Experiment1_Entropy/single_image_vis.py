#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import sys
import os
import argparse
from pathlib import Path
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Ensure we can import from the parent diagnostic_core
sys.path.append(str(Path(__file__).resolve().parent.parent))
from diagnostic_core.diagnostic_engine import ViTDiagnosticLab, get_wrapper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_vis():
    try:
        from hmr2.models import load_hmr2
    except ImportError:
        print("Error: HMR2/4D-Humans dependencies not found.")
        return

    # 1. Load Model (HMR2)
    print("\n[1/4] Loading HMR2 Model...")
    ckpt_path = '/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
    if not os.path.exists(ckpt_path):
        import glob
        candidates = glob.glob('/home/yangz/.cache/4DHumans/**/epoch=35*.ckpt', recursive=True)
        if candidates:
            ckpt_path = candidates[0]
        else:
            print("Checkpoint not found!")
            return

    model, _ = load_hmr2(ckpt_path)
    model.eval()

    # 2. Setup Diagnostic Lab
    print("[2/4] Setting up Diagnostic Lab...")
    wrapper = get_wrapper(model, 'hmr2')
    lab = ViTDiagnosticLab(wrapper, model_name='hmr2', output_root='results_vis')
    
    # 3. Load Image
    print("[3/4] Loading an image manually...")
    import glob
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    candidates = glob.glob(os.path.join(img_dir, '**/*.jpg'), recursive=True)
    if not candidates:
        print("No images found in 3DPW.")
        return
    img_path = candidates[0] # Same image as before ideally
    print(f"Using image: {img_path}")
    
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # HMR2 Preprocessing
    # Model internally crops [:, :, :, 32:-32]
    # So input must be 256x256 -> Effective 192x256 -> 12x16 patches = 192 tokens.
    input_size = 256
    img_resized = cv2.resize(img_cv, (input_size, input_size))
    
    # Normalize
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1) # C, H, W
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_norm = (img_tensor - mean) / std
    
    batch = {'img': img_norm.unsqueeze(0)}
    
    # Move to CPU
    batch = wrapper.to_device(batch, 'cpu')
    model.to('cpu')
    
    # 4. Run Inference
    print("[4/4] Running Inference & Capturing Attention...")
    with torch.no_grad():
        _ = wrapper(batch)
        
    # 5. Visualize
    print("Generating Gaze Visualization...")
    if not hasattr(lab, 'last_attention_maps'):
        print("Error: No attention maps captured.")
        return

    # Select Layers to Visualize
    target_layers = list(range(10)) 
    
    rows, cols = 2, 6
    fig, axes = plt.subplots(rows, cols, figsize=(24, 10))
    axes = axes.flatten()
    
    axes[0].imshow(img_resized)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    for i, layer_idx in enumerate(target_layers):
        ax_idx = i + 1
        if ax_idx >= len(axes): break
        ax = axes[ax_idx]
        
        if layer_idx in lab.last_attention_maps:
            attn = lab.last_attention_maps[layer_idx] # (B, H, N, N)
            
            # Average over heads
            attn_avg = attn.mean(dim=1).squeeze(0) # (N, N)
            
            N = attn_avg.shape[0]
            
            if N == 192:
                # Spatial Only (12x16)
                saliency = attn_avg.sum(dim=0) # (N)
                grid_h, grid_w = 16, 12 # 256/16, 192/16
            else:
                 # Fallback
                 saliency = attn_avg.sum(dim=0)
                 grid_size = int(np.sqrt(N))
                 grid_h, grid_w = grid_size, grid_size

            try:
                attn_map = saliency.reshape(grid_h, grid_w).float().numpy()
            except:
                print(f"Reshape failed for N={N}. Skipping layer {layer_idx}")
                continue
            
            # Resize Logic
            # The heat map represents 192x256 (Center Crop)
            # We need to pad it to 256x256 to overlay on original image
            
            target_h, target_w = 256, 192 # Height 256, Width 192
            attn_map_resized = cv2.resize(attn_map, (target_w, target_h))
            
            # Normalize
            attn_map_norm = (attn_map_resized - attn_map_resized.min()) / (attn_map_resized.max() - attn_map_resized.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_norm), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Pad: 32 left, 32 right
            # np.pad(array, ((top, bottom), (left, right), (channels, channels)))
            heatmap_padded = np.pad(heatmap, ((0, 0), (32, 32), (0, 0)), mode='constant', constant_values=0)
            
            # Verify shape
            if heatmap_padded.shape != img_resized.shape:
                heatmap_padded = cv2.resize(heatmap_padded, (input_size, input_size))
            
            alpha = 0.6
            overlay = cv2.addWeighted(img_resized, 1-alpha, heatmap_padded, alpha, 0)
            
            ax.imshow(overlay)
            ent = lab.layer_metrics[layer_idx]['entropy'][0]
            ax.set_title(f"Layer {layer_idx}\nEnt: {ent:.2f}")
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        ax.axis('off')

    for i in range(len(target_layers) + 1, len(axes)):
        axes[i].axis('off')
            
    plt.tight_layout()
    save_path = '/home/yangz/.gemini/antigravity/brain/ffe452f4-96f8-4a35-946b-54daf1f904f3/attention_gaze.png'
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    run_vis()
