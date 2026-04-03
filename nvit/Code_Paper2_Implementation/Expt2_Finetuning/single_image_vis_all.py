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

# Setup paths
ROOT = '/home/yangz/NViT-master/nvit/Paper1_Diagnostics'
sys.path.append(ROOT)

# Import from diagnostic core
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
    # Use the default checkpoint logic or specific path
    from hmr2.models import DEFAULT_CHECKPOINT
    ckpt_path = DEFAULT_CHECKPOINT
    
    # Check if we should use the finetuned model?
    # User asked for "same picture, all layers gaze".  Usually implies the base model or the current one.
    # Given we are in the middle of fine-tuning, maybe they want to see the base model first to compare?
    # Or the user just wants to see "all layers". Let's stick to HMR2 baseline for diagnostic clarity 
    # as the fine-tuned model (IndependentSurgicalModel) might not be compatible with the Wrapper yet
    # without modifications. The user context implies generalized interest.
    # I'll use the BASE HMR2 for stable visualization.
    
    model, _ = load_hmr2(ckpt_path)
    model.eval()

    # 2. Setup Diagnostic Lab
    print("[2/4] Setting up Diagnostic Lab...")
    # The wrapper needs to match the model type. HMR2 wrapper is standard.
    wrapper = get_wrapper(model, 'hmr2')
    lab = ViTDiagnosticLab(wrapper, model_name='hmr2', output_root='results_vis')
    
    # 3. Load Image
    print("[3/4] Loading an image manually...")
    import glob
    img_dir = '/home/yangz/4D-Humans/data/3DPW' # Try 3DPW first
    candidates = glob.glob(os.path.join(img_dir, '**/*.jpg'), recursive=True)
    if not candidates:
        # Fallback to demo image if 3DPW not found
        img_dir = '/home/yangz/4D-Humans/example_data/images' 
        candidates = glob.glob(os.path.join(img_dir, '*.png'), recursive=True)
        if not candidates:
             candidates = glob.glob(os.path.join(img_dir, '*.jpg'), recursive=True)
             
    if not candidates:
        print("No images found.")
        return
        
    img_path = candidates[0] # Pick the first one
    print(f"Using image: {img_path}")
    
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print(f"Failed to read {img_path}")
        return
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # HMR2 Preprocessing
    # Model internally crops [:, :, :, 32:-32]
    # Input 256x256 -> Effective 192x256 -> 12x16 patches
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
        
    # 5. Visualize ALL Layers
    print("Generating Gaze Visualization (All Layers)...")
    if not hasattr(lab, 'last_attention_maps'):
        print("Error: No attention maps captured.")
        return

    # Determine number of layers
    available_layers = sorted(lab.last_attention_maps.keys())
    print(f"Captured layers: {available_layers}")
    
    # Setup Grid
    num_layers = len(available_layers)
    cols = 8
    rows = (num_layers + 1) // cols + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(24, 4 * rows))
    axes = axes.flatten()
    
    # Subplot 0: Original
    axes[0].imshow(img_resized)
    axes[0].set_title("Original Input")
    axes[0].axis('off')
    
    for i, layer_idx in enumerate(available_layers):
        ax = axes[i + 1]
        
        attn = lab.last_attention_maps[layer_idx] # (B, H, N, N)
        # Average heads
        attn_avg = attn.mean(dim=1).squeeze(0) # (N, N)
        N = attn_avg.shape[0]
        
        # Calculate Saliency (Gaze) = Sum of attention received by each token
        # or Attention Rollout?
        # User asked for "Gaze results" (凝视结果).
        # Simply summing columns roughly gives "where the model looks".
        saliency = attn_avg.sum(dim=0) # (N)
        
        grid_h, grid_w = 16, 12 # 256/16=16, 192/16=12 for ViT-H
        # Note: HMR2 crop is center 192 (width) x 256 (height).
        # Patches are 16x16.
        # 192/16 = 12 (W), 256/16 = 16 (H). Total 192 patches.
        
        if N != grid_h * grid_w:
             # Fallback square
             side = int(np.sqrt(N))
             grid_h, grid_w = side, side
             
        try:
            attn_map = saliency.reshape(grid_h, grid_w).float().numpy()
        except Exception as e:
            print(f"Reshape error layer {layer_idx}: {e}")
            continue
            
        # Resize to match input image context (256x256 with pads)
        # The attention is on the 192x256 crop.
        # We need to visualize it on that crop.
        
        # 1. Resize heatmap to (192, 256) (W, H)
        # cv2.resize takes (W, H)
        heatmap_crop = cv2.resize(attn_map, (192, 256))
        
        # 2. Normalize
        heatmap_norm = (heatmap_crop - heatmap_crop.min()) / (heatmap_crop.max() - heatmap_crop.min() + 1e-8)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # 3. Pad back to 256x256 (32 pixels left/right)
        # HMR2 logic: center crop width 192 from 256.
        # So we pad width.
        pad_width = (256 - 192) // 2
        heatmap_full = np.pad(heatmap_color, ((0, 0), (pad_width, pad_width), (0, 0)), mode='constant')
        
        # Overlay
        alpha = 0.6
        if heatmap_full.shape == img_resized.shape:
            overlay = cv2.addWeighted(img_resized, 1-alpha, heatmap_full, alpha, 0)
            ax.imshow(overlay)
            
            # Add metrics if available
            ent = lab.layer_metrics[layer_idx]['entropy'][0] if layer_idx in lab.layer_metrics else 0
            ax.set_title(f"L{layer_idx}\nEnt:{ent:.2f}")
        else:
            ax.imshow(heatmap_full)
            
        ax.axis('off')
        
    # Hide empty subplots
    for i in range(len(available_layers) + 1, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    # Save to known artifact location
    save_path = '/home/yangz/.gemini/antigravity/brain/2582eb19-d288-4dcd-aacc-660f78c5074a/all_layers_gaze.png'
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    run_vis()
