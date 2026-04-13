import torch
import torch.nn.functional as F
import numpy as np

def verify_sampling_identity():
    """
    Undermind Rule 1.2: Identity Sanity Test for grid_sample logic.
    Verifies that (x,y) normalization mapping recovering coordinates correctly.
    """
    W, H = 192, 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Create Identity Feature Map (B, C=2, H, W)
    # Channel 0: X-coord [0, W-1] scaled to [0, 1]
    # Channel 1: Y-coord [0, H-1] scaled to [0, 1]
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    # Feature map where values = indices (normalized to [0, 1])
    fmap = torch.stack([xx.float() / (W - 1), yy.float() / (H - 1)], dim=0).unsqueeze(0) # (1, 2, 256, 192)
    
    # 2. Define Test Points in Pixel Space
    test_pixels = torch.tensor([
        [0.0, 0.0],           # Top-Left
        [W-1, 0.0],           # Top-Right
        [0.0, H-1],           # Bottom-Left
        [W-1, H-1],           # Bottom-Right
        [W//2, H//2],         # Center
    ], device=device)
    
    # 3. Apply Normalization used in nvit_hybrid.py / guided_hmr2.py
    # Logic: grid = (pixel / (dim-1)) * 2 - 1
    grid_x = (test_pixels[:, 0] / (W - 1)) * 2.0 - 1.0
    grid_y = (test_pixels[:, 1] / (H - 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0) # (1, 1, N, 2)
    
    # 4. Sample
    # align_corners=True is critical to match the grid = 2*p/(d-1) - 1 logic
    sampled = F.grid_sample(fmap, grid, align_corners=True, mode='bilinear') # (1, 2, 1, N)
    sampled = sampled.squeeze().permute(1, 0) # (N, 2)
    
    # 5. Recovery
    # Multiply by (dim-1) to get back pixels
    recovered_pixels_x = sampled[:, 0] * (W - 1)
    recovered_pixels_y = sampled[:, 1] * (H - 1)
    recovered_pixels = torch.stack([recovered_pixels_x, recovered_pixels_y], dim=-1)
    
    print(f"--- Sampling Identity Test (W={W}, H={H}) ---")
    print(f"Target Pixels:\n{test_pixels}")
    print(f"Recovered Pixels:\n{recovered_pixels}")
    
    error = torch.abs(test_pixels - recovered_pixels).max().item()
    print(f"Max Reconstruction Error: {error:.6f}")
    
    if error < 1e-4:
        print("\n✅ Identity Invariant A: PASSED (Axis and Normalization are correct)")
    else:
        print("\n❌ Identity Invariant A: FAILED")

def verify_hmr2_to_grid_mapping():
    """
    Verifies the mapping from HMR2 [-0.5, 0.5] space to grid_sample [-1, 1] space.
    """
    # 256x256 original image, 256x192 crop (center)
    # Pixel 32 in 256 space is Pixel 0 in 192 space.
    # Pixel 128 (center) in 256 space is Pixel 96 in 192 space.
    
    # HMR2 Normalization: pixel_x / 256 - 0.5
    # Pixel 32 -> 32/256 - 0.5 = 0.125 - 0.5 = -0.375
    # Pixel 224 -> 224/256 - 0.5 = 0.875 - 0.5 = 0.375
    # Pixel 128 -> 128/256 - 0.5 = 0
    
    hmr2_x = torch.tensor([-0.375, 0.0, 0.375, -0.5, 0.5])
    
    # Logic: grid_x = hmr2_x * (256/192) * 2.0
    grid_x = hmr2_x * (256.0 / 192.0) * 2.0
    
    print(f"\n--- HMR2 [-0.5, 0.5] -> 192px Grid Mapping ---")
    points = ["Left Edge (p32)", "Center (p128)", "Right Edge (p224)", "Out of Crop (p0)", "Out of Crop (p256)"]
    for i, p in enumerate(points):
        print(f" {p}: {hmr2_x[i]:.3f} -> {grid_x[i]:.2f}")

if __name__ == "__main__":
    verify_sampling_identity()
    verify_hmr2_to_grid_mapping()
