
import torch
import torch.nn as nn
import logging
from functools import partial

# Dependency
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("mamba_ssm not found")
            
        self.mixer = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        x = self.dropout(x)
        return x + residual

class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba: Scans Forward and Backward.
    Useful for Spatial Data (Patches).
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.forward_mamba = MambaBlock(dim, d_state, d_conv, expand, dropout)
        self.backward_mamba = MambaBlock(dim, d_state, d_conv, expand, dropout)
        self.fusion = nn.Linear(dim * 2, dim)
        
    def forward(self, x):
        # x: [B, L, D]
        
        # Forward
        out_fwd = self.forward_mamba(x)
        
        # Backward (Flip dim 1)
        x_rev = torch.flip(x, dims=[1])
        out_rev = self.backward_mamba(x_rev)
        out_rev = torch.flip(out_rev, dims=[1])
        
        # Fuse
        out = torch.cat([out_fwd, out_rev], dim=-1)
        out = self.fusion(out)
        return out

class PatchScanMamba(nn.Module):
    """
    Mamba that reorders patches before scanning to simulate "Center-Out" diffusion.
    Shape: (B, L, D) -> Reorder -> Mamba -> Restore -> (B, L, D)
    """
    def __init__(self, dim, img_size=(256, 192), patch_size=16, variant='spiral'):
        super().__init__()
        self.block = BidirectionalMambaBlock(dim)
        self.variant = variant
        
        # Precompute Scan Order (Spiral Out from Center)
        spiral_idx = self._build_spiral_index(img_size, patch_size)
        
        if variant == 'reverse_spiral':
            # Reverse: Periphery -> Center
            self.register_buffer('scan_idx', torch.flip(spiral_idx, dims=[0]))
        else:
            # Standard Spiral: Center -> Periphery
            self.register_buffer('scan_idx', spiral_idx)
            
        # Inverse index (to restore original order)
        self.register_buffer('inv_idx', torch.argsort(self.scan_idx))
        
    def _build_spiral_index(self, img_size, patch_size):
        # 1. Grid Dimensions
        H = img_size[0] // patch_size
        W = img_size[1] // patch_size
        N = H * W
        
        # 2. Coordinates
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        # Center coords
        cy, cx = (H-1)/2.0, (W-1)/2.0
        
        # 3. Distance from Center
        dist = (y - cy)**2 + (x - cx)**2
        dist_flat = dist.flatten()
        
        # 4. Sort by distance (Center -> Periphery)
        indices = torch.argsort(dist_flat)
        return indices
        
    def forward(self, x):
        # x: (B, T, D). T = N or N+1 (CLS)
        B, T, D = x.shape
        
        # Handle CLS
        has_cls = (T == self.scan_idx.shape[0] + 1)
        if has_cls:
            cls_token = x[:, 0:1, :]
            patches = x[:, 1:, :] 
        else:
            cls_token = None
            patches = x
            
        # Reorder Patches
        # Check if we are operating on Spatial Patches
        if patches.shape[1] == self.scan_idx.shape[0]:
            patches_reordered = patches[:, self.scan_idx, :]
            is_spatial = True
            
            # For joints, if we want reverse, we might need manual handling if not spatial.
            # But PatchScanMamba is mainly for spatial. 
            # If used on joints: we typically use 'seq' or 'bi'.
            # BUT if user wants 'reverse' on joints? 
            # Our current joint logic simply skips reordering (lines 116-122).
            # To support Reverse on Joints (which are 0..23 seq), we need to update this logic.
        else:
            # Fallback for Joints (Already topologically sorted 0..23)
            # If variant is reverse, we should flip them?
            if self.variant == 'reverse_spiral': # or strictly 'reverse'
                 patches_reordered = torch.flip(patches, dims=[1])
            else:
                 patches_reordered = patches
            is_spatial = False
        
        # Run Mamba
        out_patches = self.block(patches_reordered)
        
        # Restore Order
        if is_spatial:
            out_patches_restored = out_patches[:, self.inv_idx, :]
        elif self.variant == 'reverse_spiral':
             # Restore joint order if we flipped it
             out_patches_restored = torch.flip(out_patches, dims=[1])
        else:
            out_patches_restored = out_patches
        
        if cls_token is not None:
            out = torch.cat([cls_token, out_patches_restored], dim=1)
        else:
            out = out_patches_restored
            
        return out

class AnatomicalScanMamba(nn.Module):
    """
    Kinetic Mamba: Scans along 5 explicit anatomical paths.
    Paths are defined based on SMPL topology (24 joints).
    
    Path Definitions (Indices):
    - Torso: [0, 3, 6, 9, 12, 15] (Pelvis -> Spine -> Neck -> Head)
    - L-Arm: [13, 16, 18, 20, 22] (L-Collar -> L-Hand)
    - R-Arm: [14, 17, 19, 21, 23] (R-Collar -> R-Hand)
    - L-Leg: [1, 4, 7, 10] (L-Hip -> L-Foot)
    - R-Leg: [2, 5, 8, 11] (R-Hip -> R-Foot)
    
    Total Nodes Covered: 24 (All SMPL joints)
    """
    def __init__(self, dim, variant='kinetic_bi'):
        super().__init__()
        self.variant = variant
        self.dim = dim
        
        # Mamba Core (Shared or Separate? Shared weights usually better for generalization)
        self.block = BidirectionalMambaBlock(dim) if 'bi' in variant else MambaBlock(dim)
        
        # Define Paths
        self.paths = [
            [0, 3, 6, 9, 12, 15],      # Torso
            [13, 16, 18, 20, 22],      # L-Arm
            [14, 17, 19, 21, 23],      # R-Arm
            [1, 4, 7, 10],             # L-Leg
            [2, 5, 8, 11]              # R-Leg
        ]
        
        # Flattened Index for Scatter/Gather
        # We process 5 independent sequences. 
        # But Mamba expects (B, L, D). We can batch the paths?
        # Paths have different lengths (6, 5, 5, 4, 4).
        # Padding is needed if we batch them as (B*5, MaxL, D).
        # MAX_LEN = 6.
        
        self.max_len = 6
        self.num_paths = 5
        
        # Create padded indices for gathering
        # Shape: (5, 6). -1 for padding.
        pad_val = -1
        idx_matrix = torch.full((self.num_paths, self.max_len), pad_val, dtype=torch.long)
        for i, p in enumerate(self.paths):
            idx_matrix[i, :len(p)] = torch.tensor(p)
            
        self.register_buffer('path_indices', idx_matrix)

        # Reverse indices for 'kinetic_in' (Inward Flow)
        # If 'kinetic_bi', BidirectionalMambaBlock handles the reverse internally on the sequence provided.
        # But 'kinetic_in' implies the primary sequence direction is reversed (Leaf -> Root).
        if 'in' in variant and 'bi' not in variant:
             # Reverse the path definitions
             # We can just flip the gathering logic or flip the data after gathering
             pass # Handled in forward

    def forward(self, x):
        # x: (B, J, D). J=24.
        B, J, D = x.shape
        
        # 1. Gather into Paths: (B, 5, 6, D)
        # We need to flatten B and 5 -> (B*5, 6, D)
        
        # Expand indices for Batch: (B, 5, 6)
        # Actually, simpler to just index x directly.
        
        # Prepare valid mask
        valid_mask = (self.path_indices != -1) # (5, 6)
        
        # Gather
        # Create a zero tensor (B, 5, 6, D)
        path_data = torch.zeros(B, self.num_paths, self.max_len, D, device=x.device, dtype=x.dtype)
        
        for i in range(self.num_paths):
            valid_idx = self.path_indices[i][valid_mask[i]] # (L_i,)
            # Slice: (B, L_i, D)
            extracted = x[:, valid_idx, :]
            
            if 'in' in self.variant and 'bi' not in self.variant:
                # Reverse flow (Leaf -> Root)
                extracted = torch.flip(extracted, dims=[1])
                
            path_data[:, i, :len(valid_idx), :] = extracted

        # Reshape for Mamba: (B*5, 6, D)
        mamba_in = path_data.view(B * self.num_paths, self.max_len, D)
        
        # 2. Run Mamba
        # Mamba handles fixed length. Padding should be masked?
        # Mamba doesn't natively support padding mask in standard implementation usually.
        # But given small length (6), impacts are minimal.
        mamba_out = self.block(mamba_in) # (B*5, 6, D)
        
        # 3. Scatter back to Joints (Fusion)
        # Since joints appear in only ONE path (Disjoint set), we can just scatter.
        # If paths overlapped, we would need to sum/mean. 
        # Our paths are disjoint partitions of 24 joints.
        
        mamba_out = mamba_out.view(B, self.num_paths, self.max_len, D)
        y = torch.zeros_like(x) # (B, 24, D)
        
        for i in range(self.num_paths):
            valid_idx = self.path_indices[i][valid_mask[i]]
            processed = mamba_out[:, i, :len(valid_idx), :]
            
            if 'in' in self.variant and 'bi' not in self.variant:
                # Restore order (Root -> Leaf for storage)
                processed = torch.flip(processed, dims=[1])
            
            # Place back
            y[:, valid_idx, :] = processed.to(y.dtype)
            
        return x + y # Residual Connection
