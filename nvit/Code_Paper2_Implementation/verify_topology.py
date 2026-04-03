#!/home/yangz/.conda/envs/4D-humans/bin/python

import torch
import sys
import logging
import numpy as np
from pathlib import Path

# Setup Path
ROOT = '/home/yangz/NViT-master/nvit/Code_Paper2_Implementation'
sys.path.append(ROOT)

from nvit2_models.mamba_utils import PatchScanMamba
from smpl_topology import get_smpl_adjacency_matrix

def verify_mamba_variants():
    print(">>> Verifying Mamba Variants...")
    
    # 1. Standard Spiral (Forward)
    mamba_fwd = PatchScanMamba(dim=128, img_size=(32,32), patch_size=16, variant='spiral')
    idx_fwd = mamba_fwd.scan_idx
    print(f"Spiral Index Shape: {idx_fwd.shape}")
    print(f"Spiral First 5: {idx_fwd[:5]}")
    print(f"Spiral Last 5: {idx_fwd[-5:]}")
    
    # Check if center is first (Indices for 2x2 grid: 0,1,2,3. Center is approx 1,2?)
    # 2x2 grid: (0,0), (0,1), (1,0), (1,1). Center (0.5, 0.5).
    # All dists are equal (0.5^2+0.5^2 = 0.5). Sort might be stable.
    
    # 2. Reverse Spiral (Backward)
    mamba_rev = PatchScanMamba(dim=128, img_size=(32,32), patch_size=16, variant='reverse_spiral')
    idx_rev = mamba_rev.scan_idx
    print(f"Reverse Index Shape: {idx_rev.shape}")
    print(f"Reverse First 5: {idx_rev[:5]}")
    print(f"Reverse Last 5: {idx_rev[-5:]}")
    
    # Assertion: Reverse should be flip of Forward
    assert torch.equal(idx_rev, torch.flip(idx_fwd, dims=[0])), "FAIL: Reverse index is not flipped version of Spiral!"
    print("✅ Mamba Spiral/Reverse Logic Verified.")

def verify_gcn_variants():
    print("\n>>> Verifying GCN Variants...")
    
    # 1. Outward (Parent -> Child)
    adj_out = get_smpl_adjacency_matrix(mode='outward', add_self_loops=False)
    # Check Pelvis -> Left Hip
    # Names: pelvis(0), left_hip(1). Parent: left_hip is pelvis.
    # Edge: Pelvis(0) -> Left Hip(1). Adj[0, 1] should be 1.
    print(f"Outward: Adj[Pelvis, L_Hip] = {adj_out[0, 1]}")
    print(f"Outward: Adj[L_Hip, Pelvis] = {adj_out[1, 0]}")
    
    assert adj_out[0, 1] == 1.0, "FAIL: Outward should have Edge 0->1"
    assert adj_out[1, 0] == 0.0, "FAIL: Outward should NOT have Edge 1->0"
    
    # 2. Inward (Child -> Parent)
    adj_in = get_smpl_adjacency_matrix(mode='inward', add_self_loops=False)
    print(f"Inward: Adj[Pelvis, L_Hip] = {adj_in[0, 1]}")
    print(f"Inward: Adj[L_Hip, Pelvis] = {adj_in[1, 0]}")
    
    assert adj_in[0, 1] == 0.0, "FAIL: Inward should NOT have Edge 0->1"
    assert adj_in[1, 0] == 1.0, "FAIL: Inward should have Edge 1->0"
    
    # 3. Transpose Check
    assert torch.equal(adj_out.T, adj_in), "FAIL: Outward should be Transpose of Inward"
    print("✅ GCN Outward/Inward Logic Verified.")

if __name__ == "__main__":
    try:
        verify_mamba_variants()
        verify_gcn_variants()
        print("\n🎉 ALL TOPOLOGY TESTS PASSED.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
