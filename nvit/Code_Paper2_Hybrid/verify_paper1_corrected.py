#!/home/yangz/.conda/envs/4D-humans/bin/python

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Import local modules
from experiment_1_diagnostics import ViTDiagnosticProbe, create_patch_adjacency
from calculate_kti import calculate_kti

# --- MOCK MODEL DEFINITIONS (To avoid complex dependencies) ---
class MockBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.attn_drop = nn.Identity()

class MockViT(nn.Module):
    def __init__(self):
        super().__init__()
        # 12 Layers for ViT-Huge/Base
        self.blocks = nn.ModuleList([MockBlock() for _ in range(12)])
        
    def forward(self, x):
        return x # Pass through

def run_comparison():
    print("="*80)
    print("PAPER 1 METRIC VERIFICATION: Impact of Systematic Error")
    print("="*80)
    
    # 1. Setup Mock Model
    print("[1] Setting up HMR2-like Model (Mock)...")
    model = MockViT()
    probe = ViTDiagnosticProbe(model) # Hook it
    
    # 2. Simulate Data
    # Synthetic "Pelvis" to "Head" line
    # Image coords for 256x192
    gt_joints_correct = torch.zeros(1, 24, 2)
    gt_joints_correct[:, 0] = torch.tensor([96.0, 128.0]) # Pelvis center
    gt_joints_correct[:, 15] = torch.tensor([96.0, 50.0]) # Head top
    
    # 3. Scenario A: CORRECT (256x192)
    print("\n[Scenario A] Correct Implementation (256x192)")
    input_correct = torch.randn(1, 3, 256, 192)
    
    # Run Diagnosis (Hooks get registered)
    probe.run_diagnosis(input_correct, gt_joints=gt_joints_correct)
    
    # Inject Fake "Good" Attention (Diagonal) to simulate a working model
    # We want to see if the Metric *captures* it correctly.
    # We need to inject into Layer 5 (Arbitrary choice)
    # Patches: 192x256 image -> 16x16 patch size -> 12x16 grid = 192 patches.
    # Grid Width = 12, Grid Height = 16.
    
    # Pelvis Patch Index (96, 128):
    # x=96 -> col = 96//16 = 6
    # y=128 -> row = 128//16 = 8
    # idx = row * W + col = 8*12 + 6 = 96+6 = 102.
    
    # Head Patch Index (96, 50):
    # x=96 -> col = 6
    # y=50 -> row = 50//16 = 3
    # idx = 3*12 + 6 = 36+6 = 42.
    
    fake_attn = torch.zeros(1, 12, 192+1, 192+1) # +1 for CLS
    # Connect Pelvis (103) to Head (43) (Shifted by 1 for CLS)
    # Note: DiagnosticProbe handles CLS removal ([:, :, 1:, 1:])
    fake_attn[:, :, 102+1, 42+1] = 1.0 
    
    # Inject into probe
    probe.attention_maps[5] = fake_attn
    
    print(" -> Computing Metrics...")
    probe.compute_metrics()
    
    # 4. Scenario B: BUGGED (256x256)
    print("\n[Scenario B] The 'Systematic Error' Simulation (256x256)")
    
    # Scenario: The input was 256x256. 
    # The Model processed it (resulting in 16x16=256 patches).
    # But the Ground Truth Joints were derived from the original aspect ratio (or rescaled?)
    
    # If we feed 256x256, the Grid is 16x16 = 256 patches.
    # W=16, H=16.
    
    # The KTI function will generate an Adjacency Matrix for 16x16 Grid.
    # BUT, if the user didn't correct the Joints, the Joints might map to wrong patches.
    
    # Let's perform the "Geometric Check": 
    # Compare "Physics Mask (Correct)" vs "Physics Mask (Squashed)".
    
    # Correct Mask (16x12 Grid)
    mask_correct = create_patch_adjacency(gt_joints_correct[0], patch_size=16, img_w=192, img_h=256)
    
    # Warped Mask (16x12 Grid, but Joints Warped)
    # Simulate: User resized 256x256 -> 256x192 without aspect ratio preservation
    # X coords compressed by 0.75.
    gt_joints_warped = gt_joints_correct.clone()
    gt_joints_warped[:, 0] *= 0.75 
    
    mask_warped = create_patch_adjacency(gt_joints_warped[0], patch_size=16, img_w=192, img_h=256)
    
    # Calculate Jaccard Overlap
    intersection = (mask_correct * mask_warped).sum()
    union = mask_correct.sum() + mask_warped.sum() - intersection
    iou = intersection / (union + 1e-9)
    
    print(f"\n[Geometric Distortion Check]")
    print(f"Correct Mask Sum: {mask_correct.sum().item()}")
    print(f"Warped Mask Sum:  {mask_warped.sum().item()}")
    print(f"Intersection:     {intersection.item()}")
    print(f"IoU (Overlap):    {iou.item()*100:.2f}%")
    
    print("\n[Verdict]")
    if iou < 0.95:
        print("FAIL: The Geometric Distortion is Significant (>5%). Paper 1 KTI is Invalid.")
    else:
        print("PASS: Distortion is negligible.")

if __name__ == "__main__":
    run_comparison()
