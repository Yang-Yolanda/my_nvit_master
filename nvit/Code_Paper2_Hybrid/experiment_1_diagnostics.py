#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import math
import sys
import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# Robust Imports
try:
    from calculate_kti import calculate_kti
except ImportError:
    print("Warning: calculate_kti missing.")
    def calculate_kti(a,b): return 0.0, 0.0

try:
    from model_manager import ModelManager
    from smpl_topology import get_smpl_adjacency_matrix, SMPL_PARENTS, SMPL_JOINT_NAMES
except ImportError:
    # Dummy mocks if dependencies missing locally
    print("Warning: Dependencies missing locally (ModelManager/SMPL), using mocks.")
    SMPL_JOINT_NAMES = ['pelvis'] * 24 
    SMPL_PARENTS = {}

def get_line_patches(start_idx, end_idx, grid_w, grid_h):
    """
    Bresenham's Line Algorithm to find all patches between two joint patches.
    """
    x0, y0 = start_idx % grid_w, start_idx // grid_w
    x1, y1 = end_idx % grid_w, end_idx // grid_w
    
    patches = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        if 0 <= x0 < grid_w and 0 <= y0 < grid_h:
            patches.append(y0 * grid_w + x0)
            
        if x0 == x1 and y0 == y1:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
            
    return patches

def create_patch_adjacency(joints_2d, patch_size=16, img_w=192, img_h=256):
    """
    Generates a dense physical prior mask G.
    joints_2d: [24, 2] coordinates (x, y)
    """
    grid_h = img_h // patch_size
    grid_w = img_w // patch_size
    N = grid_h * grid_w

    joint_to_patch = {}

    # 1. Map joints to patch indices
    for i in range(len(joints_2d)):
        u, v = joints_2d[i]
        u = max(0, min(img_w-1, u)) # Clip
        v = max(0, min(img_h-1, v))
        row = int(v // patch_size)
        col = int(u // patch_size)
        idx = row * grid_w + col
        joint_to_patch[i] = int(idx)

    # 2. Draw Limbs (Connectivity)
    mask = torch.zeros((N, N))
    parents = SMPL_PARENTS
    name_to_idx = {name: i for i, name in enumerate(SMPL_JOINT_NAMES)}

    # Build the set of "Physical Patches"
    # Ideally, we want G[i, j] = 1 if patch i and patch j are "connected"
    # But KTI definition is often: Does Attention map align with physical structure?
    # So we often define G as a "Valid Region" mask.
    # Here we define G[i, j] = 1 if a direct limb exists between patch i and patch j?
    # Or simpler: Is patch j reachable from patch i via 1 hop on the skeleton?
    
    # Revised Approach:
    # A dense mask where G[i, j] = 1 if BOTH i and j are on the skeleton structure
    # AND they are close/connected. 
    # For simplicity in this experiment, we simply mark all "Skeleton Patches" as valid targets.
    # And then compute overlap.
    
    # Let's stick to the Adjacency definition:
    # A[i, j] = 1 if patch i and patch j are functionally connected.
    
    # First, collect all patches that form the skeleton (Rasterization)
    skeleton_patches = set()
    
    for child, parent in parents.items():
        if parent is None: continue
        c_idx = name_to_idx[child]
        p_idx = name_to_idx[parent]
        
        patch_c = joint_to_patch[c_idx]
        patch_p = joint_to_patch[p_idx]
        
        if patch_c < N and patch_p < N:
            # Rasterize line
            line_patches = get_line_patches(patch_c, patch_p, grid_w, grid_h)
            skeleton_patches.update(line_patches)
            
            # Connect them in the adjacency matrix (simplified: Clique or Chain)
            # Connecting every patch on the limb to every other patch on the limb
            for p1 in line_patches:
                for p2 in line_patches:
                    mask[p1, p2] = 1.0
                    
    # Also ensure joints themselves are self-connected
    for p in skeleton_patches:
        mask[p, p] = 1.0

    return mask

class ViTDiagnosticProbe:
    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self.features = {}
        self.hooks = []
        self._register_hooks()
        self.gt_joints = None # Store dynamic joints

    def _register_hooks(self):
        print("[System] Registering Diagnostic Hooks...")
        if hasattr(self.model, 'blocks'):
            for i, block in enumerate(self.model.blocks):
                if hasattr(block, 'attn') and hasattr(block.attn, 'attn_drop'):
                    self.hooks.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook(i)))
                self.hooks.append(block.register_forward_hook(self._get_feat_hook(i)))
            print(f" -> Hooked {len(self.model.blocks)} layers.")

    def _get_attn_hook(self, layer_idx):
        def hook(module, input, output):
            with torch.no_grad():
                self.attention_maps[layer_idx] = input[0].detach().cpu()
        return hook

    def _get_feat_hook(self, layer_idx):
        def hook(module, input, output):
            with torch.no_grad():
                self.features[layer_idx] = output.detach().cpu()
        return hook

    def run_diagnosis(self, x, gt_joints=None):
        """
        x: [B, 3, H, W] input image
        gt_joints: [B, 24, 2] Ground Truth 2D joints (Original Image Coordinates)
        """
        self.attention_maps = {}
        self.features = {}
        self.gt_joints = gt_joints 
        self.model.eval()
        with torch.no_grad():
            self.model(x)

    def compute_metrics(self):
        print("\n" + "="*90)
        print("DIAGNOSTIC REPORT (Paper 1 Metrics) - with DYNAMIC PATCH-KTI")
        print("="*90)
        print(f"{'Layer':<6} | {'Entropy':<10} | {'Effective Rank':<15} | {'KTI (Struct)':<12} | {'Status'}")
        print("-" * 95)

        sample_layer = list(self.attention_maps.keys())[0]
        sample_attn = self.attention_maps[sample_layer] # [B, H, N, N]
        SeqLen = sample_attn.shape[-1]
        
        # Determine Grid Size
        img_h, img_w = 256, 192 # Default HMR2
        if SeqLen == 197: # 14x14+1 -> 224x224
             img_h, img_w = 224, 224
        
        # Prepare Mask from GT Joints
        mask = None
        if self.gt_joints is not None:
             # Take the first sample in batch for reporting
             joints_sample = self.gt_joints[0].cpu().numpy() # [24, 2]
             mask = create_patch_adjacency(joints_sample, patch_size=16, img_w=img_w, img_h=img_h)
        else:
             print("[Warning] No GT Joints provided. KTI will be skipped.")

        for i in sorted(self.attention_maps.keys()):
            attn = self.attention_maps[i] # [B, H, N, N]
            
            # Handle CLS token removal for KTI
            if SeqLen in [197, 193]:
                attn_patch = attn[:, :, 1:, 1:]
            else:
                attn_patch = attn

            # 1. Entropy
            avg_attn_all = attn_patch.mean(dim=(0, 1)) # [N, N]
            entropy = -torch.sum(avg_attn_all * torch.log(avg_attn_all + 1e-9), dim=-1).mean().item()

            # 2. KTI
            kti_score = -2.0
            if mask is not None and attn_patch.shape[-1] == mask.shape[0]:
                 score, _ = calculate_kti(attn_patch, mask)
                 kti_score = score

            # 3. Effective Rank
            rank = -1.0
            if i in self.features:
                feat = self.features[i] # [B, N, C]
                feat_flat = feat.view(-1, feat.shape[-1])
                try:
                    _, S, _ = torch.svd(feat_flat.float())
                    S_norm = S / S.sum()
                    rank = torch.exp(-torch.sum(S_norm * torch.log(S_norm + 1e-9))).item()
                except:
                    pass

            status = "Healthy"
            if entropy > 4.5: status = "ADHD"
            if rank < 16.0: status = "COLLAPSED"

            print(f"{i:02d}     | {entropy:.4f}     | {rank:.2f}            | {kti_score:.4f}       | {status}")

# Mock Model and Manager for Local Testing
class MockBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.attn_drop = nn.Identity() # Dummy

class MockViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([MockBlock() for _ in range(2)]) # 2 Mock Layers
        
    def forward(self, x):
        return x

class MockModelManager:
    def __init__(self, config):
        self.model = MockViT()
        self.model.backbone = self.model # Simplified structure

    def load_model(self):
        pass

if __name__ == "__main__":
    print("[Setup] Initializing (Mock) ModelManager...")
    config = {'device': 'cpu'} # Force CPU for local test
    
    try:
        # Use Mock instead of real ModelManager
        mm = MockModelManager(config)
        mm.load_model()
        
        if mm.model is not None:
            probe = ViTDiagnosticProbe(mm.model.backbone) # Hook our mock model
            
            print("[Run] Injecting Random Noise (256x192)...")
            x = torch.randn(1, 3, 256, 192) # CPU tensor
            
            # Create Mock Joints (Simulating a simple T-Pose)
            mock_joints = torch.zeros(1, 24, 2)
            # Pelvis at 96, 128
            mock_joints[:, 0, :] = torch.tensor([96.0, 128.0]) 
            # Head at 96, 50 (Vertical line pelvis->head)
            mock_joints[:, 15, :] = torch.tensor([96.0, 50.0])
            
            print(f"Passing Mock Joints: {mock_joints.shape}")
            
            # Manually trigger hooks because our Mock model doesn't actually run attention logic
            # We inject fake attention maps for the probe to read
            probe.run_diagnosis(x, gt_joints=mock_joints)
            
            # Inject fake attention: Random vs Diagonal
            # 1. Random (should have Low KTI)
            probe.attention_maps[0] = torch.rand(1, 8, 192, 192) # Random
            
            # 2. Perfect Diagonal (should have High KTI on self-loops but maybe low on structure?)
            # Let's inject a "Line-Like" attention for Layer 1
            # We know Pelvis(patch ~134) connects to Head(patch ~38)
            # Let's say patch 134 attends strongly to patch 38
            fake_attn_structured = torch.zeros(1, 8, 192, 192)
            # Pelvis Patch ~ (128//16 * 12 + 96//16) = 8*12 + 6 = 102
            # Head Patch ~ (50//16 * 12 + 96//16) = 3*12 + 6 = 42
            p_pelvis = 102
            p_head = 42
            fake_attn_structured[:, :, p_pelvis, p_head] = 1.0 # Strong Attention
            probe.attention_maps[1] = fake_attn_structured
            
            probe.compute_metrics()
            print("\n[Success] Diagnosis Complete.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Error] {e}")
