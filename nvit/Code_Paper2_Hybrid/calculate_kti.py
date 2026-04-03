#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import torch.nn.functional as F
import numpy as np
import math
from smpl_topology import get_geodesic_distance_matrix

def calculate_kti(attention_map, distance_matrix, sigma=2.0):
    """
    Calculates Kinematic Topology Interaction (KTI) score using Matrix Comparison.
    Suitable for General Kinematic Topology Interaction evaluation.
    
    Logic:
    T_ij = exp(-D_ij^2 / sigma^2)
    KTI = CosineSimilarity(Attention, T)
    
    Args:
        attention_map (torch.Tensor): [B, H, N, N] Attention weights (should be Softmaxed)
        distance_matrix (torch.Tensor): [N, N] Geodesic distances or equivalent
        sigma (float): Width of the Gaussian kernel (controlling 'reach').
        
    Returns:
        kti_score (float): Mean Cosine Similarity across heads and batches.
    """
    # 1. Generate Soft Target (Ideal Topology Distribution)
    # [N, N]
    target_dist = torch.exp(-(distance_matrix**2) / (sigma**2))
    
    # 2. Comparison
    B, H, N, _ = attention_map.shape
    device = attention_map.device
    target = target_dist.to(device).unsqueeze(0).unsqueeze(0).expand(B, H, N, N)
    
    # Flatten across space only, keep B*H
    attn_flat = attention_map.reshape(B*H, -1)
    target_flat = target.reshape(B*H, -1)
    
    # Metric: Cosine Similarity (Rewards shape alignment, robust to magnitude)
    kti_score = F.cosine_similarity(attn_flat, target_flat, dim=-1).mean().item()
    
    return kti_score

def calculate_patch_kti(attn_map, keypoints_2d, patch_size=16, img_size=256, sigma=2.0):
    """
    Compute Soft Physically Grounded KTI for Patch-based ViTs.
    
    Logic:
    1. Map Joint Positions to nearest Patch Tokens.
    2. Construct a Token-to-Token Distance Matrix based on the Geodesic distance 
       of the joints they represent.
    3. Compute KTI using Gaussian Decay Target.
    """
    B, H, N, _ = attn_map.shape
    device = attn_map.device
    
    # Load SMPL Distances
    dist_matrix = get_geodesic_distance_matrix(directed=False).to(device)
    num_joints = dist_matrix.shape[0]

    # Grid Setup
    grid_w = img_size // patch_size
    grid_h = img_size // patch_size
    expected_patches = grid_w * grid_h
    has_cls = (N == expected_patches + 1)
    
    # Average heads
    attn_avg = attn_map.mean(dim=1) # (B, N, N)
    
    kti_scores = []
    
    for b in range(B):
        # 1. Map Joint -> Token
        joint_to_token = {}
        kp = keypoints_2d[b]
        
        for j in range(min(num_joints, kp.shape[0])):
            x, y = kp[j, :2]
            conf = kp[j, 2] if kp.shape[1] > 2 else 1.0
            
            # De-normalize if needed [-1, 1] -> Pixel
            if kp[:, :2].abs().max() < 2.0:
                x = (x + 1) * 0.5 * img_size
                y = (y + 1) * 0.5 * img_size
            
            if conf > 0 and 0 <= x < img_size and 0 <= y < img_size:
                gx, gy = int(x // patch_size), int(y // patch_size)
                token_idx = min(max(gy * grid_w + gx, 0), expected_patches - 1)
                if has_cls: token_idx += 1
                joint_to_token[j] = token_idx

        if not joint_to_token:
            kti_scores.append(0.0)
            continue

        # 2. Build Token-to-Token Topology Target T
        # Only for tokens that have joints mapped to them
        tokens = sorted(list(set(joint_to_token.values())))
        # Map token_idx -> sub_idx
        t2s = {t: i for i, t in enumerate(tokens)}
        n_sub = len(tokens)
        
        sub_target = torch.zeros((n_sub, n_sub), device=device)
        
        # Token Distance = Min Geodesic distance between any joints mapped to them
        # (Usually 1 joint per token in dense joints, but multiple possible)
        token_to_joints = {}
        for j, t in joint_to_token.items():
            if t not in token_to_joints: token_to_joints[t] = []
            token_to_joints[t].append(j)
            
        for t1 in tokens:
            for t2 in tokens:
                # Min Geodesic Dist
                min_d = float('inf')
                for j1 in token_to_joints[t1]:
                    for j2 in token_to_joints[t2]:
                        d = dist_matrix[j1, j2].item()
                        if d < min_d: min_d = d
                
                sub_target[t2s[t1], t2s[t2]] = math.exp(-(min_d**2) / (sigma**2))
        
        # 3. Extract Sub-Attention and Compare
        # We look at attention restricted to the predicted pose skeleton patches
        sub_attn = torch.zeros((n_sub, n_sub), device=device)
        for i, t1 in enumerate(tokens):
            for j, t2 in enumerate(tokens):
                sub_attn[i, j] = attn_avg[b, t1, t2]
        
        # Cosine Similarity
        if sub_attn.sum() > 0:
            sim = F.cosine_similarity(sub_attn.flatten().unsqueeze(0), 
                                      sub_target.flatten().unsqueeze(0)).item()
            kti_scores.append(sim)
        else:
            kti_scores.append(0.0)

    return np.mean(kti_scores) if kti_scores else 0.0

if __name__ == "__main__":
    print("="*60)
    print("METRIC: KINEMATIC TOPOLOGY INTERACTION (KTI)")
    print("="*60)
    print("Logic: Gaussian Decay (Sigma=2.0) + Cosine Similarity")

    # --- Test 1: Matrix KTI ---
    print("\n[Test 1] Matrix-based KTI (Joint Level)")
    dist_matrix = get_geodesic_distance_matrix(directed=False)
    
    # Case A: Ideal Topology Match
    target_perfect = torch.exp(-(dist_matrix**2) / (2.0**2)).unsqueeze(0).unsqueeze(0)
    kti_ideal = calculate_kti(target_perfect, dist_matrix)
    print(f"Ideal Topology KTI: {kti_ideal:.4f} (Expected: 1.0)")

    # Case B: Random Unstructured
    attn_random = F.softmax(torch.randn(1, 1, 24, 24), dim=-1)
    kti_random = calculate_kti(attn_random, dist_matrix)
    print(f"Random Unstructured KTI: {kti_random:.4f} (Expected: Low)")

    # --- Test 2: Patch KTI ---
    print("\n[Test 2] Patch-based KTI (Physically Grounded)")
    B, H, S = 1, 4, 256
    P = 16
    N = (S//P)**2 + 1
    
    keypoints = torch.zeros(B, 24, 3)
    # Pelvis at (128, 128)
    keypoints[:, 0, 0] = 128; keypoints[:, 0, 1] = 128; keypoints[:, 0, 2] = 1.0
    # Spine1 (Child of 0) at (128, 112) -> Dist 1.0
    keypoints[:, 3, 0] = 128; keypoints[:, 3, 1] = 112; keypoints[:, 3, 2] = 1.0
    
    # Generate Attention that looks at neighbors
    attn_map = torch.randn(B, H, N, N) * 0.1
    t0 = (128//P)*(S//P) + (128//P) + 1
    t3 = (112//P)*(S//P) + (128//P) + 1
    attn_map[:, :, t0, t3] = 10.0 # High attention to neighbor
    attn_map = F.softmax(attn_map, dim=-1)
    
    kti_patch = calculate_patch_kti(attn_map, keypoints, patch_size=P, img_size=S)
    print(f"Neighbor-Aware Patch KTI: {kti_patch:.4f}")
    
    print("\n✓ Verification Complete.")
