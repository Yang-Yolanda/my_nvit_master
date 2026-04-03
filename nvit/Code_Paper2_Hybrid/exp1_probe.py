import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add nvit to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from dssp_experiments.calculate_kti import calculate_kti
    from dssp_experiments.smpl_topology import get_smpl_adjacency_matrix, SMPL_PARENTS, SMPL_JOINT_NAMES
except ImportError:
    try:
        import calculate_kti
        from calculate_kti import calculate_kti
        import smpl_topology
        from smpl_topology import get_smpl_adjacency_matrix, SMPL_PARENTS, SMPL_JOINT_NAMES
    except ImportError:
        SMPL_JOINT_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand']
        SMPL_PARENTS = {'left_hip': 'pelvis', 'right_hip': 'pelvis', 'spine1': 'pelvis', 'left_knee': 'left_hip', 'right_knee': 'right_hip', 'spine2': 'spine1', 'left_ankle': 'left_knee', 'right_ankle': 'right_knee', 'spine3': 'spine2', 'left_foot': 'left_ankle', 'right_foot': 'right_ankle', 'neck': 'spine3', 'left_collar': 'spine3', 'right_collar': 'spine3', 'head': 'neck', 'left_shoulder': 'left_collar', 'right_shoulder': 'right_collar', 'left_elbow': 'left_shoulder', 'right_elbow': 'right_shoulder', 'left_wrist': 'left_elbow', 'right_wrist': 'right_elbow', 'left_hand': 'left_wrist', 'right_hand': 'right_wrist'}

def get_line_patches(start_idx, end_idx, grid_w, grid_h):
    x0, y0 = start_idx % grid_w, start_idx // grid_w
    x1, y1 = end_idx % grid_w, end_idx // grid_w
    patches = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    while True:
        if 0 <= x0 < grid_w and 0 <= y0 < grid_h:
            patches.append(y0 * grid_w + x0)
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx: err += dx; y0 += sy
    return patches

def create_patch_adjacency(joints_2d, patch_size=16, img_w=192, img_h=256):
    grid_h, grid_w = img_h // patch_size, img_w // patch_size
    N = grid_h * grid_w
    joint_to_patch = {}
    for i in range(len(joints_2d)):
        u, v = joints_2d[i][:2]
        u, v = max(0, min(img_w-1, u)), max(0, min(img_h-1, v))
        joint_to_patch[i] = int((v // patch_size) * grid_w + (u // patch_size))
    mask = torch.zeros((N, N))
    name_to_idx = {name: i for i, name in enumerate(SMPL_JOINT_NAMES)}
    skeleton_patches = set()
    for child, parent in SMPL_PARENTS.items():
        if parent is None or child not in name_to_idx or parent not in name_to_idx: continue
        c_idx, p_idx = name_to_idx[child], name_to_idx[parent]
        p_c, p_p = joint_to_patch[c_idx], joint_to_patch[p_idx]
        if p_c < N and p_p < N:
            line_patches = get_line_patches(p_c, p_p, grid_w, grid_h)
            skeleton_patches.update(line_patches)
            for p1 in line_patches:
                for p2 in line_patches: mask[p1, p2] = 1.0
    for p in skeleton_patches: mask[p, p] = 1.0
    return mask

def compute_spatial_distance_matrix(grid_h, grid_w, patch_size):
    y, x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
    coords = torch.stack([(y.float()+0.5)*patch_size, (x.float()+0.5)*patch_size], dim=-1).flatten(0, 1)
    return torch.cdist(coords, coords, p=2)

class ViTDiagnosticProbe:
    def __init__(self, model):
        self.model, self.attention_maps, self.features, self.hooks = model, {}, {}, []
        self._register_hooks()
        self.gt_joints, self.dist_matrix = None, None
    def _register_hooks(self):
        if hasattr(self.model, 'blocks'):
            for i, b in enumerate(self.model.blocks):
                if hasattr(b, 'attn') and hasattr(b.attn, 'attn_drop'):
                    self.hooks.append(b.attn.attn_drop.register_forward_hook(self._get_attn_hook(i)))
                self.hooks.append(b.register_forward_hook(self._get_feat_hook(i)))
    def _get_attn_hook(self, idx):
        def hook(m, i, o):
             self.attention_maps[idx] = i[0].detach().cpu()
        return hook
    def _get_feat_hook(self, idx):
        def hook(m, i, o):
             self.features[idx] = o.detach().cpu()
        return hook
    def run_diagnosis(self, x, gt_joints=None):
        self.attention_maps, self.features, self.gt_joints = {}, {}, gt_joints
        self.model.eval()
        with torch.no_grad(): self.model(x)
    def compute_metrics(self, quiet=False):
        results = []
        if not self.attention_maps: return results
        SL = self.attention_maps[list(self.attention_maps.keys())[0]].shape[-1]
        h, w, ps = (224, 224, 16) if SL==197 else (256, 192, 16)
        has_cls, N = (SL in [197, 193]), (SL-1 if SL in [197, 193] else SL)
        gh, gw = h//ps, w//ps
        if self.dist_matrix is None or self.dist_matrix.shape[0] != N:
            self.dist_matrix = compute_spatial_distance_matrix(gh, gw, ps)
        mask = create_patch_adjacency(self.gt_joints[0].cpu().numpy(), ps, w, h) if self.gt_joints is not None else None
        for i in sorted(self.attention_maps.keys()):
            attn = self.attention_maps[i][:, :, 1:, 1:] if has_cls else self.attention_maps[i]
            avg_attn = attn.mean(dim=(0, 1))
            ent = -torch.sum(avg_attn * torch.log(avg_attn + 1e-9), dim=-1).mean().item()
            kti = calculate_kti(attn, mask)[0] if mask is not None and attn.shape[-1] == mask.shape[0] else -2.0
            dist = (attn * self.dist_matrix).sum(dim=-1).mean().item()
            rank = -1.0
            if i in self.features:
                f = self.features[i].view(-1, self.features[i].shape[-1])
                try: S = torch.svd(f.float())[1]; Sn = S/S.sum(); rank = torch.exp(-torch.sum(Sn*torch.log(Sn+1e-9))).item()
                except: pass
            results.append({'layer': i, 'entropy': ent, 'rank': rank, 'kti': kti, 'dist': dist})
            if not quiet: print(f'L{i:02d} | E: {ent:.4f} | R: {rank:.1f} | K: {kti:.4f} | D: {dist:.1f}px')
        return results
