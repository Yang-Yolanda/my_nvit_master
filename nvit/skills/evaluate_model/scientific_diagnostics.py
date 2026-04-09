import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from tqdm import tqdm
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Allow importing from parent (nvit root)
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Logic for HMR2 import 
# If running via 4d_humans_diagnostic, hmr2 should be in path
try:
    from hmr2.models import load_hmr2
    from hmr2.models import load_hmr2
    from hmr2.utils import Evaluator  # ← Official MPJPE calculator
    from hmr2.configs import dataset_eval_config # ← To get correct keypoint definitions
except ImportError:
    # Fallback or debug
    from models import load_hmr2 # Attempt local if failed (unlikely)
    Evaluator = None  # Fallback
from collections import defaultdict
import types

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Wrapper System ---

class ModelWrapper(nn.Module):
    """
    Standard interface for all HMR models to be diagnosed.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def get_backbone(self):
        """Must return the ViT backbone object that has a .blocks attribute."""
        raise NotImplementedError
    
    def to_device(self, batch, device):
        """Map batch to device in a model-specific way."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    def forward(self, batch):
        """Execute forward pass and return standardized output if possible."""
        raise NotImplementedError

class HMR2Wrapper(ModelWrapper):
    def get_backbone(self):
        if hasattr(self.model, 'nvit_backbone'):
            return self.model.nvit_backbone
        return self.model.backbone
    
    def forward(self, batch):
        return self.model(batch)

class HSMRWrapper(ModelWrapper):
    def get_backbone(self):
        # Based on my research, HSMR uses .backbone.backbone or similar
        # Need to verify in HSMR code, but for now assuming .backbone
        if hasattr(self.model, 'backbone'):
            return self.model.backbone
        return self.model

    def forward(self, batch):
        # HSMR expects a tensor, but our loader provides a dict
        if isinstance(batch, dict) and 'img' in batch:
            out = self.model(batch['img'])
        else:
            out = self.model(batch)
            
        print("DEBUG: HSMRWrapper forward (Real Class)", flush=True)
        # Sanitization Removed: User fixed SKEL extension.
        return out

class PromptHMRWrapper(ModelWrapper):
    def get_backbone(self):
        # PromptHMR uses: model.image_encoder.backbone.encoder (DINOv2 ViT)
        return self.model.image_encoder.backbone.encoder

    def forward(self, batch):
        # PromptHMR expects normalized images. 
        # HMR2 uses 256. PromptHMR (ViT-H/Giant) needs 224 (Patch 14) or 252 (18x14).
        # We enforce 224x224.
        
        import torch.nn.functional as F
        
        # 1. Get Image Tensor
        img_tensor = batch['img'] if isinstance(batch, dict) else batch
        
        # 2. Resize if needed
        if img_tensor.shape[-1] != 224:
            # print(f"DEBUG PromptHMR: Resizing input {img_tensor.shape} -> 224x224", flush=True)
            img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            
        # 3. Prepare Input for PromptHMR (List of Dicts)
        # It expects uncollated inputs usually if using original forward
        # But Wrapper might call model(tensor) if model handles it?
        # Let's check original code I replaced...
        # Original code used: `return self.model(uncollated_batch)` with logic to create uncollated_batch.
        # I must preserve that logic but use `img_tensor`.
        
        if isinstance(batch, dict):
            B = img_tensor.shape[0]
            uncollated_batch = []
            
            for i in range(B):
                sample = {}
                sample['image'] = img_tensor[i] # (C, 224, 224)
                
                # Camera Intrinsics
                if 'cam_intrinsics' in batch:
                    # PHMR expects 'cam_int' (3, 3)
                    sample['cam_int'] = batch['cam_intrinsics'][i]
                elif 'cam_int' in batch:
                    sample['cam_int'] = batch['cam_int'][i]
                else:
                    # Fallback dummy
                    sample['cam_int'] = torch.eye(3, device=img_tensor.device)
                
                # Boxes (Required by PHMR forward logic?)
                # We can provide full image box (224x224 now)
                # PHMR PromptEncoder expects [x1, y1, x2, y2, confidence]
                H, W = 224, 224 
                sample['boxes'] = torch.tensor([[0, 0, W, H, 1.0]], device=img_tensor.device).float()
                
                uncollated_batch.append(sample)
                
            # Call PHMR forward with default prompts
            return self.model(uncollated_batch)
            
        return self.model(img_tensor)

class SigLIPBackboneProxy:
    def __init__(self, encoder):
        self.encoder = encoder
    @property
    def blocks(self):
        return self.encoder.layers
    def __getattr__(self, name):
        return getattr(self.encoder, name)

class SigLIPWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        # Wrap the encoder so .blocks points to .layers
        self.proxy_backbone = SigLIPBackboneProxy(model.vision_model.encoder)
        
    def get_backbone(self):
        return self.proxy_backbone

    def forward(self, batch):
        # SigLIP expects 'img' tensor
        img = batch.get('img') if isinstance(batch, dict) else batch
        
        # SigLIP expects 384x384 usually
        import torch.nn.functional as F
        if img.shape[-1] != 384:
            img = F.interpolate(img, size=(384, 384), mode='bilinear', align_corners=False)
            
        # Run model to trigger KTI hooks
        _ = self.model.vision_model(img)
        
        # Return Dummy Output for Evaluator
        batch_size = img.shape[0]
        device = img.device
        
        # CRITICAL FIX: Return RANDOM dummy keypoints (not zeros)
        # SigLIP has no pose knowledge, so we generate random keypoints
        # This allows KTI calculation to proceed (should yield low/random KTI)
        import torch
        dummy_kp_2d = torch.rand(batch_size, 44, 2, device=device) * 256  # Random in [0, 256]
        dummy_kp_3d = torch.rand(batch_size, 44, 3, device=device) * 2 - 1  # Random in [-1, 1]
        
        dummy_out = {
            'pred_keypoints_3d': dummy_kp_3d,
            'pred_keypoints_2d': dummy_kp_2d,
            'pred_vertices': torch.zeros(batch_size, 6890, 3, device=device),
            'pred_smpl_params': {'body_pose': torch.zeros(batch_size, 23, 3, 3, device=device)}
        }
        return dummy_out

class MMHuman3DWrapper(ModelWrapper):
    def get_backbone(self):
        # CNN-based or different architecture, no standard ViT blocks to patch
        return None

    def forward(self, batch):
        # batch: dict from HMR2 loader
        # mmhuman3d models expect: return model(img=img, img_metas=img_metas, sample_idx=sample_idx)
        img = batch.get('img')
        if img is None: return None
        
        # Construct Dummy img_metas
        img_metas = [{'sample_idx': i} for i in range(img.shape[0])]
        
        try:
             # Try standardized MM interface
             # We assume model is in eval mode and takes kwargs
             return self.model(img=img, img_metas=img_metas, return_loss=False)
        except Exception:
             # Fallback
             return self.model(img)

class CameraHMRWrapper(ModelWrapper):
    def get_backbone(self):
        return self.model.backbone
    
    def forward(self, batch):
        # CameraHMR needs 'cam_int' in batch
        
        # --- FIX: Force Square Input (256x256) ---
        import torch.nn.functional as F
        if batch['img'].shape[-1] != 256 or batch['img'].shape[-2] != 256:
             print(f"DEBUG CameraHMR: Resizing input from {batch['img'].shape} -> (256, 256)", flush=True)
             batch['img'] = F.interpolate(batch['img'], size=(256, 256), mode='bilinear', align_corners=False)
             print(f"DEBUG CameraHMR: New Shape: {batch['img'].shape}", flush=True)
        else:
             print(f"DEBUG CameraHMR: Input already 256x256: {batch['img'].shape}", flush=True)
        # ----------------------------------------
        
        if 'cam_int' not in batch:
            # Inject Dummy Intrinsics (B, 3, 3)
            # Focal length ~5000 is common for HMR crop space
            B = batch['img'].shape[0]
            device = batch['img'].device
            
            # [5000, 0, W/2]
            # [0, 5000, H/2]
            # [0, 0, 1]
            H, W = batch['img'].shape[2:]
            
            cam_int = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
            cam_int[:, 0, 0] = 5000.0
            cam_int[:, 1, 2] = H / 2.0
            batch['cam_int'] = cam_int
            
        out = self.model(batch)
        # CameraHMR returns: (pred_smpl_params, pred_cam, fl_h)
        # We need to compute 3D keypoints for the evaluator!
        
        pred_smpl_params, pred_cam, _ = out
        
        # We need an SMPL layer to get joints from params
        # But we don't have it easily here. 
        # HMR2 Evaluator can re-run SMPL if we give it params? No, it expects 'pred_keypoints_3d'.
        
        # Use HMR2's smpl layer if available? Or just use the model's head?
        # CameraHMR's head already ran SMPL. 
        # But it returns params.
        
        # Let's rely on `diagnostic_engine` to handle this?
        # No, `lab.run_experiment` passes `out` directly to `evaluator`.
        
        # For MPJPE, we need 'pred_keypoints_3d'.
        # We must forward the SMPL params to getting joints.
        
        # Quick Hack: Return a Dict matching HMR2 format.
        # But computing joints requires SMPL body model on device.
        
        output = {}
        output['pred_smpl_params'] = pred_smpl_params
        output['pred_cam'] = pred_cam
        
        # If we can't easily get 3D joints here without loading another SMPL, 
        # we might be stuck unless we change how CameraHMR works.
        # However, CameraHMR DOES compute them inside `smpl_head`. 
        # Let's check `smpl_head.py`? 
        # It typically returns vertices/joints too but CameraHMR forward filters them (see `camerahmr_model.py` line 55).
        
        # CRITICAL: We need 3D keypoints for MPJPE.
        # We will create a standardized SMPL-neutral layer here on the fly? Expensive.
        # Better: CameraHMR `smpl_head` (CliffHead) returns (smpl_output, ...).
        # We can re-call it? No.
        
        # Let's look at `camerahmr_model.py`:
        # pred_smpl_params, pred_cam, _, _ = self.smpl_head(...)
        # The 3rd and 4th args are likely vertices/joints.
        
        # Since I cannot easily change the model code (compiled/loaded), 
        # I will load a standalone SMPL layer in this wrapper __init__.
        
        if not hasattr(self, 'smpl_layer'):
             from smplx import SMPL
             # User provided path: /home/yangz/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl
             # smplx expects the folder containing the model files
             smpl_path = '/home/yangz/.cache/4DHumans/data/smpl/'
             
             import os
             if not os.path.exists(smpl_path):
                 # Fallback mechanisms
                 smpl_path = '/home/yangz/NViT-master/external_models/CameraHMR/data/models/'
             
             self.smpl_layer = SMPL(smpl_path, gender='neutral')
             self.smpl_layer.to(batch['img'].device)

        # Run SMPL
        # DEBUG: Check shapes
        print(f"DEBUG CameraHMR: betas={pred_smpl_params['betas'].shape}, body_pose={pred_smpl_params['body_pose'].shape}, global_orient={pred_smpl_params['global_orient'].shape}", flush=True)
        
        # CameraHMR might return rotation matrices (B, 23, 3, 3) or (B, 207)
        # SMPL layer expects (B, 23, 3, 3) or axis-angle (B, 69) depending on config.
        # If it's 6d rot, we need to convert?
        
        # If body_pose is (B, 23, 3, 3), we might need to flatten it for some SMPL implementations?
        # But smplx/body_models.py usually handles matrices if use_pca=False and use_rot6d=False?
        
        # Let's try to convert/flatten if needed based on error.
        # Error was "1x639". 
        
        # Helper for RotMat -> AA (Robust)
        def batch_rotmat_to_aa(rot_mats):
            # rot_mats: (N, 3, 3)
            # Based on kornia/pytorch3d logic
            eps = 1e-6
            trace = rot_mats[:, 0, 0] + rot_mats[:, 1, 1] + rot_mats[:, 2, 2]
            cos_theta = (trace - 1) / 2
            cos_theta = torch.clamp(cos_theta, -1 + eps, 1 - eps)
            theta = torch.acos(cos_theta)
            
            sin_theta = torch.sqrt(1 - cos_theta**2)
            factor = theta / (2 * sin_theta + eps)
            
            w = torch.stack([
                rot_mats[:, 2, 1] - rot_mats[:, 1, 2],
                rot_mats[:, 0, 2] - rot_mats[:, 2, 0],
                rot_mats[:, 1, 0] - rot_mats[:, 0, 1]
            ], dim=1)
            
            return w * factor.unsqueeze(1)

        # Convert
        B = pred_smpl_params['body_pose'].shape[0]
        
        # Body Pose
        bp_rot = pred_smpl_params['body_pose'].view(-1, 3, 3) # (B*23, 3, 3)
        bp_aa = batch_rotmat_to_aa(bp_rot).view(B, -1) # (B, 69)
        
        # Global Orient
        go_rot = pred_smpl_params['global_orient'].view(-1, 3, 3) # (B, 3, 3)
        go_aa = batch_rotmat_to_aa(go_rot).view(B, -1) # (B, 3)
        
        smpl_out = self.smpl_layer(
            betas=pred_smpl_params['betas'],
            body_pose=bp_aa,
            global_orient=go_aa
        )
        
        # Root Centering (Fix for 858mm scaling error - likely translation mismatch)
        joints = smpl_out.joints
        root = joints[:, [0], :]
        joints_centered = joints - root
        
        output['pred_keypoints_3d'] = joints_centered
        # Vertices also need centering if used
        output['pred_vertices'] = smpl_out.vertices - smpl_out.vertices[:, [0], :] # Align roughly to root too just in case

        # Project to 2D (Standard HMR Weak Perspective)
        # pred_cam: (B, 3) -> [scale, tx, ty]
        # joints: (B, J, 3) -> (x, y, z)
        # 2D = s * (x,y) + (tx, ty)
        
        pred_cam = pred_smpl_params.get('pred_cam', torch.tensor([0.9, 0, 0], device=batch['img'].device).repeat(B, 1))
        scale = pred_cam[:, 0].view(B, 1, 1)
        trans = pred_cam[:, 1:].view(B, 1, 2)
        
        # Weak Perspective Projection (Orthographic)
        # Ignore Z for projection in standard HMR coords (usually centered)
        # HMR2 dataset uses 256x256, coords in [-1, 1] range depending on implementation, 
        # but Evaluator often expects pixel coords or normalized?
        # Actually HMR2 Evaluator usually handles `pred_keypoints_2d` as Normalized [-1, 1] or Pixel space.
        # Let's check `perspective_projection` but simplest is:
        
        output['pred_keypoints_2d'] = scale * smpl_out.joints[:, :, :2] + trans
        # Slice to 44 joints if output is 45 (common mismatch)
        if output['pred_keypoints_2d'].shape[1] > 44:
             output['pred_keypoints_2d'] = output['pred_keypoints_2d'][:, :44, :]
        
        if 'pred_keypoints_3d' in out:
             # Unconditionally replace to avoid MagicMock/Shape issues (Missing Skel)
             # print("HSMR Sanitizer: Overwriting pred_keypoints_3d")
             B = batch['img'].shape[0]
             out['pred_keypoints_3d'] = torch.zeros(B, 44, 3, device=batch['img'].device)
        
        if 'pred_vertices' in out:
             # print("HSMR Sanitizer: Overwriting pred_vertices")
             B = batch['img'].shape[0]
             out['pred_vertices'] = torch.zeros(B, 6890, 3, device=batch['img'].device)

        return output

class NLFWrapper(ModelWrapper):
    def get_backbone(self):
        return self.model.backbone

    def forward(self, batch):
        return self.model(batch)

# --- End Wrapper System ---

class ViTDiagnosticLab:
    def __init__(self, model_wrapper, model_name='ViT-Model', output_root='results'):
        """
        Initialize the Generic ViT Diagnostic Lab.
        Args:
            model_wrapper: The wrapped model to diagnose.
            model_name: Name of the model.
            output_root: Root directory for results.
        """
        self.wrapper = model_wrapper
        self.model = model_wrapper.model # Direct access for patching
        self.output_dir = Path(output_root) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dynamically detect depth
        try:
            backbone = self.wrapper.get_backbone()
            if backbone is not None and hasattr(backbone, 'blocks'):
                total_layers = len(backbone.blocks)
            else:
                 # If no backbone (CNN) or unknown, we define total_layers as 0 
                 # This effectively disables masking groups generation loop
                total_layers = 0
        except:
             total_layers = 0 
             
        self.total_layers = total_layers
        if total_layers > 0:
            logger.info(f"Detected ViT depth: {total_layers} layers. Masking Enabled.")
        else:
            logger.info("No patchable ViT backbone detected. Masking Disabled (Baseline Only).")
             
        self.total_layers = total_layers
        logger.info(f"Detected model depth: {total_layers} layers.")

        self.groups = {
            'Control': {'mask_layers': [], 'mode': 'none'},
        }
        
        # Static Splits: 25%, 33%, 50%, 66%, 75%
        # Meaning: We KEEP X% and MASK the rest.
        # e.g. Keep-25% means mask L(25%) -> End.
        
        # --- Comprehensive Combinatorial Sweep (Run by Default) ---
        # 1. 1/4 Splits (25%, 50%, 75%)
        # 2. 1/3 Splits (33%, 66%)
        # 3. Hybrid Permutations
        
        ratios = {
            '1/4': [0.25, 0.50, 0.75],
            '1/3': [0.33, 0.66]
        }
        
        # Generator Loop
        for ratio_type, points in ratios.items():
            for p in points:
                split_layer = int(total_layers * p)
                split_layer = max(1, split_layer) # Safety
                
                label = int(p * 100)
                
                # Hard Encoding
                self.groups[f'Hard-{ratio_type}-{label}%'] = {
                    'mask_layers': list(range(split_layer, total_layers)),
                    'mode': 'hard'
                }
                
                # Soft Encoding
                self.groups[f'Soft-{ratio_type}-{label}%'] = {
                    'mask_layers': list(range(split_layer, total_layers)),
                    'mode': 'soft'
                }

        # Hybrid Permutations (NViT Style)
        # Type A (1/4 Hybrid): Keep 25% -> Soft 25% -> Hard 50%
        l_25 = int(total_layers * 0.25)
        l_50 = int(total_layers * 0.50)
        self.groups['Hybrid-1/4Mix'] = {
            'mask_layers': list(range(l_25, total_layers)),
            'mode': 'hybrid',
            'layer_modes': {i: 'soft' if i < l_50 else 'hard' for i in range(l_25, total_layers)}
        }

        # Type B (1/3 Hybrid): Keep 33% -> Soft 33% -> Hard 33%
        l_33 = int(total_layers * 0.33)
        l_66 = int(total_layers * 0.66)
        self.groups['Hybrid-1/3Mix'] = {
            'mask_layers': list(range(l_33, total_layers)),
            'mode': 'hybrid',
            'layer_modes': {i: 'soft' if i < l_66 else 'hard' for i in range(l_33, total_layers)}
        }

        # Type C (Sandwich - "Useless Middle" Hypothesis): Keep 25% -> Soft 50% -> Hard 25%
        # Tests if the middle 50% layers (Body) are redundant and replaceable by Soft constraints.
        l_75 = int(total_layers * 0.75)
        self.groups['Hybrid-Sandwich'] = {
            'mask_layers': list(range(l_25, total_layers)),
            'mode': 'hybrid',
            'layer_modes': {i: 'soft' if i < l_75 else 'hard' for i in range(l_25, total_layers)}
        }

        logger.info(f"Generated {len(self.groups)} experimental groups for Sweep.")
        
        self.results = []
        self.current_mask_config = {}
        self.layer_metrics = defaultdict(lambda: {'entropy': [], 'kmi': [], 'rank': []})
        
        # SMPL Adjacency (24-joint topology)
        self.smpl_adj = torch.eye(24) 
        
        # Patch the model
        self._patch_attention_modules()
        
        # Placeholder for SMPL Adjacency (44x44 or similar)
        # We use a 24-joint topology standard for SMPL
        self.smpl_adj = torch.eye(24) # TODO: Get real adjacency
        
        if hasattr(model_wrapper, 'parents'):
            self.parents = model_wrapper.parents
        else:
             # Default SMPL Parents
             self.parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

        # Monkey Patch
        self._patch_attention_modules()

    def calculate_effective_rank(self, x):
        """
        Compute Effective Rank of feature matrix x (B, N, D).
        ER = exp(Entropy(normalized_singular_values))
        """
        if x.dim() == 3:
            # Average over batch
            ranks = []
            for b in range(x.shape[0]):
                feat = x[b].float() # (N, D)
                try:
                    # SVD singular values
                    s = torch.linalg.svdvals(feat)
                    # Normalize
                    s_norm = s / (s.sum() + 1e-9)
                    # Entropy
                    entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-9))
                    ranks.append(torch.exp(entropy).item())
                except:
                    ranks.append(0.0)
            return np.mean(ranks)
        return 0.0

    def calculate_entropy(self, attn_scores):
        """
        Compute Shannon Entropy of attention scores.
        attn_scores: (B, H, N, N) - Softmaxed
        """
        # H(x) = - sum(p * log(p))
        if torch.isnan(attn_scores).any():
             # logger.warning("NaN in attention scores!")
             return 0.0
        epsilon = 1e-9
        p = attn_scores + epsilon
        # Average over batch and heads
        entropy = -torch.sum(p * torch.log(p), dim=-1).mean()
        return entropy.cpu()

    def calculate_mean_attention_distance(self, attn_scores, dist_matrix):
        """
        Compute Mean Attention Distance (in pixels/patches).
        attn_scores: (B, H, N, N)
        dist_matrix: (N, N) Pixel/Grid distance
        """
        # Average distance weighted by attention
        # MAD = sum(A_ij * D_ij)
        
        # dist_matrix (N, N) -> (1, 1, N, N)
        dist = dist_matrix.unsqueeze(0).unsqueeze(0)
        
        # Check if CLS token exists (N=197 usually)
        N = attn_scores.shape[-1]
        
        # If N = grid*grid + 1, we exclude CLS (index 0) to measure "Spatial Locality"
        # If we include CLS, we must define its position meaningfuly.
        # Current logic used (-100, -100) which skews metric.
        # Better: Measure Patch-to-Patch Distance Only.
        
        if N > 196: # Assuming 14x14 + 1
             # Slice out CLS
             attn_subset = attn_scores[:, :, 1:, 1:]
             dist_subset = dist[:, :, 1:, 1:]
             
             # Re-normalize attention?
             # If we just sum, it won't sum to 1.
             # MAD = sum(p * d) / sum(p)
             
             w_dist = (attn_subset * dist_subset).sum(dim=-1) # (B, H, N-1)
             total_attn = attn_subset.sum(dim=-1) + 1e-9
             
             mean_dist_per_token = w_dist / total_attn
             return mean_dist_per_token.mean().cpu()
        
        else:
             weighted_dist = (attn_scores * dist).sum(dim=-1) # (B, H, N)
             return weighted_dist.mean().cpu()


    def calculate_physically_grounded_kti(self, attn_map, keypoints_2d, reduce="mean"):
        """
        Compute Physically Grounded KTI (Soft Topology Interaction).
        Metric for Patch-based ViTs: Does attention flow match Geodesic Topology?
        
        Logic:
        1. Map 2D Joints -> Patch Tokens.
        2. Construct Soft Target Matrix M based on Geodesic Distance.
           M_ij = exp( -dist(Joint_a, Joint_b)^2 / sigma^2 )
        3. Score = CosineSimilarity(Attn, M)
        """
        import math
        B, H, N, _ = attn_map.shape
        device = attn_map.device
        sigma = 2.0 # Standard width
        
        # Ensure Geodesic Distance Matrix is available
        if not hasattr(self, 'geo_dist_matrix'):
             try:
                 # Lazy import / load
                 sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent)) # Root nvit
                 from nvit.smpl_topology import get_geodesic_distance_matrix
                 self.geo_dist_matrix = get_geodesic_distance_matrix(directed=False).to(device)
             except ImportError:
                 # Try backup location
                 try:
                     from smpl_topology import get_geodesic_distance_matrix
                     self.geo_dist_matrix = get_geodesic_distance_matrix(directed=False).to(device)
                 except:
                     logger.warning("Could not import smpl_topology. Using identity fallback.")
                     self.geo_dist_matrix = torch.eye(24, device=device)
        
        dist_matrix = self.geo_dist_matrix.to(device)

        # Grid Setup
        if hasattr(self, 'current_feature_grid') and self.current_feature_grid is not None:
             grid_h, grid_w = self.current_feature_grid
        else:
             import math
             grid_w = int(math.sqrt(N if N % 1 == 0 else N - 1))
             grid_h = grid_w
        
        # Determine Patch Size
        if hasattr(self.wrapper.model, 'img_size'):
             img_size = self.wrapper.model.img_size
             if isinstance(img_size, tuple): img_size = img_size[0]
             patch_size = img_size // grid_w
        else:
             patch_size = 16

        img_h_est = grid_h * patch_size
        img_w_est = grid_w * patch_size
        
        # Average heads
        attn_avg = attn_map.mean(dim=1) # (B, N, N)
        
        kti_scores = []
        
        for b in range(B):
            curr_attn = attn_avg[b]
            # 1. Map Joints to Tokens
            token_to_joints = {} 
            kp = keypoints_2d[b]
            num_joints = min(dist_matrix.shape[0], kp.shape[0])

            for j in range(num_joints):
                x, y = kp[j, :2]
                conf = kp[j, 2] if kp.shape[1] > 2 else 1.0
                
                # De-normalize if needed [-1, 1] -> Pixel
                if kp[:, :2].abs().max() < 2.0:
                    x = (x + 1) * 0.5 * img_w_est
                    y = (y + 1) * 0.5 * img_h_est
                
                if conf > 0 and 0 <= x < img_w_est and 0 <= y < img_h_est:
                    grid_x = int(x // patch_size)
                    grid_y = int(y // patch_size)
                    grid_x = min(max(grid_x, 0), grid_w - 1)
                    grid_y = min(max(grid_y, 0), grid_h - 1)
                    
                    token_idx = grid_y * grid_w + grid_x
                    if N == grid_h * grid_w + 1: token_idx += 1 # CLS shift
                    
                    if 0 <= token_idx < N: 
                        if token_idx not in token_to_joints: token_to_joints[token_idx] = []
                        token_to_joints[token_idx].append(j)

            # 2. Build Soft Geodesic Target Sub-Graph
            # We iterate only over ACTIVE tokens
            active_tokens = list(token_to_joints.keys())
            
            if not active_tokens:
                kti_scores.append(0.0)
                continue
                
            sub_attn_values = []
            sub_target_values = []

            for t_i in active_tokens:
                    for t_j in active_tokens:
                        if t_i == t_j: continue # Skip diagonal (non-trivial alignment only)
                        joints_i = token_to_joints[t_i]
                        joints_j = token_to_joints[t_j]
                        
                        # Min Geodesic Dist
                        min_geo_dist = float('inf')
                        for j_a in joints_i:
                            for j_b in joints_j:
                                d = dist_matrix[j_a, j_b].item()
                                if d < min_geo_dist: min_geo_dist = d
                        
                        target_val = math.exp(-(min_geo_dist**2) / (sigma**2))
                        
                        sub_attn_values.append(curr_attn[t_i, t_j])
                        sub_target_values.append(target_val)
            
            # 3. Compute Cosine Similarity
            if not sub_attn_values:
                kti_scores.append(0.0)
            else:
                v_attn = torch.stack(sub_attn_values)
                v_target = torch.tensor(sub_target_values, device=device)
                
                sim = torch.nn.functional.cosine_similarity(v_attn.unsqueeze(0), v_target.unsqueeze(0), dim=1).item()
                kti_scores.append(sim)

        kti_scores_arr = np.array(kti_scores, dtype=np.float32) if kti_scores else np.zeros(B, dtype=np.float32)
        if reduce == "none":
            return kti_scores_arr
        elif reduce == "mean":
            return float(kti_scores_arr.mean()) if len(kti_scores_arr) > 0 else 0.0
        else:
            raise ValueError(f"Unknown reduce type: {reduce}")

    def calculate_kti(self, attn_map, smpl_adj, reduce="mean"):
        """
        KTI calculation wrapper.
        Delegates to physically grounded KTI if keypoints available.
        """
        if hasattr(self, 'current_keypoints') and self.current_keypoints is not None:
            return self.calculate_physically_grounded_kti(attn_map, self.current_keypoints, reduce)

        elif hasattr(self, 'current_batch_adj') and self.current_batch_adj is not None:
            # Fallback: Use pre-computed adjacency matrix
            B, H, N, _ = attn_map.shape
            attn_avg = attn_map.mean(dim=1)  # (B, N, N)
            
            if not getattr(self, '_kti_fallback_logged', False):
                self._kti_fallback_logged = True
                logger.info(f"DEBUG KTI: Using fallback adjacency path. B={B}, N={N}, adj_shape={self.current_batch_adj.shape}")
            
            kti_scores = []
            for b in range(B):
                # Element-wise product and sum
                adj = self.current_batch_adj[b]  # (N, N)
                score = (attn_avg[b] * adj).sum().item()
                # Normalize by number of edges
                num_edges = adj.sum().item()
                if num_edges > 0:
                    score = score / num_edges
                kti_scores.append(score)
            
            kti_scores_arr = np.array(kti_scores, dtype=np.float32)
            result_mean = float(kti_scores_arr.mean()) if len(kti_scores_arr) > 0 else 0.0
            
            if not getattr(self, '_kti_result_logged', False):
                self._kti_result_logged = True
                logger.info(f"DEBUG KTI: Fallback result = {result_mean}")
                
            if reduce == "none":
                return kti_scores_arr
            return result_mean
        else:
            if not getattr(self, '_kti_zero_logged', False):
                self._kti_zero_logged = True
                logger.info(f"DEBUG KTI: Returning 0.0 - no keypoints or adjacency.")
            # For reduce == "none", returning a float here is problematic, but typically B is unknown here
            # We assume single sample if fallback hits
            return np.array([0.0], dtype=np.float32) if reduce == "none" else 0.0

    def _get_kinematic_mask(self, batch_size, n_tokens, device):
        """
        Generate a mask reflecting SMPL topology.
        Ex: (B, H, N, N) boolean mask or additive -inf.
        """
        # Logic: 
        # 1. Start with full mask (ones)
        # 2. Set 'kinematically disconnected' regions to 0 (or -inf)
        # 3. Requires T2J mapping to know which Token is which Joint.
        
        # PROXY IMPLEMENTATION:
        # Mask out 50% randomly to test pipeline if no T2J
        # Real impl needs T2J.
        mask = torch.zeros((batch_size, 1, n_tokens, n_tokens), device=device)
        return mask # Zero means no masking (additive). Wait, mask should be -inf.

    def _patch_attention_modules(self):
        """
        Find ATT modules and monkey-patch forward.
        """
        found = False
        self.att_modules = []
        
        # Recursive search for 'att' or 'attn'
        for name, module in self.model.named_modules():
            # Check if it looks like the ATT class from nvit/models.py
            #It has 'attn_drop' and 'scale'
            if hasattr(module, 'attn_drop') and hasattr(module, 'scale'):
                self.att_modules.append((name, module))
                found = True
                
                # Patching
                original_forward = module.forward
                
                def custom_forward_wrapper(mod_self, x):
                    # Original Logic Reconstruction from models.py / ViT
                    # B, N, C = x.shape
                    # q, k, v = mod_self.qkv(x).chunk(3, dim=-1) # Assuming standard timm/ViT qkv
                    # But nvit/models.py has: q, k, v = self.qkv(x)
                    
                    # We need to call the original logic BUT intervene on attention scores.
                    # Since we can't easily decompose the original forward if it's monolithic without access to source code variables easily,
                    # We will try to inspect the module type.
                    # If it is 'Attention' from nvit/models.py:
                    # q,k,v = self.qkv(x)
                    # attn = self.att(q,k) -> WE INTERCEPT HERE
                    
                    # So we need to monkey patch ONLY the 'att' submodule if possible, OR replicate the logic.
                    # Let's replicate the logic assuming standard Structure.
                    
                    q, k, v = mod_self.qkv(x)
                    
                    # Flatten/Permute logic from models.py
                    # q, k, v shape depends on implementation. 
                    # models.py: q, k, v returned by QKV_s or QKV
                    
                    # models.py Attention.forward:
                    # B, N, C = x.shape
                    # q, k, v = self.qkv(x)
                    # qk_dim = q.shape[2] ...
                    # q = q.reshape...
                    
                    # To be safe and avoid reshaping errors, we should monkey patch 'mod_self.att.forward' instead?
                    # The error showed: self.attn(self.norm1(x))
                    # So 'attn' is the module we patched.
                    # And 'attn' has a 'qkv' submodule.
                    
                    # Let's try to pass through to qkv, then do the math.
                    
                    B, N, C = x.shape
                    q, k, v = mod_self.qkv(x)
                    
                    # Assuming dimensions from models.py
                    num_heads = mod_self.num_heads
                    qk_dim = q.shape[2]
                    v_dim = v.shape[2]
                    
                    q = q.reshape(B, N, num_heads, qk_dim // num_heads).permute(0, 2, 1, 3)
                    k = k.reshape(B, N, num_heads, qk_dim // num_heads).permute(0, 2, 1, 3)
                    v = v.reshape(B, N, num_heads, v_dim // num_heads).permute(0, 2, 1, 3)

                    # --- ORIGINAL ATTENTION LOGIC (Replicated) ---
                    # attn_r = (q @ k.transpose(-2, -1)) * mod_self.scale
                    attn_r = (q @ k.transpose(-2, -1)) * mod_self.scale
                    
                    # --- INJECTION POINT ---
                    # Note: We miss layer_idx here unless bound. 
                    # We will bind it in the outer loop loop.
                    
                    attn = attn_r.softmax(dim=-1)
                    
                    # Metrics
                    # This wrapper is generic, the bound one below needs the index.
                    # So we just return attn here? 
                    # No, we need to continue execution: x = self.proj(attn, v)
                    
                    x_out = mod_self.proj(attn, v)
                    return x_out
                
                # Bind method
                module.forward = types.MethodType(custom_forward_wrapper, module) # Wait, forward is bound?
                # Usually forward is method. 
                # Better: replace the instance method?
                # module.forward = custom_forward_wrapper.__get__(module, module.__class__)
                # Or just assign function if it matches signature (ignoring self if wrapped properly)
    # Simpler:
                # module.forward = lambda q, k: custom_forward_wrapper(module, q, k)
                # But we need layer info.
        
        if found:
            logger.info(f"Patched {len(self.att_modules)} Attention modules.")
            # Inject layer indices
            for i, (name, mod) in enumerate(self.att_modules):
                 mod.layer_index = i
                 
                 # Redefine wrapper with index closure
                 def make_forward(idx, original_fwd):
                     def forward(x):
                         B, N, C = x.shape
                         # Inspect output of qkv
                         qkv = mod.qkv(x)
                         # Assuming qkv is (B, N, 3*head_dim*num_heads)
                         # standard timm QKV returns one tensor.
                         q, k, v = qkv.chunk(3, dim=-1)
                         
                         num_heads = mod.num_heads
                         qk_dim = q.shape[2]
                         v_dim = v.shape[2]
                        
                         q = q.reshape(B, N, num_heads, qk_dim // num_heads).permute(0, 2, 1, 3)
                         k = k.reshape(B, N, num_heads, qk_dim // num_heads).permute(0, 2, 1, 3)
                         v = v.reshape(B, N, num_heads, v_dim // num_heads).permute(0, 2, 1, 3)

                         attn_r = (q @ k.transpose(-2, -1)) * mod.scale
                         
                         # Masking
                         if idx in self.current_mask_config:
                             # Create mask: (B, H, N, N)
                             # Distance-based "Kinematic Proxy" Mask
                             # 14x14 grid. Mask tokens > distance 3.
                             # This forces local attention, simulating "Kinematic Constraints" (connect only to neighbors).
                             
                             n_tokens = q.shape[2] # 196 + 1 (cls)? 
                             # q has shape (B, N, Heads, Dim) after permute? 
                             # No, q before was (B, N, C). permute: (B, Heads, N, Dim)
                             # q inside forward here is (B, Heads, N, Dim). N is tokens.
                             
                             N = q.shape[2]
                             # Assuming N=197 (196 patch + 1 cls) or 196.
                             # Grid size 14x14
                             import math
                             grid_size = int(math.sqrt(N)) # 14
                             
                             if hasattr(self, 'current_feature_grid') and self.current_feature_grid is not None:
                                 grid_h, grid_w = self.current_feature_grid
                             else:
                                 import math
                                 grid_w = int(math.sqrt(N))
                                 grid_h = grid_w
                             
                             # Check cache validity: device AND shape
                             if (not hasattr(self, 'dist_mask') or 
                                 self.dist_mask_device != q.device or 
                                 self.dist_mask.shape[-1] != N):
                                 
                                 # Precompute mask
                                 if grid_h * grid_w == N:
                                    y, x_grid = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
                                    coords = torch.stack([x_grid.flatten(), y.flatten()], dim=1).float()
                                 elif grid_h * grid_w == N - 1:
                                     # CLS token case (197 vs 196)
                                     y, x_grid = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
                                     coords = torch.stack([x_grid.flatten(), y.flatten()], dim=1).float()
                                     cls_coord = torch.tensor([[-100., -100.]])
                                     coords = torch.cat([cls_coord, coords], dim=0)
                                 else:
                                     coords = torch.arange(N).unsqueeze(1).float()
                                     pass
                                 
                                 if 'coords' not in locals():
                                     coords = torch.arange(N).unsqueeze(1).float()
                                 
                                 dist = torch.cdist(coords, coords) # (N, N)
                                 
                                 # Determine Mask Mode
                                 current_mode = self.current_mask_config.get('mode', 'hard')
                                 if current_mode == 'hybrid':
                                     current_mode = self.current_mask_config['layer_modes'].get(idx, 'hard')

                                 # Mask Value Logic
                                 # Hard: -inf
                                 # Soft: -10.0 (Strong penalty but not absolute)
                                 mask_val = float('-inf') if current_mode == 'hard' else -10.0
                                 
                                 mask = torch.zeros_like(dist)
                                 mask[dist > 2.5] = mask_val
                                 
                                 self.dist_mask = mask.to(q.device)
                                 self.dist_mask_device = q.device
                             
                             # Broadcast self.dist_mask (N, N) to (B, H, N, N)
                             # Additive mask
                             attn_r += self.dist_mask.unsqueeze(0).unsqueeze(0)
                         
                         attn = attn_r.softmax(dim=-1)
                         
                         # Metrics
                         if not self.model.training: 
                             if idx not in self.layer_metrics:
                                 self.layer_metrics[idx] = {'entropy': [], 'kmi': [], 'rank': []}
                             e = self.calculate_entropy(attn)
                             self.layer_metrics[idx]['entropy'].append(e)
                             
                             # KTI Calculation (Proxy)
                             k = self.calculate_kti(attn, self.smpl_adj, reduce="none")
                             self.layer_metrics[idx]['kmi'].append(k)
                         
                         # Final Projection logic for standard ViT
                         # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                         # x = self.proj(x)
                         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                         x_out = mod.proj(x)
                         x_out = mod.proj_drop(x_out)
                         return x_out
                     return forward
                
                 mod.forward = make_forward(i, mod.forward)

    def calculate_effective_rank(self, features):
        """
        Compute Effective Rank of feature matrix.
        features: (B, N, D)
        """
        # Flatten batch: (B*N, D) or per sample?
        # Usually per sample rank averaged
        B, N, D = features.shape
        ranks = []
        
        for b in range(B):
            feat = features[b] # (N, D)
            # Use float32 SVD
            try:
                # SVD
                # torch.linalg.svd returns U, S, Vh
                _, S, _ = torch.linalg.svd(feat.float())
                
                # Normalize singular values
                # p_k = sigma_k / sum(sigma)
                sigma_sum = S.sum()
                if sigma_sum > 0:
                    p = S / sigma_sum
                    # Entropy of Singular Values: H = -sum(p log p)
                    entropy = -torch.sum(p * torch.log(p + 1e-9))
                    # Effective Rank = exp(H)
                    erank = torch.exp(entropy).cpu()
                    ranks.append(erank)
                else:
                    ranks.append(0.0)
            except Exception as e:
                # logger.warning(f"SVD Failed: {e}")
                ranks.append(0.0)
                
        return np.mean(ranks) if ranks else 0.0

    def _patch_attention_modules(self):
        found = False
        self.att_modules = []
        
        # We need to hook ATTENTION (for Entropy/KTI) and FEATURES (for Rank)
        # HMR2 backbone structure: model.backbone.blocks[i]
        
        # 1. Hooking Logic
        # We will use register_forward_hook on the Block modules to get features
        # And patch the Attention inside to get attn weights
        
        # Hooking logic using standardized wrapper access
        backbone = self.wrapper.get_backbone()
        
        if hasattr(backbone, 'blocks'):
            for i, blk in enumerate(backbone.blocks):
                # 1. Hook Features (Output of Block) for Effective Rank and Mamba Pseudo-Attention
                # We need a closure to capture 'i' and whether it has 'attn'
                has_attn_attr = hasattr(blk, 'attn')
                
                def make_feat_hook(idx, has_attn):
                    def feat_hook(module, input, output):
                        # output is (x) [B, N, D]
                        if not self.model.training:
                            features = output.detach()
                            er = self.calculate_effective_rank(features)
                            self.layer_metrics[idx]['rank'].append(er)
                            
                            # If no standard attention (Mamba/GCN), compute Token Affinity Pseudo-Attention
                            if not has_attn:
                                import math
                                B, N, D = features.shape
                                # Token Affinity = Softmax(X @ X^T / sqrt(D))
                                sim = torch.bmm(features, features.transpose(1, 2)) / math.sqrt(D)
                                pseudo_attn = torch.nn.functional.softmax(sim, dim=-1) # (B, N, N)
                                pseudo_attn_h = pseudo_attn.unsqueeze(1) # (B, 1, N, N)
                                
                                # Entropy
                                e = self.calculate_entropy(pseudo_attn_h)
                                self.layer_metrics[idx]['entropy'].append(e)
                                
                                # KTI (Topology)
                                k = self.calculate_kti(pseudo_attn_h, self.smpl_adj, reduce="none")
                                self.layer_metrics[idx]['kmi'].append(k)
                                
                    return feat_hook
                
                blk.register_forward_hook(make_feat_hook(i, has_attn_attr))
                
                # 2. Patch Attention (Monkey Patch) for Entropy/KTI/Mask
                if hasattr(blk, 'attn'):
                    module = blk.attn
                    self.att_modules.append((f"layer_{i}", module))
                    found = True
                    
                    # Original Forward
                    original_forward = module.forward
                    # The rest of monkey patch logic...
                    
                    # ... [Insert Monkey Patch Code Here] ...
                    # Re-implementing the monkey patch inside the loop to be clean
                    
                    # Store original to avoid recursion if verified multiple times?
                    # Assuming _patch_attention_modules called once.
                    if not hasattr(module, '_original_forward'):
                        module._original_forward = original_forward
                    
                    def make_attn_forward(idx, mod):
                        def forward(x, *args, **kwargs):
                             B, N, C = x.shape
                             # qkv = mod.qkv(x).reshape(B, N, 3, mod.num_heads, mod.head_dim).permute(2, 0, 3, 1, 4)
                             # q, k, v = qkv.unbind(0)
                             # q, k, v = qkv[0], qkv[1], qkv[2] 
                             
                             # HMR2/timm ViT standard qkv
                             qkv = mod.qkv(x) # (B, N, 3*dim)
                             qkv = qkv.reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
                             q, k, v = qkv[0], qkv[1], qkv[2]
                             
                             attn_r = (q @ k.transpose(-2, -1)) * mod.scale
                             
                             # Masking Check
                             mask_type = self.current_mask_config.get('mode', 'none')
                             if idx in self.current_mask_config.get('mask_layers', []) and mask_type != 'none':
                                  if hasattr(self, 'current_feature_grid') and self.current_feature_grid is not None:
                                      grid_h, grid_w = self.current_feature_grid
                                  else:
                                      import math
                                      grid_w = int(math.sqrt(N))
                                      grid_h = grid_w
                                  
                                  # Check cache / Recompute if needed
                                  if (not hasattr(self, 'dist_matrix_cache') or 
                                      self.dist_matrix_device != q.device or 
                                      self.dist_matrix_cache.shape[-1] != N):
                                      
                                      if grid_h * grid_w == N:
                                         y, x_grid = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
                                         coords = torch.stack([x_grid.flatten(), y.flatten()], dim=1).float()
                                      elif grid_h * grid_w == N - 1:
                                          y, x_grid = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
                                          coords = torch.stack([x_grid.flatten(), y.flatten()], dim=1).float()
                                          cls_coord = torch.tensor([[-100., -100.]])
                                          coords = torch.cat([cls_coord, coords], dim=0)
                                      else:
                                          coords = torch.arange(N).unsqueeze(1).float()
                                      
                                      # Distance Matrix (N, N)
                                      dist = torch.cdist(coords, coords)
                                      self.dist_matrix_cache = dist.to(q.device)
                                      self.dist_matrix_device = q.device
                                  
                                  dist = self.dist_matrix_cache

                                  
                                  # Resolve Mask Mode (Handle Hybrid)
                                  current_mode = mask_type
                                  if mask_type == 'hybrid':
                                      layer_modes = self.current_mask_config.get('layer_modes', {})
                                      current_mode = layer_modes.get(idx, 'none')

                                  if current_mode == 'hard':
                                      # Hard Mask: Kinematic Topology Constraint (KTI)
                                      if hasattr(self, 'current_batch_adj') and self.current_batch_adj is not None:
                                          # adj (B, N, N) -> (B, 1, N, N)
                                          adj = self.current_batch_adj.unsqueeze(1).to(q.device)
                                          mask = torch.zeros_like(adj)
                                          mask[adj == 0] = float('-inf')
                                          attn_r = attn_r + mask
                                      else:
                                          # Fallback
                                          if idx == 0 and b == 0: # Log once per forward
                                              logger.warning("Hard Mask Fallback: No KTI Adjacency found! Using Grid Distance.")
                                          mask = torch.zeros_like(dist)
                                          mask[dist > 3.5] = float('-inf')
                                          attn_r = attn_r + mask.unsqueeze(0).unsqueeze(0)
                                      
                                  elif current_mode == 'soft':
                                      # Soft Mask: Kinematic Bias
                                      if hasattr(self, 'current_batch_adj') and self.current_batch_adj is not None:
                                          # Penalty for unconnected tokens
                                          adj = self.current_batch_adj.unsqueeze(1).to(q.device)
                                          mask = torch.zeros_like(adj)
                                          # Apply finite penalty (e.g., -5.0) to disconnected regions
                                          mask[adj == 0] = -5.0
                                          attn_r = attn_r + mask
                                      else:
                                          # Fallback Gaussian
                                          sigma = 10.0 
                                          soft_bias = - (dist**2) / (2 * sigma)
                                          attn_r = attn_r + soft_bias.unsqueeze(0).unsqueeze(0)

                             attn = attn_r.softmax(dim=-1)
                             
                             # Metrics
                             if not self.model.training:
                                 if idx not in self.layer_metrics:
                                     self.layer_metrics[idx] = {'entropy': [], 'kmi': [], 'rank': [], 'dist': []}
                                 
                                 e = self.calculate_entropy(attn)
                                 self.layer_metrics[idx]['entropy'].append(e)
                                 
                                 # Attention Distance
                                 # dist matrix should exist from masking check above (or we create it if no masking?)
                                 # We need dist matrix even if masking is 'none'.
                                 if not hasattr(self, 'dist_matrix_cache'):
                                      # Just create it if missing (copy-paste creation logic or factor out?)
                                      # Factoring out is better but for now let's reuse if exists, else skip?
                                      # No, user wants it. We must ensure dist_matrix exists.
                                      pass

                                 # Only calculate if dist matrix exists (it is created in Masking block)
                                 # Wait, Masking block is conditional: `if idx in mask_layers...`
                                 # So if layer not masked, dist matrix might not exist!
                                 # We need to ensure dist matrix logic is global.
                                 
                                 # --- Lazy Init Dist Matrix (Global) ---
                                 if (not hasattr(self, 'dist_matrix_cache') or 
                                      self.dist_matrix_device != q.device or 
                                      self.dist_matrix_cache.shape[-1] != N):
                                      
                                      import math
                                      grid_w = int(math.sqrt(N))
                                      if grid_w*grid_w != N and grid_w*grid_w != N-1: grid_w = 14 # Fallback
                                      grid_h = grid_w
                                      
                                      if grid_h * grid_w == N:
                                         y, x_grid = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
                                         coords = torch.stack([x_grid.flatten(), y.flatten()], dim=1).float()
                                      elif grid_h * grid_w == N - 1:
                                          y, x_grid = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
                                          coords = torch.stack([x_grid.flatten(), y.flatten()], dim=1).float()
                                          cls_coord = torch.tensor([[-100., -100.]]) # Far away
                                          coords = torch.cat([cls_coord, coords], dim=0)
                                      else:
                                          coords = torch.arange(N).unsqueeze(1).float()
                                      
                                      dist = torch.cdist(coords, coords)
                                      self.dist_matrix_cache = dist.to(q.device)
                                      self.dist_matrix_device = q.device
                                 
                                 d = self.calculate_mean_attention_distance(attn, self.dist_matrix_cache)
                                 self.layer_metrics[idx]['dist'].append(d)
                                 
                                 k = self.calculate_kti(attn, self.smpl_adj, reduce="none")
                                 self.layer_metrics[idx]['kmi'].append(k)
                                 
                                 # Store Raw Attention for Visualization (Only last batch overwrites)
                                 if not hasattr(self, 'last_attention_maps'):
                                     self.last_attention_maps = {}
                                 # detach and cpu to save memory
                                 self.last_attention_maps[idx] = attn.detach().cpu()
                             
                             attn = mod.attn_drop(attn)
                             
                             # Proj
                             x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                             x = mod.proj(x)
                             x = mod.proj_drop(x)
                             return x
                             
                        return forward

                    module.forward = make_attn_forward(i, module)

    # Legacy method kept for interface compatibility if needed, but replaced by the one above
    def _patch_dummy(self):
        pass

    def compute_kmi_adjacency(self, batch):
        """
        Pre-compute the KTI-based adjacency mask for the current batch.
        Returns: (B, N, N) binary mask (1=Connected, 0=Disconnected).
        """
        if 'img' not in batch: return None
        
        # Grid Size inferred from image
        img_h, img_w = batch['img'].shape[-2:]
        patch_size = 16 
        grid_h = img_h // patch_size
        grid_w = img_w // patch_size
        N = grid_h * grid_w
        
        # --- CRITICAL FIX FOR HMR2 VI-T POSE ---
        # HMR2's backbone outputs exactly 192 tokens (16x12) natively, even if input is 256x256.
        if N == 256 and 'HMR' in str(type(self.wrapper)).upper():
             grid_h = 16
             grid_w = 12
             N = 192
        
        # Get Keypoints
        kp_2d = None
        if 'keypoints_2d' in batch:
            kp_2d = batch['keypoints_2d']
        elif 'keypoints' in batch:
            kp_2d = batch['keypoints']
            
        if kp_2d is None: return None
        
        device = batch['img'].device
        B = kp_2d.shape[0]
        
        # --- FIX: Support 256x192 (16x12) for HMR2/HSMR ---
        if grid_h * patch_size == 256 and grid_w * patch_size == 256:
             # Force detection if harness level fix didn't catch it
             # But if it's already 16x12, we keep it.
             pass
        
        img_h_est = grid_h * patch_size
        img_w_est = grid_w * patch_size
        
        # SMPL Kinematic Tree (24 Joints)
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        
        # HMR2/HSMR Output Joint Mapping (OpenPose-style index mapping to SMPL)
        # In HMR2's 44-joint output:
        # Index 0 is Nose (Not Pelvis!)
        # Index 39 is Regressed Pelvis (Root)
        joint_mapping = {
            0: 39, # Pelvis (SMPL 0 -> HMR2 39)
            1: 12, # L_Hip (SMPL 1 -> HMR2 12)
            2: 9,  # R_Hip (SMPL 2 -> HMR2 9)
            3: 0,  # Spine (Approx)
            4: 13, # L_Knee
            5: 10, # R_Knee
            6: 0,  # Spine
            7: 14, # L_Ankle
            8: 11, # R_Ankle
            # ... and so on for full 24. For KTI, Root/Hips/Neck are most critical for topology.
        }
        
        # Fallback identity for others if not specified (Standard SMPL 0-23)
        # But we MUST fix the Pelvis=39 for HMR2
        is_hmr_model = 'HMR' in str(type(self.wrapper)).upper() or 'HSMR' in str(type(self.wrapper)).upper()
        
        # Initialize with Identity (Self-loops are crucial!)
        adj_batch = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
        
        has_cls = (N == grid_h * grid_w + 1)
        
        if has_cls:
            # CLS Token (Index 0) acts as Global Node
            # It should see everything, and everything should see it
            adj_batch[:, 0, :] = 1.0
            adj_batch[:, :, 0] = 1.0
        
        for b in range(B):
            kp = kp_2d[b]
            valid_joints = {}
            
            # FIX: Handle Center Crop (Scaling vs Cropping)
            full_size = max(img_h_est, img_w_est)
            crop_x = (full_size - img_w_est) // 2
            crop_y = (full_size - img_h_est) // 2

            # DEBUG: Adjacency Diagnostics
            if b == 0 and not getattr(self, '_kmi_adj_debug_printed', False):
                 self._kmi_adj_debug_printed = True
                 if 'img' in batch:
                     logger.info(f"DEBUG (Mask): Batch Img Shape: {batch['img'].shape}")
                 else:
                     logger.info("DEBUG (Mask): No Img in batch")
                 logger.info(f"DEBUG (Mask): Grid (h,w): ({grid_h}, {grid_w}) Est Img: ({img_h_est}, {img_w_est})")
                 logger.info(f"DEBUG (Mask): First 5 Keypoints (Raw): {kp[:5].tolist()}")

            for j in range(24):
                # Map SMPL joint index 'j' to Model index 'm_idx'
                m_idx = j
                if is_hmr_model:
                     # Critical HMR2 Remapping: Pelvis is 39
                     if j == 0: m_idx = 39
                     elif j == 1: m_idx = 12 # L_Hip
                     elif j == 2: m_idx = 9  # R_Hip
                
                if m_idx >= kp.shape[0]: continue
                
                x, y = kp[m_idx, :2]
                if kp.shape[1] > 2: conf = kp[m_idx, 2] # Use m_idx
                else: conf = 1.0
                
                # Check for Normalized Coordinates (HMR2 uses [-1, 1])
                # WE MAP TO FULL IMAGE FIRST
                if kp[:, :2].abs().max() < 2.0:
                    x_full = (x + 1) * 0.5 * full_size
                    y_full = (y + 1) * 0.5 * full_size
                else:
                    x_full = x
                    y_full = y
                
                # Apply Crop Shift
                x = x_full - crop_x
                y = y_full - crop_y
                
                # Check visibility/bounds
                if conf > 0 and 0 <= x < img_w_est and 0 <= y < img_h_est:
                    # Map to grid
                    grid_x = int(x // patch_size)
                    grid_y = int(y // patch_size)
                    
                    # Clamp to be safe
                    grid_x = min(max(grid_x, 0), grid_w - 1)
                    grid_y = min(max(grid_y, 0), grid_h - 1)
                    
                    token_idx = grid_y * grid_w + grid_x
                    
                    if has_cls:
                        token_idx += 1 
                    
                    if 0 <= token_idx < N:
                        valid_joints[j] = token_idx
            
            # Edges
            for child, parent in enumerate(parents):
                if parent != -1 and child in valid_joints and parent in valid_joints:
                    t_c = valid_joints[child]
                    t_p = valid_joints[parent]
                    adj_batch[b, t_c, t_p] = 1.0
                    adj_batch[b, t_p, t_c] = 1.0
            
        return adj_batch

    def run_experiment(self, data_loader, evaluator, dataset_cfg, num_batches=10):
        """
        Run the diagnostic suite across all experimental groups.
        Args:
            data_loader: PyTorch DataLoader providing evaluation data.
            evaluator: An Evaluator object (e.g., hmr2.utils.Evaluator) to compute accuracy.
            dataset_cfg: Configuration for the dataset (to handle keypoint mappings).
            num_batches: Number of batches to run per group.
        """
        self.model.eval()
        results_csv = []
        
        # Determine device
        device = next(self.model.parameters()).device

        for group_name, config in self.groups.items():
            logger.info(f"Starting {group_name}")
            self.current_mask_config = config
            if self.current_mask_config is None: # Legacy safety
                 logger.warning(f"Group {group_name} config is None, skipping.")
                 continue

            # --- EXPERIMENT 3 FIX: Enforce Official Evaluator ---
            # We re-instantiate the evaluator for each group to ensure clean state and correct config.
            # keypoint_list comes from dataset_cfg (usually 14 LSP joints)
            # pelvis_ind=39 is critical for HMR2 (Root alignment)
            try:
                evaluator = Evaluator(
                    dataset_length=int(1e8), 
                    keypoint_list=dataset_cfg.KEYPOINT_LIST, 
                    pelvis_ind=39, 
                    metrics=['mode_mpjpe', 'mode_re']
                )
                # logger.info(f"Initialized Official Evaluator for {group_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Official Evaluator: {e}")
                # Fallback to the passed argument if instantiation fails (unlikely)
                pass 
            # ----------------------------------------------------

            self.layer_metrics.clear()
            
            # Reset evaluator for each group
            # Manually reset because Evaluator class has no reset()
            if hasattr(evaluator, 'metrics'):
                for metric in evaluator.metrics:
                    if hasattr(evaluator, metric):
                        # Re-init zero array of same size
                        arr = getattr(evaluator, metric)
                        setattr(evaluator, metric, np.zeros_like(arr))
                evaluator.counter = 0
            
            with torch.no_grad():
                for i, batch in enumerate(tqdm(data_loader, desc=f"Experiment Group: {group_name}")):
                    
                    if i % 10 == 0:
                         logger.info(f"DEBUG Loop: i={i}, num_batches={num_batches}")
                    
                    # Forward pass without forced interpolation
                    
                    if i >= num_batches: 
                        logger.info("DEBUG Loop: Reached limit. Breaking.")
                        break
                    
                    batch = self.wrapper.to_device(batch, device)
                    
                    if isinstance(batch, dict) and 'img' in batch:
                        _, _, H_img, W_img = batch['img'].shape
                        # Assuming patch size 16
                        self.current_feature_grid = (H_img // 16, W_img // 16)
                    else:
                        self.current_feature_grid = None

                    # --- HMR2/HSMR SPECIFIC FIX: Center Crop Awareness ---
                    # Must apply AFTER auto-detection to override it.
                    target_name = self.output_dir.name.lower()
                    if any(x in target_name for x in ['hmr2', 'hsmr', '4d-humans']):
                         # logger.info(f"DEBUG: Foracing Grid (16, 12) for {target_name}")
                         self.current_feature_grid = (16, 12)
                    # ------------------------------------------------

                    # Store Keypoints for KTI mapping AND Pre-compute Adjacency
                    self.current_keypoints = None
                    self.current_batch_adj = None
                    
                    if isinstance(batch, dict):
                        if 'keypoints_2d' in batch:
                            self.current_keypoints = batch['keypoints_2d']
                        elif 'keypoints' in batch:
                            self.current_keypoints = batch['keypoints']
                            
                        if self.current_keypoints is not None:
                            with torch.no_grad():
                                self.current_batch_adj = self.compute_kmi_adjacency(batch)
                        
                    # Forward pass via wrapper
                    out = self.wrapper(batch)
                    
                    # Accuracy Evaluation
                    evaluator(out, batch)
            
            # Aggregate Metrics from Evaluator
            metrics_dict = evaluator.get_metrics_dict()
            avg_mpjpe = metrics_dict.get('mode_mpjpe', 0.0) # Official PA-MPJPE in mm
            
            # Aggregate
            avg_entropy = np.mean([np.mean(v['entropy']) for v in self.layer_metrics.values() if v['entropy']])
            avg_kmi = np.mean([np.mean(v['kmi']) for v in self.layer_metrics.values() if v['kmi']]) if any(v['kmi'] for v in self.layer_metrics.values()) else 0.0
            avg_rank = np.mean([np.mean(v['rank']) for v in self.layer_metrics.values() if v['rank']]) if any(v['rank'] for v in self.layer_metrics.values()) else 0.0
            avg_mad = np.mean([np.mean(v['dist']) for v in self.layer_metrics.values() if v.get('dist')]) if any(v.get('dist') for v in self.layer_metrics.values()) else 0.0

            row = {
                'Group': group_name,
                'MPJPE': avg_mpjpe, 
                'Avg_Entropy': avg_entropy,
                'Avg_KTI': avg_kmi,
                'Avg_Rank': avg_rank,
                'Avg_MAD': avg_mad,
                'Mask_Type': config.get('mode', 'none'),
                # Add full config for reproducibility
                'Mask_Layers': str(config.get('mask_layers', [])),
                'Hybrid_Config': str(config.get('layer_modes', {}))
            }
            results_csv.append(row)
            
            # --- Incremental Save ---
            out_file = self.output_dir / 'results.csv'
            # Check if file exists to determine header
            write_header = not out_file.exists()
            pd.DataFrame([row]).to_csv(out_file, mode='a', header=write_header, index=False)
            logger.info(f"Saved results for {group_name} to {out_file}. MPJPE={avg_mpjpe:.2f} Rank={avg_rank:.4f}")
            # ------------------------
            
            
            # Save Layer-wise Metrics (Critical for Paper 1 Plots: Entropy/KTI vs Depth)
            # Need to serialize defaultdict(list)
            serializable_metrics = {}
            for layer_idx, metrics in self.layer_metrics.items():
                serializable_metrics[layer_idx] = {
                    'entropy': [float(x) for x in metrics['entropy']],
                    'kmi': [float(x) for x in metrics['kmi']],
                    'rank': [float(x) for x in metrics['rank']],
                    'dist': [float(x) for x in metrics.get('dist', [])]
                }
            
            import json
            metrics_file = self.output_dir / f'layer_metrics_{group_name.replace("/","-")}.json'
            with open(metrics_file, 'w') as f:
                json.dump(serializable_metrics, f)            
        # Save Final (Optional backup)
        df = pd.DataFrame(results_csv)
        self._plot_results(df)

    def _plot_results(self, df):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('MPJPE', color='tab:red')
        # ...
        plt.savefig(self.output_dir / 'diagnostic_plot.png')


    def add_adaptive_group(self, split_layer_1, split_layer_2=None, name=None):
        """
        Dynamically add a masking group based on suggested split layers.
        Supports 2-stage replacement (ViT -> Mamba -> GCN) via Masking Proxy.
        """
        if split_layer_2 is None:
            # Fallback for old single-split calls
            split_layer_2 = min(split_layer_1 + 8, self.total_layers)
            
        if name is None:
            name = f"Adaptive-{split_layer_1}-{split_layer_2}"
            
        # Proxy Logic:
        # Layers 0 -> split_layer_1: ViT (No Mask)
        # Layers split_layer_1 -> split_layer_2: Mamba (Soft Mask / Connected)
        # Layers split_layer_2 -> End: GCN (Hard Mask / Rigid)
        
        layer_modes = {}
        for i in range(self.total_layers):
            if i < split_layer_1:
                pass # No mask
            elif i < split_layer_2:
                layer_modes[i] = 'soft' # Mamba Proxy
            else:
                layer_modes[i] = 'hard' # GCN Proxy
                
        self.groups[name] = {
            'mask_layers': list(range(split_layer_1, self.total_layers)),
            'mode': 'hybrid',
            'layer_modes': layer_modes
        }
        logger.info(f"Added Adaptive Group: {name} (L{split_layer_1}: Soft, L{split_layer_2}: Hard)")

    def analyze_metrics_and_suggest_split(self):
        """
        Analyze collected Control metrics (Entropy/Rank) to find 2 optimal replacement points.
        Returns: (split_1, split_2)
        """
        if not self.layer_metrics:
            logger.warning("No metrics collected for adaptive analysis.")
            return (8, 24)

        # Calculate Average Entropy per layer across batches
        layer_indices = sorted(self.layer_metrics.keys())
        entropies = []
        for l in layer_indices:
            vals = self.layer_metrics[l]['entropy']
            if vals:
                entropies.append(np.mean(vals))
            else:
                entropies.append(100.0) # Dummy high
        
        # 1. First Dip (Global -> Focal)
        # Simple heuristic: Find local minimum in first half
        # Or look for steepest drop? 
        # User said: "Entropy first drops to lowest"
        
        # Smooth with moving average window=3
        entropies_smooth = np.convolve(entropies, np.ones(3)/3, mode='same')
        
        # Find local minima
        # But we need robustness.
        # Let's pick the absolute minimum in the first 12 layers?
        # Or find where derivative changes sign.
        
        split_1 = 8 # Default
        split_2 = 24 # Default
        
        try:
             # Find first significant dip
             # Usually around layer 4-8.
             min_idx = np.argmin(entropies[:16]) # Search first half
             split_1 = layer_indices[min_idx]
             
             # Find second dip (or steady state)
             # User said "Entropy drops again".
             # Actually, often entropy goes UP after first dip (Mixing), then down again.
             # So we look for minimum in the second half.
             min_idx_2 = np.argmin(entropies[16:]) + 16
             split_2 = layer_indices[min_idx_2]
             
             # Ensure ordering
             if split_2 <= split_1:
                 split_2 = split_1 + 4
                 
        except Exception as e:
            logger.error(f"Adaptive Analysis Failed: {e}")
            
        logger.info(f"Adaptive Suggestions based on Entropy Curve: {split_1} (Mamba Start), {split_2} (GCN Start)")
        return (split_1, split_2)

def get_wrapper(model, model_name):
    """Factory to get the right wrapper for a model."""
    name_lower = model_name.lower()
    if 'hmr2' in name_lower or '4d-humans' in name_lower:
        return HMR2Wrapper(model)
    elif 'camerahmr' in name_lower: # Add this
        return CameraHMRWrapper(model)
    elif 'hsmr' in name_lower:
        return HSMRWrapper(model)
    elif 'prompthmr' in name_lower or 'parameter-efficient' in name_lower: # prompt_hmr folder name
        return PromptHMRWrapper(model)
    elif 'mmhuman3d' in name_lower or 'spin' in name_lower or 'cliff' in name_lower or 'pare' in name_lower:
        return MMHuman3DWrapper(model)
    elif 'nlf' in name_lower:
        return NLFWrapper(model)
    elif 'siglip' in name_lower:
        return SigLIPWrapper(model)
    # Default fallback
    return HMR2Wrapper(model)
