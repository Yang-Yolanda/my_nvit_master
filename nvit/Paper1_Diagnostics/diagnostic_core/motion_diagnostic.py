
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

class MotionCoherenceEngine:
    """
    Diagnostic Engine for Paper 2: Evaluates Motion Coherence and Anatomical Realism.
    Combined Score (MCS) = w1 * DiscScore + w2 * KinematicScore + w3 * TemporalSmoothness
    """
    def __init__(self, discriminator: nn.Module = None, device: str = 'cuda'):
        self.discriminator = discriminator
        self.device = torch.device(device)
        if self.discriminator:
            self.discriminator.to(self.device).eval()

    def compute_anatomical_feasibility(self, smpl_pose: torch.Tensor) -> torch.Tensor:
        """
        Check joint angle limits. SMPL pose is typically (B, 24, 3) in AA or (B, 24, 3, 3) in RotMat.
        Simplified version: Check if rotations are within reasonable Euclidean norms.
        """
        if smpl_pose.dim() == 4: # RotMat (B, 24, 3, 3)
            # Trace of RotMat indicates angle. Trace = 1 + 2cos(theta)
            traces = torch.diagonal(smpl_pose, dim1=-2, dim2=-1).sum(-1)
            angles = torch.acos((traces - 1) / 2.0).nan_to_num(0.0)
            # Penalty for angles > 2 rad (approx 115 deg) - very rough proxy
            penalty = torch.relu(angles - 2.0).mean()
            return 1.0 - torch.tanh(penalty)
        else:
            # Assume Axis-Angle (B, 24*3) or (B, 24, 3)
            angles = torch.norm(smpl_pose.reshape(smpl_pose.shape[0], -1, 3), dim=-1)
            penalty = torch.relu(angles - 2.0).mean()
            return 1.0 - torch.tanh(penalty)

    def compute_temporal_smoothness(self, poses_seq: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, D) sequence of poses.
        Returns: 1.0 - average velocity (normalized).
        """
        if poses_seq.dim() < 3:
            return torch.tensor(1.0, device=self.device)
        
        diff = poses_seq[:, 1:] - poses_seq[:, :-1]
        velocity = torch.norm(diff, dim=-1).mean()
        return 1.0 / (1.0 + velocity)

    @torch.no_grad()
    def evaluate_mcs(self, batch_output: Dict[str, Any], sequence: bool = False) -> Dict[str, float]:
        """
        Computes MCS and its components.
        """
        # 1. Realism Score (Adversarial)
        disc_score = 0.0
        if self.discriminator and 'pred_smpl_params' in batch_output:
            params = batch_output['pred_smpl_params']
            rotmats = params['body_pose'] # (B, 23, 3, 3)
            shape = params['betas']
            
            # ST-GCN expects [B, (T), V, C] where C is 9 for RotMat
            # Reshape (B, 23, 3, 3) -> (B, 23, 9)
            disc_input = rotmats.reshape(rotmats.shape[0], 23, 9)
            d_out = self.discriminator(disc_input, shape)
            disc_score = torch.sigmoid(d_out).mean().item()

        # 2. Kinematic Score
        kin_score = 1.0
        if 'pred_smpl_params' in batch_output:
            kin_score = self.compute_anatomical_feasibility(batch_output['pred_smpl_params']['body_pose']).item()

        # 3. Temporal Score (if batch is a sequence)
        temp_score = 1.0
        if sequence and 'pred_smpl_params' in batch_output:
             temp_score = self.compute_temporal_smoothness(batch_output['pred_smpl_params']['body_pose']).item()

        # Combined MCS
        mcs = 0.4 * disc_score + 0.4 * kin_score + 0.2 * temp_score
        
        return {
            "mcs": mcs,
            "disc_realism": disc_score,
            "kinematic_feasibility": kin_score,
            "temporal_smoothness": temp_score
        }
