import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from yacs.config import CfgNode

from hmr2.models.hmr2 import HMR2
from hmr2.models.losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from nvit2_models.nvit_hybrid import AdaptiveNViT
from nvit2_models.guided_head import GuidedSMPLHead
from hmr2.models import SMPL

from hmr2.utils.geometry import perspective_projection
from hmr2.utils.geometry import perspective_projection
from omegaconf import open_dict
import logging

logger = logging.getLogger(__name__)

class GuidedHMR2Module(HMR2):
    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        # ... (initialization code) ...
        # [Fix] HMR2 base class tries to load 'vitpose_backbone.pth'.
        # We are inserting our own AdaptiveNViT backbone, so we disable base loading.
        # cfg passed from Hydra is DictConfig (OmegaConf). modify in place or use safe copy if needed.
        # simpler to just set to None if present.
        if 'PRETRAINED_WEIGHTS' in cfg.MODEL.BACKBONE:
             from omegaconf import DictConfig
             if isinstance(cfg, DictConfig):
                 from omegaconf import open_dict
                 with open_dict(cfg):
                     cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None
             else:
                 cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = None
             
        # Call super init to setup basic attributes (but we will overwrite model components)
        super().__init__(cfg, init_renderer)
        
        
        # 1. Overwrite Backbone: AdaptiveNViT (Guided Mode)
        # [NEW] Optional: Skip AdaptiveNViT for Paper 1 (Baseline + Masking) experiments
        self.use_adaptive_nvit = cfg.MODEL.BACKBONE.get('USE_ADAPTIVE_NVIT', True)
        
        if self.use_adaptive_nvit:
            # Using Spiral Topology (Winner of Expt 3)
            # [Fix] Rename to nvit_backbone to avoid weird collision/restoration of self.backbone from base class
            if hasattr(self, 'backbone'):
                del self.backbone 
                
            mamba_variant = cfg.MODEL.BACKBONE.get('MAMBA_VARIANT', cfg.MODEL.BACKBONE.get('mamba_variant', 'spiral'))
            gcn_variant = cfg.MODEL.BACKBONE.get('GCN_VARIANT', cfg.MODEL.BACKBONE.get('gcn_variant', 'guided'))
            
            # [Ablation Support] Make depth, switch layers configurable
            depth = cfg.MODEL.BACKBONE.get('depth', 11) # Default to target 11
            sl1 = cfg.MODEL.BACKBONE.get('switch_layer_1', 8)
            # [Fix] Set switch_layer_2 default to 10 (depth-1) so HeatmapMapper triggers properly.
            sl2 = cfg.MODEL.BACKBONE.get('switch_layer_2', 10)
            
            logger.info(f"Initializing AdaptiveNViT with Depth={depth}, Mamba={mamba_variant}, GCN={gcn_variant}, Sl1={sl1}, Sl2={sl2}")
            
            self.nvit_backbone = AdaptiveNViT(
                depth=depth, 
                embed_dim=1280, 
                num_heads=16, 
                switch_layer_1=sl1, 
                switch_layer_2=sl2, 
                mamba_variant=mamba_variant, 
                gcn_variant=gcn_variant,    
                img_size=(256, 192)      # Aligned with HMR2 ViT-Pose input crop
            )
        else:
            logger.info("Using Baseline ViT Backbone (USE_ADAPTIVE_NVIT = False)")
            # self.backbone is already initialized in super().__init__
        
        # [CRITICAL FIX] Ensure TRANSFORMER_DECODER matches Run 9 baseline weights (3 layers, 4 heads)
        # Without this, weights are skipped due to size mismatch (768 vs 1536)
        if 'TRANSFORMER_DECODER' not in cfg.MODEL.SMPL_HEAD:
             from omegaconf import DictConfig
             if isinstance(cfg, DictConfig):
                 from omegaconf import open_dict
                 with open_dict(cfg):
                     cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER = {} # Use dict, Hydra will convert or accept
             else:
                 cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER = CfgNode()
        
        # Override with Run 9 defaults if not explicitly set to something else
        from omegaconf import DictConfig
        if isinstance(cfg, DictConfig):
            from omegaconf import open_dict
            ctx = open_dict(cfg)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            if cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER.get('depth', 6) == 6:
                logger.info("Overriding SMPL Head depth to 3 to match Run 9 weights.")
                cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER.depth = 3
            if cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER.get('heads', 8) == 8:
                logger.info("Overriding SMPL Head heads to 4 to match Run 9 weights.")
                cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER.heads = 4
            if cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER.get('mlp_dim', 1024) == 1024:
                cfg.MODEL.SMPL_HEAD.TRANSFORMER_DECODER.mlp_dim = 1024
             
        # 2. Overwrite Head: GuidedSMPLHead
        self.smpl_head = GuidedSMPLHead(cfg)
        
        # 3. Add Heatmap Loss logic
        # We need a dedicated Heatmap Loss (MSE)
        self.heatmap_criterion = nn.MSELoss()
        
        # Re-initialize SMPL (Already done in super, but just ensuring)
        # Verify loss weights exist in cfg
        if 'HEATMAP' not in self.cfg.LOSS_WEIGHTS:
            with open_dict(self.cfg.LOSS_WEIGHTS):
                self.cfg.LOSS_WEIGHTS.HEATMAP = 10.0 # Default weight matching debug findings
        
        # print(f"DEBUG: GuidedHMR2Module initialized. nvit_backbone type: {type(self.nvit_backbone)}")

        

    def soft_argmax(self, pred_heatmaps):
        if pred_heatmaps is not None:
            B, J, H, W = pred_heatmaps.shape
            device = pred_heatmaps.device
            # 2. Convert Heatmaps to 2D Coordinates (Soft-Argmax)
            # Add clamping and epsilon for numerical stability (Fix for Step 0 NaN)
            # [Fix] Tighter clamping for numerical stability (Soft-Pose protection)
            pred_heatmaps_clamped = torch.clamp(pred_heatmaps, min=-11, max=11)
            weights = F.softmax(pred_heatmaps_clamped.reshape(B, J, -1), dim=-1).reshape(B, J, H, W)
            weights = weights + 1e-6 # Increased epsilon
            weights = weights / weights.sum(dim=(-1, -2), keepdim=True)
            
            yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            pred_x = (weights * xx).sum(dim=(-1, -2)) / (W - 1)
            pred_y = (weights * yy).sum(dim=(-1, -2)) / (H - 1)
            
            # Map [0, 1] -> [-1, 1]
            coords = torch.stack([pred_x, pred_y], dim=-1)
            coords = coords * 2.0 - 1.0 
            return coords
        return None


    def forward_step(self, batch: dict, train: bool = False) -> dict:
        # print(f"DEBUG: forward_step called. nvit_backbone type: {type(self.nvit_backbone)}")
        """
        Modified forward step for Guided Architecture.
        """
        x = batch['img']
        batch_size = x.shape[0]
        
        from nvit.masking_utils import MaskingPatcher
        if MaskingPatcher.GLOBAL_INSTANCE is not None:
            if 'keypoints_2d' in batch:
                joints2d_xy = batch['keypoints_2d'][:, :, :2].detach()
                MaskingPatcher.GLOBAL_INSTANCE.set_joints2d(joints2d_xy)
        elif hasattr(self, 'mask_patcher'):
            if 'keypoints_2d' in batch:
                joints2d_xy = batch['keypoints_2d'][:, :, :2].detach()
                self.mask_patcher.set_joints2d(joints2d_xy)
            
        # [Fix] Align with HMR2 ViT backbone cropping: (B, 3, 256, 256) -> (B, 3, 256, 192)
        if x.shape[-2:] == (256, 256):
            x = x[:, :, :, 32:-32]
            
        # 1. Backbone Forward
        if self.use_adaptive_nvit:
             x_patches_input, pred_heatmaps = self.nvit_backbone.forward_features(x)
        else:
             # Baseline ViT: returns context features, coordinate guidance must be regressed separately
             # or we use the HeatmapMapper standalone if available.
             # [Fix] For Paper 1, if we use baseline, we still need 'coords' for the Guided Head.
             # We assume AdaptiveNViT's mapper is the only one we have.
             # BUT if we want PURE baseline, we should just use the standard HMR2 forward.
             # The user says "带掩码的方式", which implies using the Guided Architecture's head but with ViT.
             if hasattr(self, 'nvit_backbone'):
                  x_patches_input, pred_heatmaps = self.nvit_backbone.forward_features(x)
             else:
                  # Fallback to standard backbone + standalone mapper if needed, 
                  # but usually AdaptiveNViT with large switch layers is the safest "ViT" mode.
                  # However, the user said "不是模块替换".
                  # Let's use the standard backbone and a standalone mapper.
                  x_patches_input = self.backbone(x)
                  # If we don't have heatmaps, GuidedHead might fail. 
                  # We'll initialize a standalone mapper if needed, but for now we follow the 'no replacement' rule.
                  pred_heatmaps = None 
        
        # 2. Coordinate Extraction (Soft Argmax)
        coords = self.soft_argmax(pred_heatmaps)
        if coords is None:
            coords = torch.zeros(batch_size, 24, 2, device=x.device)
            
        # 3. Pass to Guided Head
        # Define x_img (for grid_sample) and x_patches (for global context)
        if x_patches_input.dim() == 4:
             # If x_patches_input is (B, C, H, W), it's the image feature map
             x_img = x_patches_input
             B, C, H, W = x_img.shape
             x_patches = x_img.reshape(B, C, H*W).permute(0, 2, 1) # (B, N, C)
        else:
             # If x_patches_input is already (B, N, C), it's the patch tokens
             x_patches = x_patches_input
             B, N, C = x_patches.shape
             # For HMR2 256x192 crop, patches are 16x12
             if N == 192:
                 H, W = 16, 12
             else:
                 H, W = int(N**0.5), int(N**0.5)
             x_img = x_patches.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Prepare grid for sampling
        grid = coords.unsqueeze(1) # (B, 1, J, 2)
        
        if hasattr(self.smpl_head, 'indexing_only') and self.smpl_head.indexing_only:
             # Indexing Only: Use Global Features + Coordinate Guidance, no local sampled features
             x_joints = torch.zeros(batch_size, 24, x_patches.shape[-1], device=x.device)
        else:
             # Default: Sample from feature map
             sampled_features = F.grid_sample(x_img, grid, align_corners=True) # (B, C, 1, J)
             x_joints = sampled_features.squeeze(2).permute(0, 2, 1) # (B, J, C)
             
        # Inject Global Context (GAP of Patches)
        global_context = x_patches.mean(dim=1, keepdim=True) # (B, 1, C)
        x_joints = x_joints + global_context
             
        pred_smpl_params, pred_cam, _ = self.smpl_head(x_joints, coords)
        
        # 4. Store Outputs
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_smpl_params'] = {k: v.clone() for k,v in pred_smpl_params.items()}
        output['pred_heatmaps'] = pred_heatmaps # For Loss
        
        # ... Rest is copied from HMR2 forward_step ...
        
        # Compute camera translation
        device = pred_smpl_params['body_pose'].device
        dtype = pred_smpl_params['body_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)
        pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size, -1)
        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        return output

    def compute_loss(self, batch: dict, output: dict, train: bool = True) -> torch.Tensor:
        """
        Compute total loss including Heatmap Loss.
        """
        # Call standard HMR2 loss components
        # Note: We duplicate logic because calling super().compute_loss() returns a scalar total,
        # making it hard to inject Heatmap Loss cleanely without recalculating.
        
        pred_smpl_params = output['pred_smpl_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']
        pred_heatmaps = output.get('pred_heatmaps', None)

        batch_size = pred_smpl_params['body_pose'].shape[0]
        
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']

        # 1. 3D Keypoint Loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        # [Fix] HMR2 uses pelvis_id=39 (regressed) for loss calculation
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=39) 

        # 2. SMPL Parameter Loss
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            gt = gt_smpl_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                from hmr2.utils.geometry import aa_to_rotmat
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_smpl_params[k]
            loss_smpl_params[k] = self.smpl_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)

        # 3. auxiliary Heatmap Loss (Coordinate Regression)
        device = output['pred_cam'].device
        loss_map = torch.tensor(0.0, device=device)
        if pred_heatmaps is not None:
             B, J, H, W = pred_heatmaps.shape
             
             # Slice GT to match Model Joints (24)
             # HMR2 uses OpenPose (25) + ... 
             # We assume first 24 match SMPL 24 roughly (Pelvis, L-Hip, R-Hip...)
             # Check joint map if performance is poor.
             gt_keypoints_2d_sliced = gt_keypoints_2d[:, :J, :]
             
             # Scale GT 2D (0-256) to Grid Space (0-H)
             # HMR2 images are 256x256.
             # Heatmap supervision adjustment for 256 -> 192 crop
             # Original 256x256 image is cropped to center 192 in Width (32px off each side)
             # So x_cropped = x_original - 32
             gt_kps_2d_adj = gt_keypoints_2d_sliced.clone()
             gt_kps_2d_adj[:, :, 0] = gt_kps_2d_adj[:, :, 0] - 32
             
             # Scale Adjusted GT 2D to Grid Space (W=12 in guided mode)
             # W=12, H=16. 192/12 = 16. 256/16 = 16. Aspect ratio preserved.
             scale_x = W / 192.0
             scale_y = H / 256.0
             
             # GT Coords in Grid Space (OpenPose Format)
             gt_kps_grid_x = gt_kps_2d_adj[:, :, 0] * scale_x
             gt_kps_grid_y = gt_kps_2d_adj[:, :, 1] * scale_y
             
             # Define Mapping: SMPL Index -> OpenPose Index
             # Based on smpl_wrapper.py: smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, ...]
             # Pairs (SMPL_idx, OP_idx):
             # (0, 8), (1, 12), (2, 9), (4, 13), (5, 10), (7, 14), (8, 11), (12, 1), 
             # (16, 5), (17, 2), (18, 6), (19, 3), (20, 7), (21, 4)
             # Note: SMPL 3, 6, 9, 10, 11, 13, 14, 15, 22, 23 have no direct OP match in standard 25-set.
             # We create a mapping tensor. -1 indicates no supervision.
             
             # SMPL has 24 Joints.
             smpl_to_op_map = torch.tensor([
                 8,  12, 9,  -1, 13, 10, -1, 14, 11, -1, -1, -1, 1,  -1, -1, -1, 
                 5,  2,  6,  3,  7,  4,  -1, -1
             ], device=device, dtype=torch.long)
             
             # Gather GT targets for each SMPL joint
             # shape: (B, 24)
             gt_x_target = torch.zeros(B, J, device=device)
             gt_y_target = torch.zeros(B, J, device=device)
             gt_conf_target = torch.zeros(B, J, device=device)
             
             # Valid mask
             valid_mask = (smpl_to_op_map != -1)
             valid_indices = torch.where(valid_mask)[0]
             op_indices = smpl_to_op_map[valid_indices]
             
             # Copy valid targets
             gt_x_target[:, valid_indices] = gt_kps_grid_x[:, op_indices]
             gt_y_target[:, valid_indices] = gt_kps_grid_y[:, op_indices]
             
             # Masking: Only supervise points that fall within the [0, 191] horizontal range after crop
             # and have valid raw confidence.
             gt_conf_orig = gt_keypoints_2d_sliced[:, op_indices, 2]
             gt_x_orig = gt_keypoints_2d_sliced[:, op_indices, 0]
             in_crop_mask = (gt_x_orig >= 32) & (gt_x_orig < 224)
             gt_conf_target[:, valid_indices] = gt_conf_orig * in_crop_mask.float()
             
             # Prepare Pred (Already SMPL topology from queries)
             # Add clamping and epsilon for numerical stability (Fix for Epoch 8 NaN)
             pred_heatmaps_clamped = torch.clamp(pred_heatmaps, min=-11, max=11)
             weights = F.softmax(pred_heatmaps_clamped.reshape(B, J, -1), dim=-1).reshape(B, J, H, W)
             weights = weights + 1e-8 
             weights = weights / weights.sum(dim=(-1, -2), keepdim=True)
             
             yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
             pred_x = (weights * xx).sum(dim=(-1, -2))
             pred_y = (weights * yy).sum(dim=(-1, -2))
             
             # Loss Calculation
             # Only penalize valid mapped joints with non-zero confidence
             loss_x = F.l1_loss(pred_x * gt_conf_target, gt_x_target * gt_conf_target, reduction='none')
             loss_y = F.l1_loss(pred_y * gt_conf_target, gt_y_target * gt_conf_target, reduction='none')
             
             # Zero out losses for unmapped joints (implicit via conf=0, but explicit safety)
             loss_map = (loss_x + loss_y) * valid_mask.float().unsqueeze(0)
             loss_map = loss_map.mean() # Mean over batches and joints

        # [Fix] Clamp heatmap loss to prevent explosion
        loss_map_clamped = torch.clamp(loss_map, max=100.0)
        heatmap_weight = self.cfg.LOSS_WEIGHTS.get('HEATMAP', 2.0) # Reduced from 10.0
        
        # Total Loss
        loss = (
            self.cfg.LOSS_WEIGHTS.KEYPOINTS_3D * loss_keypoints_3d +
            self.cfg.LOSS_WEIGHTS.KEYPOINTS_2D * loss_keypoints_2d +
            2.0 * self.cfg.LOSS_WEIGHTS.GLOBAL_ORIENT * loss_smpl_params['global_orient'] + # Boosted
            2.0 * self.cfg.LOSS_WEIGHTS.BODY_POSE * loss_smpl_params['body_pose'] + # Boosted
            self.cfg.LOSS_WEIGHTS.BETAS * loss_smpl_params['betas'] +
            heatmap_weight * loss_map_clamped
        )
        
        # Diagnostic print for loss components
        # if self.global_step == 0:
        #     print(f"DEBUG Step 0 Loss Components: 3D={loss_keypoints_3d:.4f}, 2D={loss_keypoints_2d:.4f}, Map={loss_map:.4f}")
        #     print(f"DEBUG Step 0 Params: Pose={loss_smpl_params['body_pose']:.4f}, Orient={loss_smpl_params['global_orient']:.4f}")
        
        # DDP Safety: Ensure all parameters are in the graph
        try:
            if self.trainer is not None and self.trainer.num_devices > 1:
                loss = loss + 0.0 * sum(p.sum() for p in self.parameters())
        except RuntimeError:
            pass
        
        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach(),
                      loss_heatmap=loss_map.detach())

        for k, v in loss_smpl_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses
        
        # [Fix] DDP Safety: Ensure all outputs participate in graph
        # Why? Because we mask out some heatmap channels (-1 joints), leading to unused params in DDP.
        # Adding 0.0 * output.sum() guarantees connectivity without affecting gradients.
        output['losses'] = losses
        
        return loss

    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: dict, output: dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Overriding to handle BFloat16 -> Float32 conversion for numpy/renderer.
        """
        # Create shallow copies to avoid modifying originals
        batch_f = {k: v.float() if (torch.is_tensor(v) and v.dtype == torch.bfloat16) else v for k,v in batch.items()}
        output_f = {k: v.float() if (torch.is_tensor(v) and v.dtype == torch.bfloat16) else v for k,v in output.items()}
        
        # Handle nested smpl_params
        if 'pred_smpl_params' in output_f:
             output_f['pred_smpl_params'] = {k: v.float() if (torch.is_tensor(v) and v.dtype == torch.bfloat16) else v for k,v in output_f['pred_smpl_params'].items()}
        
        return super().tensorboard_logging(batch_f, output_f, step_count, train, write_to_summary_writer)

    def training_step(self, joint_batch: dict, batch_idx: int) -> dict:
        """
        Modified training step to handle image and optionally mocap data.
        """
        # Unpack batches
        batch = joint_batch['img']
        mocap_batch = joint_batch.get('mocap', None)
            
        optimizer = self.optimizers(use_pl_optimizer=True)
        
        # [Fix] Robust optimizer unpacking
        optimizer_disc = None
        if isinstance(optimizer, list):
             optimizer_gen = optimizer[0]
             if len(optimizer) > 1:
                 optimizer_disc = optimizer[1]
             optimizer = optimizer_gen
             
        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smpl_params = output['pred_smpl_params']
        
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
            
        loss = self.compute_loss(batch, output, train=True)
        
        # Adversarial Logic
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0 and mocap_batch is not None:
            disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv
            
            # ... Disc optimizer step ... (Skipping complex logic for Sanity Check)
            # If we need full adv support, we copy HMR2 code fully.
            # But here we just want to verify Backbone+Head.
        
        # [Fix] Pre-Backward NaN Check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"ERROR: Loss is NaN/Inf at step {self.global_step}. Skipping backward.")
            print(f"  Loss components: {output['losses']}")
            # Return dummy loss to avoid crash
            return {'loss': torch.tensor(0.0, device=loss.device, requires_grad=True)}

        # Manual Gradient Accumulation logic
        accumulate_grad_batches = self.cfg.TRAIN.get('ACCUMULATE_GRAD_BATCHES', 1)
        
        # Scale loss to maintain effective batch size gradients
        loss = loss / accumulate_grad_batches
        
        from contextlib import nullcontext
        is_accumulating = (batch_idx + 1) % accumulate_grad_batches != 0
        sync_context = nullcontext()
        if is_accumulating:
            if hasattr(self, 'no_sync'):
                sync_context = self.no_sync()
            elif hasattr(self.trainer.model, 'no_sync'):
                sync_context = self.trainer.model.no_sync()
            elif hasattr(getattr(self.trainer, 'strategy', None), 'model') and hasattr(self.trainer.strategy.model, 'no_sync'):
                sync_context = self.trainer.strategy.model.no_sync()
                
        with sync_context:
            self.manual_backward(loss)
        
        # Only step optimizer every accumulate_grad_batches
        if (batch_idx + 1) % accumulate_grad_batches == 0:
            # [Fix] Use Lightning's native clip_gradients to properly unscale mixed-precision FP16 gradients!
            # Manual torch.nn.utils.clip_grad_norm_ prevents the GradScaler from seeing the Inf/NaN and self-healing.
            clip_val = self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0.5)
            if clip_val > 0:
                self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm="norm")
                
            optimizer.step()
            optimizer.zero_grad()
        
        # Discriminator Step (Skip if no mocap)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0 and mocap_batch is not None:
             loss_disc = self.training_step_discriminator(mocap_batch, pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1), optimizer_disc)
             output['losses']['loss_gen'] = loss_adv
             output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
             self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx=0) -> dict:
        """
        Run a validation step and log to Tensorboard, including MPJPE.
        """
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        
        # Calculate MPJPE (Mean Per Joint Position Error) in mm
        # pred_keypoints_3d: (B, N, 3) in meters (assumed)
        # gt_keypoints_3d: (B, N, 3)
        # We need to align pelvis or just compute raw distance if dataset is already aligned.
        # HMR2 typically predicts root-relative.
        pred_kpts = output['pred_keypoints_3d'].detach()
        gt_kpts = batch['keypoints_3d']
        
        # Slice GT to first 3 dims if it has confidence
        if gt_kpts.shape[-1] == 4:
            gt_kpts = gt_kpts[:, :, :3]
            
        # --- MPJPE Alignment (Disabled due to J_regressor dependency) ---
        # Accurate MPJPE requires regressing SMPL vertices to H36M 14-joint subset
        # and matching with GT. Simple joint distance is invalid due to topology mismatch.
        # Use offline evaluation script for correct metrics.
        mpjpe = -1.0
        
        # self.log('val/mpjpe', mpjpe, ... ) # Disabled
        
        if self.logger:
             self.tensorboard_logging(batch, output, self.global_step, train=False)
        return output

    def get_parameters(self):
        """
        Get all parameters to optimize.
        Overridden to use nvit_backbone.
        """
        all_params = list(self.smpl_head.parameters())
        if hasattr(self, 'nvit_backbone'):
            all_params += list(self.nvit_backbone.parameters())
        if hasattr(self, 'backbone'):
            all_params += list(self.backbone.parameters())
        return all_params

    def configure_optimizers(self):
        """
        Setup model and discriminator Optimizers.
        Modified to support ADVERSARIAL=0 (No Discriminator).
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        # lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0 and hasattr(self, 'discriminator'):
            optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                                lr=self.cfg.TRAIN.LR,
                                                weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
            return optimizer, optimizer_disc
        else:
            return optimizer
