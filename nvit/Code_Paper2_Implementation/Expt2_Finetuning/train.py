import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import sys
import os
from pathlib import Path
import torch.nn.functional as F

# Setup paths
ROOT = '/home/yangz/NViT-master/nvit/Code_Paper2_Implementation'
sys.path.append(ROOT)

from nvit2_models.nvit_hybrid import AdaptiveNViT
from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
from hmr2.models.heads import build_smpl_head
from hmr2.models.losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from hmr2.models.smpl_wrapper import SMPL
from hmr2.utils.pose_utils import eval_pose
from hmr2.models.discriminator import Discriminator
from hmr2.utils.geometry import aa_to_rotmat

def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Convert 3x3 rotation matrix to Rodrigues vector in a differentiable way.
    """
    if rotation_matrix.dim() == 4: 
        rotation_matrix = rotation_matrix.reshape(-1, 3, 3)
    
    batch = rotation_matrix
    
    # trace = r00 + r11 + r22
    trace = batch[:, 0, 0] + batch[:, 1, 1] + batch[:, 2, 2]
    cos = (trace - 1) / 2
    cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos)
    
    sin = torch.sqrt(1 - cos * cos)
    
    # Avoid div by zero for small angles
    mask = (sin > 1e-4)
    
    rx = batch[:, 2, 1] - batch[:, 1, 2]
    ry = batch[:, 0, 2] - batch[:, 2, 0]
    rz = batch[:, 1, 0] - batch[:, 0, 1]
    
    factor = torch.zeros_like(theta)
    factor[mask] = theta[mask] / (2 * sin[mask])
    factor[~mask] = 0.5
    
    axis = torch.stack((rx, ry, rz), dim=1)
    return axis * factor.unsqueeze(1)

class IndependentSurgicalModel(pl.LightningModule):
    def __init__(self, m_cfg, mamba_v, gcn_v, s1, s2, freeze):
        super().__init__()
        self.save_hyperparameters(ignore=['m_cfg'])
        self.m_cfg = m_cfg
        
        # 1. Backbone
        self.backbone = AdaptiveNViT(
             depth=32, embed_dim=1280, num_heads=16,
             switch_layer_1=s1, 
             switch_layer_2=s2,
             mamba_variant=mamba_v,
             gcn_variant=gcn_v,
             img_size=(256, 256)
        )
        
        # 2. SMPL Head
        self.smpl_head = build_smpl_head(m_cfg)
        
        # 3. SMPL
        smpl_cfg = {k.lower(): v for k,v in dict(m_cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg)
        
        # 4. Losses
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.smpl_parameter_loss = ParameterLoss()
        
        if freeze:
            self.backbone.surgical_freeze(freeze_depth=s1)
            
        # 5. Adversarial Setup
        self.discriminator = Discriminator()
        self.automatic_optimization = False # Manual optimization for GAN

    def forward(self, x):
        # Backbone now returns single feature tensor by default
        features = self.backbone(x)
        # smpl_head returns (pred_smpl_params, pred_cam, pred_smpl_params_list)
        pred_smpl_params, pred_cam, _ = self.smpl_head(features)
        
        # Calculate 3D/2D keypoints using SMPL
        smpl_output = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        
        # Project to 2D
        from hmr2.utils.geometry import perspective_projection
        batch_size = pred_keypoints_3d.shape[0]
        device = pred_keypoints_3d.device
        focal_length = torch.full((batch_size, 2), 5000.0, device=device)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d, translation=pred_cam, focal_length=focal_length)
        
        output = {
            'pred_smpl_params': {k: v for k,v in pred_smpl_params.items()},
            'pred_cam': pred_cam,
            'pred_keypoints_3d': pred_keypoints_3d,
            'pred_keypoints_2d': pred_keypoints_2d
        }
        return output

    def compute_loss(self, batch, output):
        pred_smpl_params = output['pred_smpl_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']
        
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smpl_params = batch['smpl_params']
        has_smpl_params = batch['has_smpl_params']
        
        loss_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d)
        
        loss_smpl = 0
        for k, pred in pred_smpl_params.items():
            if k == 'betas':
                gt = gt_smpl_params[k].view(pred.shape[0], -1)
                loss_smpl += self.smpl_parameter_loss(pred, gt, has_smpl_params[k])
            else:
                # pred is rotation matrix (B, ..., 3, 3)
                # Convert to AA (B, ..., 3)
                batch_size = pred.shape[0]
                pred_aa = rotation_matrix_to_angle_axis(pred.reshape(-1, 3, 3)).reshape(batch_size, -1)
                gt = gt_smpl_params[k].view(batch_size, -1)
                loss_smpl += self.smpl_parameter_loss(pred_aa, gt, has_smpl_params[k]) * 0.001 # Body Pose 0.001
            
        # Weights from HMR2 Config:
        # KEYPOINTS_3D: 0.05
        # KEYPOINTS_2D: 0.01
        # GLOBAL_ORIENT: 0.001 (Handled inside loop above roughly)
        # BODY_POSE: 0.001
        # BETAS: 0.0005
        
        # Override previous generic summation for SMPL
        loss_smpl = 0
        
        # Body Pose: Pred (B, 23, 3, 3) -> AA (B, 69)
        pred_body_pose_aa = rotation_matrix_to_angle_axis(pred_smpl_params['body_pose'].reshape(-1, 3, 3)).reshape(batch_size, -1)
        loss_smpl += self.smpl_parameter_loss(pred_body_pose_aa, gt_smpl_params['body_pose'].view(batch_size, -1), has_smpl_params['body_pose']) * 0.001
        
        # Global Orient: Pred (B, 1, 3, 3) -> AA (B, 3)
        pred_global_orient_aa = rotation_matrix_to_angle_axis(pred_smpl_params['global_orient'].reshape(-1, 3, 3)).reshape(batch_size, -1)
        loss_smpl += self.smpl_parameter_loss(pred_global_orient_aa, gt_smpl_params['global_orient'].view(batch_size, -1), has_smpl_params['global_orient']) * 0.001
        
        # Betas: (B, 10)
        loss_smpl += self.smpl_parameter_loss(pred_smpl_params['betas'], gt_smpl_params['betas'].view(batch_size, -1), has_smpl_params['betas']) * 0.0005
        
        loss = 0.05 * loss_3d + \
               0.01 * loss_2d + \
               loss_smpl
        return loss

    def training_step_discriminator(self, batch, body_pose, betas, optimizer_disc):
        batch_size = body_pose.shape[0]
        gt_body_pose = batch['smpl_params']['body_pose']
        gt_betas = batch['smpl_params']['betas']
        
        # Convert GT AA to RotMat for Discriminator
        gt_rotmat = aa_to_rotmat(gt_body_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        
        # Fake Pass
        disc_fake_out = self.discriminator(body_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        
        # Real Pass
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        
        loss_disc = loss_fake + loss_real
        
        optimizer_disc.zero_grad()
        self.manual_backward(loss_disc)
        optimizer_disc.step()
        return loss_disc.detach()

    def training_step(self, joint_batch, batch_idx):
        if isinstance(joint_batch, dict) and 'img' in joint_batch:
             batch = joint_batch['img']
        else:
             batch = joint_batch
             
        opt_gen, opt_disc = self.optimizers()
        
        # --- Generator Step ---
        x = batch['img']
        output = self.forward(x)
        pred_smpl_params = output['pred_smpl_params']
        batch_size = x.shape[0]
        
        # Recon Loss
        loss_recon = self.compute_loss(batch, output)
        
        # Adv Loss (Generator wants Disc to predict 1)
        disc_out = self.discriminator(pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1))
        loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
        
        # Total Gen Loss
        # Adv Weight from config = 0.0005
        loss_gen = loss_recon + 0.0005 * loss_adv
        
        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        # Clip Grads
        self.clip_gradients(opt_gen, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_gen.step()
        
        # --- Discriminator Step ---
        loss_disc = self.training_step_discriminator(batch, pred_smpl_params['body_pose'].reshape(batch_size, -1), pred_smpl_params['betas'].reshape(batch_size, -1), opt_disc)
        
        self.log('train/loss_gen', loss_gen, prog_bar=True)
        self.log('train/loss_disc', loss_disc, prog_bar=True)
        self.log('train/loss_recon', loss_recon, prog_bar=True)
        
        return loss_gen
        
    def validation_step(self, batch, batch_idx):
        # Val dataloader is simple, so batch is the dict directly
        x = batch['img']
        output = self.forward(x)
        loss = self.compute_loss(batch, output)
        self.log('val_loss', loss, prog_bar=True)
        
        # --- MPJPE Evaluation (H36M 14-Joints) ---
        # 3DPW-TEST Keypoint List (from hmr2/configs/datasets_eval.yaml)
        J14_INDICES = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 43]
        
        pred_kpts = output['pred_keypoints_3d'].detach() # (B, 44, 3)
        gt_kpts = batch['keypoints_3d'][:, :, :-1] # (B, 44, 3)
        
        # Align pelvis (Idx 0 of Body25 is usually Pelvis? No, openpose has specific pelvis)
        # HMR2 Eval uses PELVIS_IND=0? No, let's use the one from config if possible.
        # But commonly we just zero-center based on the mean of hips or explicit pelvis.
        # For simplicity, we use center of Reference (Pelvis) which is often index 39 (25+14) or similar in some sets
        # BUT hmr2 models/losses.py uses pelvis_id=25+14=39.
        pelvis_idx = 39 
        
        pred_pelvis = pred_kpts[:, [pelvis_idx], :]
        gt_pelvis = gt_kpts[:, [pelvis_idx], :]
        
        pred_kpts_aligned = pred_kpts - pred_pelvis
        gt_kpts_aligned = gt_kpts - gt_pelvis
        
        # Select J14
        pred_j14 = pred_kpts_aligned[:, J14_INDICES, :]
        gt_j14 = gt_kpts_aligned[:, J14_INDICES, :]
        
        # Calc MPJPE - Move to CPU to avoid Half/SVD issues
        mpjpe_batch, _ = eval_pose(pred_j14.cpu().float(), gt_j14.cpu().float())
        mpjpe = mpjpe_batch.mean()
        
        self.log('val_mpjpe', mpjpe, prog_bar=True)
        
        return {'val_loss': loss, 'val_mpjpe': mpjpe}

    def configure_optimizers(self):
        optimizer_gen = torch.optim.AdamW(self.backbone.parameters(), lr=self.m_cfg.TRAIN.LR)
        # Include SMPL head in gen optimizer? Usually yes.
        # Original code: params=param_groups (backbone + smpl_head).
        # We should probably optimize all params except discriminator in gen.
        gen_params = list(self.backbone.parameters()) + list(self.smpl_head.parameters())
        optimizer_gen = torch.optim.AdamW(gen_params, lr=self.m_cfg.TRAIN.LR)
        
        optimizer_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=self.m_cfg.TRAIN.LR)
        
        return [optimizer_gen, optimizer_disc]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mamba', type=str, default='positive')
    parser.add_argument('--gcn', type=str, default='skeleton')
    parser.add_argument('--s1', type=int, default=8)
    parser.add_argument('--s2', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--devices', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()
    
    _, m_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    
    model = IndependentSurgicalModel(
        m_cfg=m_cfg,
        mamba_v=args.mamba,
        gcn_v=args.gcn,
        s1=args.s1,
        s2=args.s2,
        freeze=True
    )
    
    # 3. Data Module
    from hmr2.configs import dataset_eval_config
    from hmr2.datasets import ImageDataset
    from torch.utils.data import DataLoader, IterableDataset
    import webdataset as wds
    from bio_dataset import BioMambaDataset, BioWebDataset # [New Import]
    import glob
    import os

    # Updated SurgicalDataModule with Validation
    class SurgicalDataModule(pl.LightningDataModule):
        def __init__(self, m_cfg, batch_size):
            super().__init__()
            self.m_cfg = m_cfg
            self.batch_size = batch_size
            self.dataset_cfg = dataset_eval_config()
            self.train_ds = None
            self.val_ds = None
            
        def setup(self, stage=None):
            # 1. 3DPW - Load as Map, wrap to infinite cycle
            cfg_3dpw = {k.lower(): v for k, v in self.dataset_cfg['3DPW-TEST'].items()}
            cfg_3dpw['dataset_file'] = '/home/yangz/4D-Humans/data/metadata/3dpw_test.npz'
            cfg_3dpw['img_dir'] = '/home/yangz/4D-Humans/data/3DPW'
            ds_3dpw = BioMambaDataset(self.m_cfg, **cfg_3dpw, train=True)
            
            def cycle_wrapper(ds):
                while True:
                    for i in range(len(ds)):
                        yield ds[i]
            
            ds_3dpw_inf = cycle_wrapper(ds_3dpw)
            
            datasets = [ds_3dpw_inf]
            weights = [0.2] # Base weight for 3DPW
            
            # 2. COCO (WebDataset)
            coco_path = "/home/yangz/4D-Humans/data/finetune_ext/coco"
            
            # Disable SUPPRESS_BAD_POSES to avoid missing prior file
            self.m_cfg.defrost()
            self.m_cfg.DATASETS.SUPPRESS_BAD_POSES = False
            self.m_cfg.freeze()

            if len(glob.glob(f"{coco_path}/*.tar")) > 0:
                print(f"✅ Found COCO in {coco_path}")
                url_coco = f"{coco_path}/{{000000..000034}}.tar"
                ds_coco_wds = ImageDataset.load_tars_as_webdataset(
                    self.m_cfg, urls=url_coco, train=True, epoch_size=100_000
                )
                ds_coco = BioWebDataset(ds_coco_wds)
                datasets.append(ds_coco)
                weights.append(0.4)
            else:
                print(f"⚠️ COCO not found in {coco_path}")

            # 3. H36M (WebDataset - Checks for existence)
            h36m_path = "/home/yangz/4D-Humans/data/finetune_ext/h36m"
            h36m_tars = glob.glob(f"{h36m_path}/*.tar")
            if len(h36m_tars) > 5: # Threshold to ensure it's not just metadata
                print(f"✅ Found H36M in {h36m_path}")
                # Construct range string dynamically? Or assume standard?
                # For safety, let's just use BraceExpand or simplified listing if possible.
                # But existing helper uses BraceExpand. We'll use a glob pattern or assume 00..312 like config.
                # If just a few files, we might be safer using `Dataset.load_tars_as_webdataset` on list of files?
                # No, ImageDataset expects url string.
                # Let's try to detect the max index.
                max_idx = len(h36m_tars) - 1 # Approximation
                # Padding to 6 digits
                max_str = f"{max_idx:06d}"
                url_h36m = f"{h36m_path}/{{000000..{max_str}}}.tar"
                
                ds_h36m_wds = ImageDataset.load_tars_as_webdataset(
                    self.m_cfg, urls=url_h36m, train=True, epoch_size=100_000
                )
                ds_h36m = BioWebDataset(ds_h36m_wds)
                datasets.append(ds_h36m)
                weights.append(0.4)
            else:
                print(f"⏳ H36M not yet fully uploaded in {h36m_path} (Found {len(h36m_tars)} tars)")

            # Normalize weights
            total_w = sum(weights)
            weights = [w/total_w for w in weights]
            
            print(f"🚀 Mixing Datasets with weights: {weights}")
            self.train_ds = wds.RandomMix(datasets, weights)

            # Validation (Keep 3DPW Fixed)
            self.val_ds = BioMambaDataset(self.m_cfg, **cfg_3dpw, train=False)
            
        def train_dataloader(self):
            # No shuffle for IterableDataset, handled by WebDataset shuffle
            return {'img': DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=4)}

        def val_dataloader(self):
             return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

    dm = SurgicalDataModule(m_cfg, batch_size=args.batch_size)
    
    # Checkpoint callback to capture best model
    # Ensure output dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        monitor='val_loss', mode='min', save_top_k=1,
        filename='latest_ft_model'
    )

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.devices, # Configurable
        strategy='auto', # Changed from DDP to auto for single-GPU simplicity
        precision='16-mixed',
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        benchmark=True # Enable cudnn autotuner
    )
    
    print(f"💉 Starting Independent Surgical Fine-Tuning: Mamba={args.mamba}, GCN={args.gcn}")
    trainer.fit(model, datamodule=dm)
    
    # --- Evaluation and Logging ---
    print("📊 Evaluating final model...")
    val_results = trainer.validate(model, datamodule=dm)
    final_loss = val_results[0]['val_loss']
    
    # Append to CSV
    csv_path = Path(ROOT) / "structure_rankings_finetuned.csv"
    if not csv_path.exists():
        with open(csv_path, 'w') as f:
            f.write("Mamba,GCN,Final_Loss\n")
            
    with open(csv_path, 'a') as f:
        f.write(f"{args.mamba},{args.gcn},{final_loss:.4f}\n")
    print(f"✅ Saved results to {csv_path}")

if __name__ == '__main__':
    main()
