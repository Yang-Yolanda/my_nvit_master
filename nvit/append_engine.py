#!/home/yangz/.conda/envs/4D-humans/bin/python

# ==============================================================================
# 4. GAN Training Optimization (Paper 2 Style)
# ==============================================================================
def train_one_epoch_gan(args, student: nn.Module, teacher: nn.Module, discriminator: nn.Module,
                        data_loader: Iterable, 
                        optimizer_gen: torch.optim.Optimizer, 
                        optimizer_disc: torch.optim.Optimizer,
                        device: torch.device, epoch: int, loss_scaler, 
                        max_norm: float = 0, model_ema: Optional[ModelEma] = None, 
                        mixup_fn: Optional[Mixup] = None,
                        main_loss_coeff=1.0, original_loss_coeff=1.0, 
                        smpl_param_loss_fn=None):
    
    student.train()
    teacher.eval()
    discriminator.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_gen', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_disc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = 'Epoch: [{}] (GAN)'.format(epoch)
    print_freq = 50

    if smpl_param_loss_fn is None:
        smpl_param_loss_fn = ParameterLoss()

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch = move_to_device(batch, device)
        
        # 1. Unpack Batch
        if isinstance(batch, dict):
            samples = batch 
            targets = batch
        else:
            samples, targets = batch

        # ----------------------------------------------------------------------
        # Phase 1: Generator Step (Student + Distillation + Adv)
        # ----------------------------------------------------------------------
        with torch.amp.autocast('cuda'):
            # Teacher Forward (No Grad)
            with torch.no_grad():
                teacher_output = teacher(samples)
                # Ensure teacher output is consistent for distillation
                if isinstance(teacher_output, dict):
                    teacher_joints = teacher_output.get('pred_keypoints_3d')
                else:
                    teacher_joints = teacher_output

            # Student Forward
            student_output = student(samples)
            if isinstance(student_output, dict):
                pred_smpl_params = student_output['pred_smpl_params']
                pred_keypoints_3d = student_output['pred_keypoints_3d']
                pred_keypoints_2d = student_output['pred_keypoints_2d']
            else:
                 # Should not happen in HMR2 flow but safekeeping
                 raise ValueError("Student output must be dict for GAN training")

            # --- A. Reconstruction Losses (Paper 2 Weights) ---
            # 3D / 2D Losses are typically handled by orig_criterion wrapper, but we need granular control here
            # or we trust the orig_criterion to return the weighted sum. 
            # Paper 2: 0.05 * 3D + 0.01 * 2D
            # Let's calculate manually to match Paper 2 exactly 
            
            # Loss 2D/3D (assuming L1)
            gt_keypoints_2d = targets['keypoints_2d']
            gt_keypoints_3d = targets['keypoints_3d']
            
            loss_2d = F.l1_loss(pred_keypoints_2d, gt_keypoints_2d)
            loss_3d = F.l1_loss(pred_keypoints_3d, gt_keypoints_3d)

            # SMPL Losses (Split)
            loss_smpl = 0.0
            gt_smpl_params = targets['smpl_params']
            has_smpl_params = targets['has_smpl_params']
            
            # Body Pose (0.001)
            pred_body_pose = pred_smpl_params['body_pose'] # (B, 23, 3, 3)
            # Convert to AA for loss? HMR2 usually computes on Rotation Matrix directly if ParameterLoss supports it, 
            # Or converts. Paper 2 code converts Pred to AA for loss.
            # Let's check ParameterLoss. It usually takes (N, D).
            # Paper 2: loss += parameter_loss(pred_aa, gt_pose)
            
            # We will use the 'smpl_parameter_loss' logic from Paper 2
            # 1. Body Pose
            loss_smpl += smpl_param_loss_fn(pred_body_pose.reshape(pred_body_pose.shape[0], -1), 
                                            aa_to_rotmat(gt_smpl_params['body_pose'].reshape(-1, 3)).reshape(pred_body_pose.shape[0], -1),
                                            has_smpl_params['body_pose']) * 0.001
            
            # 2. Betas (0.0005)
            loss_smpl += smpl_param_loss_fn(pred_smpl_params['betas'], 
                                            gt_smpl_params['betas'], 
                                            has_smpl_params['betas']) * 0.0005
                                            
            loss_recon = 0.05 * loss_3d + 0.01 * loss_2d + loss_smpl

            # --- B. Distillation Loss (Paper 1 Feature) ---
            # We keep this to maintain student-teacher consistency
            # Assuming simple MSE/L1 on 3D joints
            loss_distill = F.mse_loss(pred_keypoints_3d, teacher_joints)

            # --- C. Adversarial Loss (Generator Part) ---
            # Generator wants Discriminator to output 1 (Real)
            # Input to Disc: Pose (RotMat) + Shape
            # Disc expects (B, 23*9) and (B, 10)
            disc_out = discriminator(pred_smpl_params['body_pose'].reshape(pred_body_pose.shape[0], -1), 
                                     pred_smpl_params['betas'])
            loss_adv = ((disc_out - 1.0) ** 2).sum() / disc_out.shape[0]

            # Total Generator Loss
            # Adv Weight: 0.0005 (Paper 2)
            # Distill Weight: main_loss_coeff
            loss_gen = original_loss_coeff * loss_recon + main_loss_coeff * loss_distill + 0.0005 * loss_adv

            if args.distributed:
                # DDP gradient sync fix
                loss_gen += 0.0 * sum(p.sum() for p in student.parameters() if p.requires_grad)

        # Update Generator
        optimizer_gen.zero_grad(set_to_none=True)
        loss_scaler(loss_gen, optimizer_gen, clip_grad=max_norm, parameters=student.parameters(), create_graph=False)
        
        # ----------------------------------------------------------------------
        # Phase 2: Discriminator Step
        # ----------------------------------------------------------------------
        with torch.amp.autocast('cuda'):
            # Real Input: GT Pose (RotMat) + GT Betas
            gt_body_pose_aa = gt_smpl_params['body_pose'] # (B, 23, 3)
            gt_rotmat = aa_to_rotmat(gt_body_pose_aa.reshape(-1, 3)).view(gt_body_pose_aa.shape[0], -1) # (B, 23*9)
            gt_betas = gt_smpl_params['betas']

            # Fake Input: Student Pred (Detached)
            fake_rotmat = pred_smpl_params['body_pose'].detach().reshape(gt_rotmat.shape[0], -1)
            fake_betas = pred_smpl_params['betas'].detach()

            # Forward
            disc_real = discriminator(gt_rotmat, gt_betas)
            disc_fake = discriminator(fake_rotmat, fake_betas)

            # LSGAN Loss
            loss_real = ((disc_real - 1.0) ** 2).sum() / disc_real.shape[0]
            loss_fake = ((disc_fake - 0.0) ** 2).sum() / disc_fake.shape[0]
            loss_disc = loss_real + loss_fake
            
            # Adv Weight (Paper 2 uses LOSS_WEIGHTS.ADVERSARIAL for Update? 
            # Usually Disc update is raw loss, Gen update is weighted)
            # HMR2 code: loss = cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
            # Assuming weight 0.0005
            loss_disc_weighted = 0.0005 * loss_disc

        # Update Discriminator
        optimizer_disc.zero_grad(set_to_none=True)
        loss_scaler(loss_disc_weighted, optimizer_disc, clip_grad=max_norm, parameters=discriminator.parameters(), create_graph=False)

        # ----------------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------------
        torch.cuda.synchronize()
        
        if model_ema is not None:
            model_ema.update(student)

        metric_logger.update(loss_gen=loss_gen.item())
        metric_logger.update(loss_disc=loss_disc_weighted.item())
        metric_logger.update(lr=optimizer_gen.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
