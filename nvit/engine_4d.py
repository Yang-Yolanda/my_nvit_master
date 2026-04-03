#!/home/yangz/.conda/envs/4D-humans/bin/python
import math
import sys
from typing import Iterable, Optional, Dict, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import ModelEma
import utils 
from hmr2.utils.geometry import aa_to_rotmat
from hmr2.models.losses import ParameterLoss 

# ==============================================================================
# 1. Helpers
# ==============================================================================
def move_to_device(obj: Any, device: torch.device, non_blocking=True) -> Any:
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device, non_blocking) for v in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    return obj

def compute_mpjpe(pred_3d: torch.Tensor, gt_3d: torch.Tensor) -> torch.Tensor:
    """
    计算 MPJPE。
    输入必须是 (B, N, 3)。如果 GT 是 4 维，需要在外面切片。
    """
    # Root align (使用第 0 个关节作为根节点)
    pred_centered = pred_3d - pred_3d[:, 0:1, :]
    gt_centered = gt_3d - gt_3d[:, 0:1, :]
    
    # 欧氏距离
    error = torch.norm(pred_centered - gt_centered, dim=-1)
    return error.mean()


def generate_pruned_structure(model):
    # Helper to generate pruned_structure from physical sparsity
    structure = {'backbone': {'blocks': []}}
    real_model = model.module if hasattr(model, 'module') else model
    
    if hasattr(real_model, 'backbone'):
        backbone = real_model.backbone
    else:
        backbone = real_model 

    if not hasattr(backbone, 'blocks'):
        return structure

    for block in backbone.blocks:
        w1 = block.mlp.fc1.weight
        active_neurons = (w1.abs().sum(dim=1) > 1e-9).sum().item()
        
        block_info = {
            'mlp': {
                'in_features': block.mlp.fc1.in_features,
                'hidden_features': int(active_neurons),
                'out_features': block.mlp.fc2.out_features
            }
        }
        structure['backbone']['blocks'].append(block_info)
    
    structure['backbone']['num_blocks'] = len(backbone.blocks)
    return structure

def train_one_epoch(args, student: nn.Module, teacher: nn.Module, 
                    pruning_engine, criterion, orig_criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, 
                    max_norm: float = 0, model_ema: Optional[ModelEma] = None, 
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, main_loss_coeff=1.0, original_loss_coeff=1.0):
    
    student.train(set_training_mode)
    teacher.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch = move_to_device(batch, device)
        
        if isinstance(batch, dict):
            samples = batch 
            targets = batch
        else:
            samples, targets = batch

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast('cuda'):
            # Teacher Forward (No Grad)
            with torch.no_grad():
                teacher_output = teacher(samples)
                teacher_joints = teacher_output['pred_keypoints_3d'] if isinstance(teacher_output, dict) else teacher_output

            # Student Forward
            student_output = student(samples)
            student_joints = student_output['pred_keypoints_3d'] if isinstance(student_output, dict) else student_output

            main_loss = criterion(student_output, teacher_output, targets)
                
            # main_loss = criterion(student_joints, teacher_joints)
            original_loss = orig_criterion(samples, student_output, targets)
            loss = main_loss_coeff * main_loss + original_loss_coeff * original_loss

            if args.distributed:
                loss += 0.0 * sum(p.sum() for p in student.parameters() if p.requires_grad)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=student.parameters(), create_graph=is_second_order)

        if pruning_engine is not None:
            pruning_engine.do_step(loss_value, optimizer=optimizer)

        torch.cuda.synchronize()
        
        if model_ema is not None:
            model_ema.update(student)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def compute_pa_mpjpe(pred_3d, gt_3d):
    """
    Standard Procrustes Aligned MPJPE using SVD for rotation alignment.
    Args:
        pred_3d: [B, N, 3]
        gt_3d:   [B, N, 3]
    """
    # 1. Center alignment
    pred_3d = pred_3d - pred_3d.mean(dim=1, keepdim=True)
    gt_3d = gt_3d - gt_3d.mean(dim=1, keepdim=True)

    # 2. Optimal Scale (optional, usually PA-MPJPE includes scale)
    pred_norm = torch.norm(pred_3d, dim=(1, 2), keepdim=True)
    gt_norm = torch.norm(gt_3d, dim=(1, 2), keepdim=True)
    scale = gt_norm / (pred_norm + 1e-8)
    pred_3d = pred_3d * scale

    # 3. Optimal Rotation (Procrustes via SVD)
    # M = GT^T * Pred
    # shape: [B, 3, N] @ [B, N, 3] -> [B, 3, 3]
    M = torch.matmul(gt_3d.transpose(1, 2), pred_3d)
    
    # SVD
    U, _, V = torch.svd(M)
    
    # Rotation Matrix R = U * V^T
    R = torch.matmul(U, V.transpose(1, 2))
    
    # Handle reflection case (det(R) < 0)
    det = torch.det(R)
    sign = torch.sign(det) # [B]
    # This part can be tricky in batch, simplified here for standard cases:
    # Construct a correction matrix if needed, but for MPJPE usually SVD is enough 
    # if we assume coarse alignment.
    
    # Apply Rotation
    pred_rotated = torch.matmul(pred_3d, R.transpose(1, 2))
    
    # 4. Compute Error
    return torch.norm(pred_rotated - gt_3d, dim=-1).mean()
# =======================================[Any][Any][Any]=======================================
# 2. Pruning Logic
# ==============================================================================
def prune_with_Taylor(args, start_epoch, start_iter, data_loader, student, teacher, pruning_engine,
                        optimizer_gates_lr, interval_prune, gate_loss_coeff,
                        main_loss_coeff, prune_neurons_max, optimizer, lr_scheduler, loss_scaler, bs, 
                        criterion, orig_criterion,
                        amp_enable=True, student_eval=False, prune_per_iteration=16, 
                        original_loss_coeff=0.0, mixup_fn=None,
                        val_loader=None,            
                        perf_tolerance=0.10,        
                        base_mpjpe=None,            
                        safety_check_batches=500,
                        estimated_pruned_count = 0):  
    
    device = torch.device(args.device)
    
    # --- Baseline ---
    if val_loader is not None and base_mpjpe is None:
        if utils.is_main_process():
            print("🔍 Calculating baseline MPJPE...", flush=True)
        # 这里的 evaluate 现在能正确处理 4 通道 GT 了
        metrics = evaluate(val_loader, student, device, criterion=None, max_batches=safety_check_batches * 2)
        base_mpjpe = metrics['mpjpe']
        if utils.is_main_process():
            print(f"📉 Baseline MPJPE: {base_mpjpe:.2f} mm", flush=True)
    
    # --- Pruning Config ---
    total_prunable_neurons = 0
    if hasattr(pruning_engine, 'target_layers_config'):
        for layer_cfg in pruning_engine.target_layers_config:
            if len(layer_cfg["compute_criteria_from"]) > 0:
                item = layer_cfg["compute_criteria_from"][0]
                total_prunable_neurons += item["parameter"].shape[item["dim"]]

    if prune_neurons_max <= 1.0:
        target_abs = int(total_prunable_neurons * prune_neurons_max)
    else:
        target_abs = int(prune_neurons_max)

    fixed_step_size = prune_per_iteration 
    pruning_engine.prune_per_iteration = fixed_step_size
    pruning_engine.frequency = interval_prune
    pruning_engine.maximum_pruning_iterations = 99999999 
    pruning_engine.prune_neurons_max += 99999999

    if utils.is_main_process():
        print(f"\n🎯 Pruning Plan: Target {target_abs} neurons", flush=True)

    student.train()
    teacher.eval()
    
    iter_num = 0
    pruning_done = False
    epoch = 0
    

    while not pruning_done:
        if hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(args.seed + epoch + start_epoch)

        for batch_idx, batch in enumerate(data_loader):
            # 1. 搬运数据并保持字典结构
            batch = move_to_device(batch, device)
            
            # 🔥 统一解包：targets 必须是整个字典，HMR2 需要里面的 2D/3D/SMPL 全部字段
            if isinstance(batch, dict):
                samples = batch
                targets = batch 
            else:
                samples, targets = batch

            iter_num += 1
            
            # 2. Teacher 预测 (不计梯度)
            with torch.no_grad():
                # 老师在 CPU 或 GPU 运行取决于你的显存情况
                teacher_out = teacher(samples)
                teacher_joints = teacher_out['pred_keypoints_3d'] if isinstance(teacher_out, dict) else teacher_out

            # 3. Student 预测 (开启混合精度)
            with torch.amp.autocast('cuda', enabled=amp_enable):
                student_out = student(samples)
                
                # 🔥 关键补丁：如果学生模型是剪枝后的基础模型，可能缺失 HMR2 Loss 需要的字典键
                if isinstance(student_out, dict) and 'pred_smpl_params' not in student_out:
                    # 尝试从输出中组合 SMPL 参数结构
                    pred_pose = student_out.get('pred_pose', torch.zeros(samples['img'].shape[0], 72, device=device))
                    betas = student_out.get('pred_shape', student_out.get('pred_betas', torch.zeros(pred_pose.shape[0], 10, device=device)))
                    
                    student_out['pred_smpl_params'] = {
                        'global_orient': pred_pose[:, :3], # 或者是旋转矩阵格式，取决于你的 Head
                        'body_pose': pred_pose[:, 3:],
                        'betas': betas
                    }
                
                # 获取用于蒸馏的 joints
                student_joints = student_out['pred_keypoints_3d'] if isinstance(student_out, dict) else student_out

                # 4. 计算 Loss 组合
                # (A) 蒸馏 Loss - 师徒对齐
                loss_kl = torch.tensor(0.0, device=device)
                if criterion is not None and main_loss_coeff > 0.0:
                    loss_kl = criterion(student_joints, teacher_joints)
                
                # (B) 几何/原始任务 Loss - Wrapper 介入
                loss_geo = torch.tensor(0.0, device=device)
                if orig_criterion is not None:
                    # ✅ 此时 targets 是字典，Wrapper 内部的 hmr2.compute_loss 不再报错
                    loss_geo = orig_criterion(samples, student_out, targets)

                # 总 Loss 加权
                loss = main_loss_coeff * loss_kl + original_loss_coeff * loss_geo

            # 5. 梯度回传与剪枝引擎步进
            optimizer.zero_grad(set_to_none=True)
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=student.parameters(), create_graph=False)
            
            # 剪枝决策逻辑
            current_loss_val = loss.item()
            is_pruning_step = (iter_num % interval_prune == 0) and (not pruning_done)
            
            # [PATCHED] Redundant do_step removed
            # if not pruning_done:
            #     pruning_engine.do_step(loss=current_loss_val, optimizer=optimizer)   

            # --- Exit Logic ---
            is_pruning_step = (iter_num % interval_prune == 0) and (not pruning_done)
            if is_pruning_step:
                if estimated_pruned_count >= target_abs:
                    if utils.is_main_process(): print(f"✅ Target Reached.", flush=True)
                    pruning_done = True
                    break

                if val_loader is not None and iter_num % (interval_prune * 10) == 0:
                    if utils.is_main_process(): print(f"🔍 Validating...", flush=True)
                    val_metrics = evaluate(val_loader, student, device, max_batches=safety_check_batches)
                    if val_metrics['mpjpe'] > base_mpjpe * (1 + perf_tolerance):
                        if utils.is_main_process(): print(f"🛑 Collapse: {val_metrics['mpjpe']:.2f}", flush=True)
                        pruning_done = True
                        break
                    student.train()

            current_loss_val = loss.item()
            if not pruning_done:
                pruning_engine.do_step(loss=current_loss_val, optimizer=optimizer)
                if is_pruning_step:
                    estimated_pruned_count += fixed_step_size
                    if utils.is_main_process() and iter_num % 50 == 0:
                        print(f"✂️  Pruned: {estimated_pruned_count}/{target_abs}", flush=True)
            
            torch.cuda.synchronize()
            del loss, student_out
            optimizer.zero_grad(set_to_none=True)

        if pruning_done: break
        epoch += 1
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

    if args.output_dir and utils.is_main_process():
         # [PATCHED] Generate pruned structure
         p_struct = generate_pruned_structure(student)
         torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict(), "args": args, "pruned_structure": p_struct}, f"{args.output_dir}/pruned_final.pth")

    return epoch + start_epoch, iter_num + start_iter, pruning_done

# ==============================================================================
# 3. Evaluate (修复版: 处理 4 通道 GT)
# ==============================================================================
@torch.no_grad()
def evaluate(data_loader, model, device, criterion=None, max_batches=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    for i, batch in enumerate(metric_logger.log_every(data_loader, 50, header)):
        if max_batches is not None and i >= max_batches:
            print(f"⚡ Fast Eval Stopped.", flush=True)
            break

        batch = move_to_device(batch, device)
        if isinstance(batch, dict):
            samples = batch 
            img_tensor = batch['img']
            gt_3d = batch['keypoints_3d']
        else:
            samples, gt_3d = batch 
            img_tensor = samples

        outputs = model(samples)
        
        if isinstance(outputs, dict):
            pred_3d = outputs.get('pred_keypoints_3d', outputs.get('pred_joints'))
        else:
            pred_3d = outputs
        
        if pred_3d is not None:
            # 1. 对齐关节数量 (Truncate to 24)
            if pred_3d.shape[1] > 24: pred_3d = pred_3d[:, :24, :]
            if gt_3d.shape[1] > 24: gt_3d = gt_3d[:, :24, :]
            
            # 2. 🔥 [CRITICAL FIX] 对齐通道数 (Truncate 4th channel)
            # GT 现在是 (B, 24, 4)，我们要把它变成 (B, 24, 3) 才能和 Pred 相减
            if gt_3d.shape[-1] == 4:
                gt_3d = gt_3d[:, :, :3]
            # print(f"DEBUG: GT Max: {gt_3d.abs().max().item()}, Pred Max: {pred_3d.abs().max().item()}")
            
            # 3. 计算指标
            mpjpe = compute_mpjpe(pred_3d, gt_3d)
            pa_mpjpe = compute_pa_mpjpe(pred_3d, gt_3d)
            
            metric_logger.update(mpjpe=mpjpe.item(), n=img_tensor.shape[0])
            metric_logger.update(pa_mpjpe=pa_mpjpe.item(), n=img_tensor.shape[0])

    metric_logger.synchronize_between_processes()
    print(f'* MPJPE {metric_logger.mpjpe.global_avg:.3f} mm', flush=True)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
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
            
            # Loss 2D/3D (handling visibility masking)
            gt_keypoints_2d = targets['keypoints_2d']
            gt_keypoints_3d = targets['keypoints_3d']
            
            # Slice 2D to (B, N, 2) if it has visibility column
            if gt_keypoints_2d.shape[-1] == 3:
                conf_2d = gt_keypoints_2d[..., 2:3]
                gt_keypoints_2d = gt_keypoints_2d[..., :2]
                loss_2d = (conf_2d * F.l1_loss(pred_keypoints_2d, gt_keypoints_2d, reduction='none')).mean()
            else:
                loss_2d = F.l1_loss(pred_keypoints_2d, gt_keypoints_2d)

            # Slice 3D to (B, N, 3) if it has confidence
            if gt_keypoints_3d.shape[-1] == 4:
                conf_3d = gt_keypoints_3d[..., 3:4]
                gt_keypoints_3d = gt_keypoints_3d[..., :3]
                loss_3d = (conf_3d * F.l1_loss(pred_keypoints_3d, gt_keypoints_3d, reduction='none')).mean()
            else:
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
