#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import torch.nn as nn

class GeometricConsistencyLoss(nn.Module):
    def __init__(self, 
                 mpjpe_coeff=1.0,
                 bone_coeff=0.5,
                 pa_mpjpe_coeff=1.0,      
                 velocity_coeff=0.1,      
                 twist_coeff=0.5,
                 symmetry_coeff=0.05,      
                 contact_coeff=0.1):       
        super().__init__()
        
        self.mpjpe_coeff = mpjpe_coeff
        self.bone_coeff = bone_coeff
        self.pa_mpjpe_coeff = pa_mpjpe_coeff
        self.velocity_coeff = velocity_coeff
        self.twist_coeff = twist_coeff
        self.symmetry_coeff = symmetry_coeff
        self.contact_coeff = contact_coeff
        
        # 1. SMPL 骨骼连接
        indices = [
            [0, 1], [0, 2], [1, 4], [2, 5], [4, 7], [5, 8],  # 下肢
            [0, 3], [3, 6], [6, 9], [9, 12], [12, 15],       # 躯干
            [9, 13], [9, 14], [13, 16], [14, 17],            # 肩部
            [16, 18], [17, 19], [18, 20], [19, 21],          # 上肢
            [20, 22], [21, 23]                               # 手部
        ]
        self.register_buffer('bone_indices', torch.tensor(indices).long())
        
        # 2. 对称关节对
        symmetric_pairs = [
            (1, 2), (4, 5), (7, 8),       # 腿
            (13, 14), (16, 17), (18, 19), # 臂
            (20, 21), (22, 23)            # 手
        ]
        self.register_buffer('symmetric_pairs', torch.tensor(symmetric_pairs).long())

        # 3. 角度三元组
        angle_triplets = [
            [1, 4, 7],   # 左膝
            [2, 5, 8],   # 右膝
            [16, 18, 20],# 左肘
            [17, 19, 21],# 右肘
            [0, 3, 6],   # 脊柱
            [9, 12, 15]  # 颈部
        ]
        self.register_buffer('angle_triplets', torch.tensor(angle_triplets).long())

    def compute_bone_loss(self, pred, gt):
        parents = self.bone_indices[:, 0]
        children = self.bone_indices[:, 1]
        n_joints = pred.shape[1]
        valid_mask = (parents < n_joints) & (children < n_joints)
        parents = parents[valid_mask]
        children = children[valid_mask]

        pred_bones = pred[:, parents] - pred[:, children]
        gt_bones = gt[:, parents] - gt[:, children]
        return torch.mean(torch.abs(torch.norm(pred_bones, dim=-1) - torch.norm(gt_bones, dim=-1)))
    
    def compute_twist_loss(self, pred, gt):
        p_idx = self.angle_triplets[:, 0]
        c_idx = self.angle_triplets[:, 1]
        ch_idx = self.angle_triplets[:, 2]
        
        n_joints = pred.shape[1]
        valid_mask = (p_idx < n_joints) & (c_idx < n_joints) & (ch_idx < n_joints)
        p_idx, c_idx, ch_idx = p_idx[valid_mask], c_idx[valid_mask], ch_idx[valid_mask]

        def get_normals(joints):
            u = torch.nn.functional.normalize(joints[:, p_idx] - joints[:, c_idx], dim=-1)
            v = torch.nn.functional.normalize(joints[:, ch_idx] - joints[:, c_idx], dim=-1)
            normal = torch.cross(u, v, dim=-1)
            return torch.nn.functional.normalize(normal, dim=-1)

        loss = 1.0 - (get_normals(pred) * get_normals(gt)).sum(dim=-1)
        return torch.nan_to_num(loss, nan=0.0).mean()

    def compute_pa_mpjpe(self, pred, gt):
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        gt_centered = gt - gt.mean(dim=1, keepdim=True)
        scale = torch.norm(gt_centered, dim=(1, 2), keepdim=True) / (torch.norm(pred_centered, dim=(1, 2), keepdim=True) + 1e-8)
        return torch.norm(pred_centered * scale - gt_centered, dim=-1).mean()

    def compute_symmetry_loss(self, pred):
        n_joints = pred.shape[1]
        total_loss = 0.0
        count = 0
        for l, r in self.symmetric_pairs:
            if l < n_joints and r < n_joints:
                left, right = pred[:, l], pred[:, r].clone()
                right[:, 0] = -right[:, 0]
                total_loss += torch.norm(left - right, dim=-1).mean()
                count += 1
        return total_loss / count if count > 0 else torch.tensor(0.0, device=pred.device)

    def forward(self, outputs, targets):
        pred = outputs.get('pred_keypoints_3d', outputs.get('pred_joints')) if isinstance(outputs, dict) else outputs[0]
        gt = targets.get('keypoints_3d', targets.get('joints_3d')) if isinstance(targets, dict) else targets
        
        pred, gt = pred.float(), gt.float()
        
        # Handle dimension mismatch: gt may have 4 dims (x,y,z,conf), pred has 3 (x,y,z)
        if gt.shape[-1] == 4:
            gt = gt[..., :3]  # Only take x,y,z, discard confidence
        if pred.shape[-1] == 4:
            pred = pred[..., :3]
            
        n_joints = min(pred.shape[1], gt.shape[1])
        pred, gt = pred[:, :n_joints, :], gt[:, :n_joints, :]
        
        # Root-Relative
        pred_rel = pred - pred[:, 0:1, :]
        gt_rel = gt - gt[:, 0:1, :]

        loss = (self.mpjpe_coeff * torch.norm(pred_rel - gt_rel, dim=-1).mean() +
                self.bone_coeff * self.compute_bone_loss(pred_rel, gt_rel) +
                self.twist_coeff * self.compute_twist_loss(pred_rel, gt_rel) +
                self.pa_mpjpe_coeff * self.compute_pa_mpjpe(pred_rel, gt_rel) +
                self.symmetry_coeff * self.compute_symmetry_loss(pred_rel))
        
        return loss

# ... (上面是之前的 GeometricConsistencyLoss 和 HMR2LossWrapper) ...

class HMR2LossWrapper(nn.Module):
    def __init__(self, hmr2_model_instance, geometric_loss_fn):
        super().__init__()
        self.hmr2_model = hmr2_model_instance
        self.geometric_loss_fn = geometric_loss_fn

    # 🔑 关键修复：确保参数列表能对应 (self, samples, outputs, targets)
    def forward(self, samples, outputs, targets):
        """
        对应 engine_4d.py 中的调用：orig_criterion(samples, student_output, targets)
        """
        # 1. 自动处理 DDP 包装
        real_model = self.hmr2_model.module if hasattr(self.hmr2_model, 'module') else self.hmr2_model
        
        # 2. 计算 HMR2 核心损失 (2D/3D 关键点, SMPL 参数等)
        # 注意：HMR2 的 compute_loss 签名通常是 (targets, outputs, train=True)
        loss_hmr2_dict = real_model.compute_loss(targets, outputs, train=True)
        
        # 如果返回的是字典，将其求和
        if isinstance(loss_hmr2_dict, dict):
            loss_hmr2 = sum(loss_hmr2_dict.values())
        else:
            loss_hmr2 = loss_hmr2_dict
        
        # 3. 计算额外的几何一致性损失
        # 你的 GeometricConsistencyLoss 定义也是接收三个参数
        loss_geo = self.geometric_loss_fn(outputs, targets)
        
        return loss_hmr2 + loss_geo

class HMR2DistillationLoss(torch.nn.Module):
    def __init__(self, base_criterion, teacher_model, alpha=0.5, temperature=3.0):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature

    # 🔑 确保这里的参数个数是 self + 3 个参数 = 4 个
    def forward(self, student_output, teacher_output, targets):
        """
        对应 engine_4d.py 中的调用：criterion(student_output, teacher_output, targets)
        """
        # 1. 提取关键点 (处理可能是 dict 或 tensor 的情况)
        s_joints = student_output['pred_keypoints_3d'] if isinstance(student_output, dict) else student_output
        t_joints = teacher_output['pred_keypoints_3d'] if isinstance(teacher_output, dict) else teacher_output

        # 2. 只取坐标部分 [B, 24, 3]
        s_j = s_joints[:, :24, :3]
        t_j = t_joints[:, :24, :3]

        # 3. Root-Relative 对齐 (消除全局位移误差)
        s_rel = s_j - s_j[:, 0:1, :]
        t_rel = t_j - t_j[:, 0:1, :]
        
        # 4. 计算蒸馏损失 (MSE)
        distill_loss = torch.nn.functional.mse_loss(s_rel, t_rel)
        
        return distill_loss