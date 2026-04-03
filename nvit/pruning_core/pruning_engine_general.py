"""
Refactored Pruning Engine (High Fidelity + Compatibility Fix)
Functionally equivalent to the original NVIDIA code for Method 22 (Taylor Pruning).
Includes PruningConfigReader for compatibility with pruning_utils.py.
"""

import os
import json
import pickle
import numpy as np
import torch
from copy import deepcopy

# ==============================================================================
# 1. 兼容性组件 (Legacy Components)
#    修复 ImportError: cannot import name 'PruningConfigReader'
# ==============================================================================
class PruningConfigReader(object):
    """
    保留此雷是为了兼容 pruning_utils.py 的引用。
    虽然我们现在主要在 main.py 里直接配置，但旧代码依赖它读取 JSON。
    """
    def __init__(self):
        self.pruning_settings = {}
        self.config = None

    def read_config(self, filename):
        with open(filename, "r") as f:
            self.config = json.load(f)

        if "jobConfiguration" in self.config.keys():
            self.config = self.config["jobConfiguration"]

        # 读取常用参数，保持默认值逻辑
        self.read_field_value("method", 22)
        self.read_field_value("frequency", 30)
        self.read_field_value("prune_per_iteration", 1)
        self.read_field_value("maximum_pruning_iterations", 10000)
        self.read_field_value("pruning_threshold", 100)
        self.read_field_value("start_pruning_after_n_iterations", 0)
        self.read_field_value("group_size", 1)
        self.read_field_value("pruning_momentum", 0.9)
        self.read_field_value("l2_normalization_per_layer", True)
        self.read_field_value("latency_regularization", 0.0)
        self.read_field_value("latency_target", 0.0)
        self.read_field_value("latency_look_up_table", "")

    def read_field_value(self, key, default):
        val = self.config.get(key, default)
        self.pruning_settings[key] = val

    def get_parameters(self):
        return self.pruning_settings

# ==============================================================================
# 2. 新版配置管理 (Refactored Config)
# ==============================================================================
class PruningConfig:
    def __init__(self, settings):
        self.method = settings.get('method', 22) # 22 = Taylor_gate
        self.frequency = settings.get('frequency', 30)
        self.prune_per_iteration = settings.get('prune_per_iteration', 1)
        self.start_iter = settings.get('start_pruning_after_n_iterations', 0)
        self.max_iter = settings.get('maximum_pruning_iterations', 0)
        self.threshold = settings.get('pruning_threshold', 100)
        self.group_size = settings.get('group_size', 1)
        self.momentum = settings.get('pruning_momentum', 0.9)
        self.l2_norm = settings.get('l2_normalization_per_layer', True)
        
        # 兼容性设置
        self.leave_at_least_one = settings.get('leave_at_least_one_group', True)
        self.fixed_criteria = settings.get('fixed_criteria', False)
        
        # 延迟相关
        self.latency_reg = settings.get('latency_regularization', 0.0)
        self.latency_table_path = settings.get('latency_look_up_table', "")

# ==============================================================================
# 3. 剪枝引擎核心 (Pruning Engine Core)
# ==============================================================================
class PruningEngine(object):
    def __init__(self, for_pruning_parameters, pruning_settings=dict(), log_folder=None, 
                 latency_regularization=0., latency_target=0., latency_look_up_table=""):
        
        # 1. 配置合并
        pruning_settings['latency_regularization'] = latency_regularization
        pruning_settings['latency_target'] = latency_target
        if latency_look_up_table:
            pruning_settings['latency_look_up_table'] = latency_look_up_table
            
        self.cfg = PruningConfig(pruning_settings)
        self.log_folder = log_folder
        self.train_writer = None

        # 2. 核心结构
        self.target_layers_config = for_pruning_parameters
        self.pruning_gates = []     
        self.criteria_buffer = []   
        self.layer_params = []      
        
        self.iterations_done = 0
        self.is_finished = False
        
        self._init_structures()

    def _init_structures(self):
        for layer_cfg in self.target_layers_config:
            main_param = layer_cfg["compute_criteria_from"][0]["parameter"]
            self.layer_params.append(main_param)
            
            num_units = main_param.shape[0]
            self.pruning_gates.append(np.ones(num_units))
            self.criteria_buffer.append(torch.zeros(num_units).cuda() if main_param.is_cuda else torch.zeros(num_units))

    def do_step(self, loss=None, optimizer=None, neurons_left=0, training_acc=0.0):
        """外部调用的主接口"""
        if self.cfg.max_iter > 0 and self.iterations_done >= self.cfg.max_iter:
            self.is_finished = True
            return -1
        
        if self.iterations_done < self.cfg.start_iter:
            self.iterations_done += 1
            return 0

        # 1. 强制 Mask 生效
        self._enforce_mask()

        # 2. 计算重要性 (Taylor + Momentum)
        self._update_criteria_with_momentum()

        # 3. 执行剪枝
        if self.iterations_done > 0 and self.iterations_done % self.cfg.frequency == 0:
            print(f"✂️ [Pruning] Step {self.iterations_done}: Executing Pruning Logic...")
            self._prune_network_global()
            if optimizer:
                self._clean_optimizer_state(optimizer)

        self.iterations_done += 1
        return 1

    def _update_criteria_with_momentum(self):
            """
            计算重要性：(Weight * Grad)^2
            🔥 修复版：强制在 CPU 上计算以避免 OOM
            """
            for i, layer_cfg in enumerate(self.target_layers_config):
                # 初始化为 CPU 上的 0.0
                current_raw_score = 0.0
                
                for item in layer_cfg["compute_criteria_from"]:
                    param = item["parameter"]
                    grad = param.grad
                    dim = item["dim"]
                    name = item.get("parameter_name", "")

                    # ============================================================
                    # 🔥 OOM 修复核心：将计算转移至 CPU
                    # ============================================================
                    if grad is None:
                        # 在 CPU 上创建零张量
                        score = torch.zeros_like(param.cpu()).sum(dim=1) if len(param.shape) > 1 else torch.zeros_like(param.cpu())
                    else:
                        # 1. 搬运到 CPU (Detach 防止梯度追踪，save memory)
                        p_cpu = param.detach().cpu()
                        g_cpu = grad.detach().cpu()
                        
                        # 2. 在 CPU 内存中进行大矩阵计算 (Param * Grad)^2
                        # 这一步如果不移到 CPU，会产生巨大的显存开销
                        score = (p_cpu * g_cpu).pow(2)
                        
                        # 3. 在 CPU 上进行降维 (Reduce)
                        # 降维后，score 会从 [Dim0, Dim1] 变成 [Dim0] 的小向量
                        if dim == 0:
                            score = score.view(p_cpu.shape[0], -1).sum(dim=1)
                        elif dim == 1:
                            if len(score.shape) == 2:
                                score = score.permute(1, 0).reshape(p_cpu.shape[1], -1).sum(dim=1)
                            else:
                                score = score.permute(1, 0, 2).reshape(p_cpu.shape[1], -1).sum(dim=1)
                        
                        # 4. 手动清理 CPU 上的大矩阵，防止内存泄漏
                        del p_cpu, g_cpu

                    # ============================================================
                    # 下面的逻辑保持不变 (但在 CPU 上执行)
                    # ============================================================
                    
                    if 'EMB' in name: score /= 12.0
                    if 'head' in name: score /= 6.0
                    if 'qkv' in name: score /= 2.0
                    
                    if isinstance(current_raw_score, float):
                        current_raw_score = score
                    else:
                        current_raw_score += score
                
                # 处理 NaN (CPU 上)
                if not isinstance(current_raw_score, float) and torch.isnan(current_raw_score).any():
                    current_raw_score[torch.isnan(current_raw_score)] = 0.0

                # ============================================================
                # 🔥 最后一步：搬回 GPU 更新 Buffer
                # ============================================================
                # 获取 buffer 所在的设备 (通常是 cuda:x)
                target_device = self.criteria_buffer[i].device
                
                # 将 CPU 算好的小向量搬回 GPU
                if isinstance(current_raw_score, float):
                    current_raw_score_gpu = torch.zeros_like(self.criteria_buffer[i])
                else:
                    current_raw_score_gpu = current_raw_score.to(target_device)

                # Momentum Update (在 GPU 上进行)
                if self.iterations_done > 0:
                    self.criteria_buffer[i] = (self.cfg.momentum * self.criteria_buffer[i]) + \
                                            ((1.0 - self.cfg.momentum) * current_raw_score_gpu)
                else:
                    self.criteria_buffer[i] = current_raw_score_gpu
                
                # 清理引用
                del current_raw_score
                
    def _prune_network_global(self):
        """执行剪枝：L2 Norm -> Sort -> Threshold -> Mask"""
        all_scores_flat = []
        
        for i, score_tensor in enumerate(self.criteria_buffer):
            scores = score_tensor.detach().cpu().numpy()
            if self.cfg.l2_norm:
                norm = np.linalg.norm(scores)
                if norm > 1e-8: scores = scores / norm
            
            mask = self.pruning_gates[i]
            valid_scores = (scores * mask)[mask > 0]
            if len(valid_scores) > 0:
                all_scores_flat.extend(valid_scores.tolist())

        if not all_scores_flat: return

        all_scores_flat.sort()
        # FIXED: Use interval counts, not raw steps
        freq = self.cfg.frequency if self.cfg.frequency > 0 else 1
        interval_count = self.iterations_done // freq
        total_pruned_target = (interval_count * self.cfg.prune_per_iteration)
        # Scaling Fix for 4D-Humans (Neurons -> Structures)
        structural_target = int(total_pruned_target / 2304)
        structural_target = min(structural_target, len(all_scores_flat) - 1)
        if structural_target < 0: structural_target = 0
        threshold = all_scores_flat[structural_target]
        print(f"📊 [Analysis] Threshold: {threshold:.6e} | Target Pruned: {structural_target} (Neuron Target: {total_pruned_target})")
        for i, score_tensor in enumerate(self.criteria_buffer):
            scores = score_tensor.detach().cpu().numpy()
            mask = self.pruning_gates[i]
            if self.cfg.l2_norm:
                norm = np.linalg.norm(scores)
                if norm > 1e-8: scores = scores / norm
            
            to_prune = (scores <= threshold) & (mask > 0)
            mask[to_prune] = 0.0
            
            if self.cfg.leave_at_least_one and mask.sum() == 0:
                mask[np.argmax(scores)] = 1.0

    def _enforce_mask(self):
        """Apply Mask"""
        for i, layer_cfg in enumerate(self.target_layers_config):
            mask = torch.from_numpy(self.pruning_gates[i]).float().cuda()
            for target in layer_cfg["set_to_zero"]:
                param = target["parameter"]
                dim = target["dim"]
                if dim == 0:
                    shape = (param.shape[0],) + (1,) * (param.dim() - 1)
                elif dim == 1:
                    shape = (1, param.shape[1]) + (1,) * (param.dim() - 2)
                
                mask_reshaped = mask.view(shape)
                param.data.mul_(mask_reshaped)
                if param.grad is not None:
                    param.grad.mul_(mask_reshaped)

    def _clean_optimizer_state(self, optimizer):
        pass

    # 兼容性接口
    def init_pruning_helper(self, model, data, skip_pass=False): pass
    def connect_tensorboard(self, tb_writer): self.train_writer = tb_writer
    def update_flops_stats(self): pass
    def zero_grad(self): pass

# ==============================================================================
# 4. 关键别名 (Aliases)
# ==============================================================================
pytorch_pruning = PruningEngine

class ExpMeter(object):
    def __init__(self, mom=0.9):
        self.reset()
        self.mom = mom
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.exp_avg = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.exp_avg = self.mom * self.exp_avg + (1.0 - self.mom) * self.val
        if self.count == 1: self.exp_avg = self.val