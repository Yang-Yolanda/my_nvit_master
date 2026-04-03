#!/home/yangz/.conda/envs/4D-humans/bin/python
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
import sys
from pathlib import Path

# Add 4D-Humans to path to import hmr2
sys.path.insert(0, '/home/yangz/4D-Humans')
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.distributed as dist
import numpy as np

import json
import torch
import numpy as np
from hmr2.models import load_hmr2  # 假设这是你的模型类名
# 如果你的 load_hmr2 是个函数，请相应调整，这里假设我们操作 HMR2 类或其配置

def load_pruned_hmr2(json_config_path, checkpoint_path, base_model_type='vit_huge'):
    """
    根据剪枝配置文件(JSON)加载 HMR2 模型
    
    Args:
        json_config_path: 你的 pruning_config.json 路径
        checkpoint_path: 剪枝后的 .pth 权重路径
        base_model_type: 基座模型类型 ('vit_huge' or 'vit_base')
    """
    print(f"🔧 Loading Pruned Model Config from: {json_config_path}")
    
    with open(json_config_path, 'r') as f:
        config_dict = json.load(f)

    # ==========================================
    # 1. 定义基座参数 (Base Params)
    # ==========================================
    if base_model_type == 'vit_huge':
        BASE_DIM = 1280
        BASE_HEAD_DIM = 80  # 1280 / 16
        BASE_MLP = 5120
    elif base_model_type == 'vit_base':
        BASE_DIM = 768
        BASE_HEAD_DIM = 64
        BASE_MLP = 3072
    else:
        raise ValueError("Unknown base model type")

    # ==========================================
    # 2. 解析 JSON -> 每一层的整数维度列表
    # ==========================================
    # 自动探测层数
    layer_indices = [int(k.split('.')[2]) for k in config_dict.keys() if 'backbone.blocks.' in k and 'attn.qkv' in k]
    num_layers = max(layer_indices) + 1 if layer_indices else 32
    
    embed_dims = []  # 每一层的 embedding dimension
    num_heads = []   # 每一层的 head 数量
    mlp_ratios = []  # 每一层的 mlp ratio
    
    print(f"🧐 Detected {num_layers} layers. Parsing dimensions...")

    for i in range(num_layers):
        # --- 解析 Attention ---
        qkv_key = f"backbone.blocks.{i}.attn.qkv"
        # 默认 1.0 如果 key 不存在
        ratio_qkv = config_dict.get(qkv_key, 1.0) 
        
        # 计算当前层的 dim
        curr_dim = int(BASE_DIM * ratio_qkv)
        
        # 计算 Head 数量 (保持 Head Dim 不变)
        curr_num_head = max(1, int(round(curr_dim / BASE_HEAD_DIM)))
        
        embed_dims.append(curr_dim)
        num_heads.append(curr_num_head)
        
        # --- 解析 MLP ---
        mlp_key = f"backbone.blocks.{i}.mlp.fc1"
        ratio_mlp = config_dict.get(mlp_key, 1.0)
        
        # HMR2/ViT 通常接受 mlp_ratio (float) 或 mlp_dims (list)
        # 这里我们计算出 mlp_ratio。注意：如果 backbone 变窄了，ratio 需要相对于当前 dim
        curr_mlp_dim = int(BASE_MLP * ratio_mlp)
        curr_ratio = curr_mlp_dim / curr_dim if curr_dim > 0 else 4.0
        mlp_ratios.append(curr_ratio)

    # 最终的 Embed Dim 通常取最后一层的输出或者第一层 (取决于你的剪枝策略是否允许维度变化)
    # ⚠️ HMR2 的 Backbone 如果是变宽度的，需要你的 ViT 代码支持 per-layer dim
    # 这里假设我们取整个 backbone 的平均或最大值作为 container dim，或者传入 list
    final_embed_dim = embed_dims[-1] 

    # ==========================================
    # 3. 实例化模型 (修改配置)
    # ==========================================
    # 这里你需要根据你的 HMR2 构造函数调整
    # 假设你的 model_cfg 可以接受 list 作为参数
    
    model_cfg = {
        'img_size': 256,
        'patch_size': 16,
        'embed_dim': embed_dims,   # 🔥 传入列表！需要你的 ViT 支持
        'depth': num_layers,
        'num_heads': num_heads,    # 🔥 传入列表
        'mlp_ratio': mlp_ratios,   # 🔥 传入列表
        # ... 其他 HMR2 参数
    }
    
    print("🏗️ Instantiating HMR2 model with pruned architecture...")
    # 注意：这里可能需要修改 hmr2.models.HMR2 的 __init__ 来接受这些列表
    # 如果你的库还没改，你需要去改一下 ViT 的定义
    student_model = HMR2(**model_cfg) 
    
    # ==========================================
    # 4. 加载权重 (处理 Head 维度不匹配)
    # ==========================================
    print(f"📥 Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint['state_dict']

    # 🔥 自动修剪 Head 权重逻辑
    model_dict = student_model.state_dict()
    
    # 这里的关键是：Backbone 输出维度变了 (例如 1280 -> 800)
    # 但 Checkpoint 里的 Head 权重可能还是旧的 (如果之前保存没处理好)
    # 或者新初始化的 Head 权重维度是对的 (800)，但 state_dict 里是错的
    
    for name, param in state_dict.items():
        if name not in model_dict:
            continue
            
        # 检查维度是否匹配
        if param.shape != model_dict[name].shape:
            # 这是一个典型的 Head 维度不匹配
            if 'head' in name or 'smpl_head' in name:
                print(f"⚠️ Resizing {name}: {param.shape} -> {model_dict[name].shape}")
                
                # 假设是 Linear 层 (Out, In)
                if len(param.shape) == 2 and len(model_dict[name].shape) == 2:
                    new_in_dim = model_dict[name].shape[1]
                    # 简单的切片加载 (取前 N 个通道)
                    # 最完美的做法是用 mask，但在推理加载阶段，切片通常够用了
                    new_param = param[:, :new_in_dim]
                    state_dict[name] = new_param
                    
    # 加载处理后的权重
    student_model.load_state_dict(state_dict, strict=False)
    print("✅ Pruned HMR2 model loaded successfully!")
    
    return student_model

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def print_model_parameters(model, with_values=False):
    print(f"{'Param name':40} {'Shape':30} {'Type':15}")
    print('-'*90)
    for name, param in model.named_parameters():
        if 'weight' in name or 'mask' in name:
            print(f'{name:40} {str(param.shape):30} {str(param.dtype):15}')
            if with_values:
                print(param)
                
def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'weight' in name or 'mask' in name:
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
            #print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
        if 'weight' in name or 'mask' in name:
            tensor = np.abs(tensor)
            if len(tensor.shape)==4:
                dim0 = np.sum(np.sum(tensor, axis=0),axis=(1,2))
                dim1 = np.sum(np.sum(tensor, axis=1),axis=(1,2))
            if len(tensor.shape)==2:
                dim0 = np.sum(tensor, axis=0)
                dim1 = np.sum(tensor, axis=1)
            nz_count0 = np.count_nonzero(dim0)
            nz_count1 = np.count_nonzero(dim1)
            print(f'{name:20} | dim0 = {nz_count0:7} / {len(dim0):7} ({100 * nz_count0 / len(dim0):6.2f}%) | dim1 = {nz_count1:7} / {len(dim1):7} ({100 * nz_count1 / len(dim1):6.2f}%)')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# ==========================================
# 打开 utils.py，将这段代码粘贴到最后
# ==========================================
import torch
from torch.cuda.amp import GradScaler as TorchGradScaler

class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = TorchGradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=False)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # 必须先反量化才能裁剪梯度
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)