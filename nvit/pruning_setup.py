#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
from pruning_core.pruning_engine_general import pytorch_pruning

def create_pruning_structure_vit(
    model, 
    prune_qk=True, 
    prune_v=True, 
    prune_MLP=True, 
    prune_head=False,        # 控制分类头
    prune_emb=False,         # 控制 Embedding 层
    prune_discriminator=False # 控制判别器/GAN组件
):
    """
    高度可控的 ViT 剪枝打标函数
    通过外部传参决定哪些组件的 Linear 层进入剪枝池
    """
    prunable_count = 0
    # 将名称转为小写以便统一匹配
    
    for name, module in model.named_modules():
        # 剪枝主要针对线性层
        if not isinstance(module, nn.Linear): 
            continue
            
        should_prune = False
        name_lower = name.lower()
        
        # --- 核心逻辑分支 ---

        # 1. 基础 Backbone 控制 (Attention)
        if 'attn' in name_lower:
            if 'qkv' in name_lower:
                if prune_qk or prune_v: should_prune = True
            elif 'proj' in name_lower:
                should_prune = True # Attention 输出投影通常随 QKV 剪枝

        # 2. MLP 层控制
        elif 'mlp' in name_lower and ('fc1' in name_lower or 'fc2' in name_lower):
            if prune_MLP: should_prune = True

        # 3. Embedding 层控制 (通常是 patch_embed 或 pos_embed 后的变换)
        elif 'embed' in name_lower:
            if prune_emb: should_prune = True
            else: should_prune = False # 显式保护

        # 4. Head / Decoder / SMPL 控制 (分类或回归头)
        elif any(key in name_lower for key in ['head', 'decoder', 'smpl']):
            if prune_head: should_prune = True
            else: should_prune = False # 显式保护

        # 5. Discriminator 判别器控制
        elif 'disc' in name_lower:
            if prune_discriminator: should_prune = True
            else: should_prune = False # 显式保护

        # --- 最终执行打标 ---
        if should_prune:
            # 标记该模块为可剪枝，供后续扫描引擎使用
            module.is_prunable = True
            module.pruning_dim = 0
            prunable_count += 1
            # print(f"DEBUG: 选中剪枝层 -> {name}")
            
    print(f"✅ [Structure] Marked {prunable_count} layers for pruning (Head:{prune_head}, Emb:{prune_emb}, Disc:{prune_discriminator})")
    return prunable_count

def scan_pruning_targets(model, target_prefix="backbone."):
    """自动扫描 Linear 层并生成配置"""
    formatted_params = []
    count = 0
    print(f"🛠️  [Config] Auto-detecting targets in '{target_prefix}'...")

    for name, module in model.named_modules():
        if name.startswith(target_prefix) and isinstance(module, nn.Linear):
            layer_config = {
                "compute_criteria_from": [{"parameter": module.weight, "parameter_name": f"{name}.weight", "dim": 0, "fix": False, "type": "linear"}],
                "set_to_zero": [{"parameter": module.weight, "parameter_name": f"{name}.weight", "dim": 0, "shift": 0}]
            }
            if module.bias is not None:
                layer_config["set_to_zero"].append({"parameter": module.bias, "parameter_name": f"{name}.bias", "dim": 0, "shift": 0})
            formatted_params.append(layer_config)
            count += 1
            
    print(f"✅ [Config] Found {count} Linear layers.")
    return formatted_params

def setup_pruning_engine_wrapper(args, model, model_without_ddp, formatted_params):
    """初始化外部剪枝引擎 (已修复 AttributeError)"""
    
    # 1. 基础配置
    base_settings = {
        "pruning_momentum": 0.9,
        "prune_per_iteration": args.prune_per_iter,
        "start_pruning_after_n_iterations": 0,
        "pruning_threshold": -1.0, 
        "group_size": 16,
    }
    
    # 2. Tensorboard 初始化
    train_writer = None
    if args.output_dir:
        tb_dir = os.path.join(args.output_dir, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        train_writer = SummaryWriter(log_dir=tb_dir)

    print("🚀 [Engine] Initializing PyTorch Pruning Engine...")
    
    # 3. 引擎初始化
    engine = pytorch_pruning(
        for_pruning_parameters=formatted_params,
        pruning_settings=base_settings,
        log_folder=args.output_dir,
        latency_regularization=0.0,
        latency_target=-1.0,
        latency_look_up_table=""
    )
    
    # 4. 强制参数覆盖 (✨ 修复点在这里)
    engine.prune_per_iteration = args.prune_per_iter
    engine.method = 22
    engine.train_writer = train_writer
    engine.prune_neurons_max = 0
    engine.maximum_pruning_iterations = 0
    
    # 🔥 [新增] 手动初始化计数器，解决 AttributeError
    engine.pruning_iterations_done = 0 

    # 5. Helper 初始化
    target = model_without_ddp.backbone if hasattr(model_without_ddp, 'backbone') else model_without_ddp
    engine.init_pruning_helper(target, None, skip_pass=True)
    
    return engine