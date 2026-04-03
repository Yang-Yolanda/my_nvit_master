#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import json
import argparse
import os
from pathlib import Path
import numpy as np

def generate_pruning_config(checkpoint_path, output_path):
    print(f"🔍 Loading checkpoint from: {checkpoint_path}")
    
    # 1. 加载模型权重
    # 注意：根据你的保存方式，权重可能在 'model', 'state_dict' 或直接是字典
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    config_dict = {}
    
    print("⚙️ Analyzing layer sparsity...")
    
    # 2. 遍历所有参数
    for name, param in state_dict.items():
        # 我们只关心权重 (weight)，不关心偏置 (bias)
        if 'weight' not in name:
            continue
            
        # 排除一些不需要计算比例的层（例如 Positional Embedding, CLS token 等）
        # 通常我们只关心 Linear 层 (attn, mlp, head)
        keywords = ['attn', 'mlp', 'head', 'discriminator', 'proj', 'fc']
        if not any(k in name for k in keywords):
            continue

        # 3. 计算保留比例 (Ratio)
        # 逻辑：计算非零元素的数量 / 总元素数量
        tensor = param.float().numpy()
        total_params = tensor.size
        
        # 使用极小阈值判断是否为0 (防止浮点数精度问题)
        non_zero_params = np.count_nonzero(np.abs(tensor) > 1e-8)
        
        ratio = non_zero_params / total_params
        
        # 如果比例是 1.0 (没剪枝) 或者 0.0 (全剪了)，根据需要决定是否保留
        # 这里为了完整性，我们全部保留
        
        # 4. 格式化 Key 名称
        # 将 "backbone.blocks.0.attn.qkv.weight" -> "backbone.blocks.0.attn.qkv"
        # 以匹配你之前的 JSON 格式
        json_key = name.replace('.weight', '')
        
        # 移除 DDP 可能带来的 'module.' 前缀
        if json_key.startswith('module.'):
            json_key = json_key[7:]
            
        config_dict[json_key] = ratio
        
        # 打印部分日志
        if 'blocks.0.' in name or 'smpl_head' in name:
            print(f"   Analyzed {json_key}: {ratio:.4f}")

    # 5. 保存为 JSON
    print(f"💾 Saving configuration to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    print("✅ Done! You can now use this JSON file for training.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate pruning config from checkpoint')
    
    # 输入：你剪枝过的模型路径 (例如 output/hmr2_pruned/checkpoint.pth)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the pruned model checkpoint')
    
    # 输出：你想保存的 json 文件名
    parser.add_argument('--output', type=str, default='pruning_config.json', help='Path to save the output JSON')
    
    args = parser.parse_args()
    
    generate_pruning_config(args.checkpoint, args.output)