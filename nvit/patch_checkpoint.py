#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
from pathlib import Path

# 1. 设定路径
ckpt_path = '/home/yangz/NViT-master/nvit/output/hmr2_ddp_v1/pruned_final.pth'
# 原始官方 HMR2 权重的路径（ModelManager 需要它来初始化基础架构）
original_hmr2_path = "/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"

# 2. 加载旧的 checkpoint
print(f"正在读取: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')

# 3. 提取 state_dict (如果它现在嵌套在 'model' 下)
state_dict = ckpt['model'] if 'model' in ckpt else ckpt

# 4. 手动构建物理结构信息 (这是 ModelManager 压缩层的依据)
# 我们通过扫描 state_dict 中各层的 shape 来反推结构
pruned_structure = {"backbone": {"blocks": []}}

# 假设 HMR2 有 12 个 ViT Blocks
for i in range(12):
    fc1_weight = state_dict.get(f'backbone.blocks.{i}.mlp.fc1.weight')
    if fc1_weight is not None:
        # 反推维度：Linear(in_features, out_features)
        # weight 的 shape 是 [out_features, in_features]
        hidden_dim = fc1_weight.shape[0]
        input_dim = fc1_weight.shape[1]
        
        pruned_structure["backbone"]["blocks"].append({
            "mlp": {
                "in_features": input_dim,
                "hidden_features": hidden_dim,
                "out_features": input_dim # 通常 fc2 输出等于 fc1 输入
            }
        })

# 5. 封装成完整的格式
new_ckpt = {
    'state_dict': state_dict,
    'original_model_path': original_hmr2_path,
    'pruned_structure': pruned_structure,
    'is_pruned': True
}

# 6. 覆盖保存
torch.save(new_ckpt, ckpt_path)
print("✅ 补丁修复完成！'original_model_path' 和 'pruned_structure' 已补齐。")