#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch

# 加载你的剪枝模型
ckpt = torch.load('your_pruned_model.pth', map_location='cpu')
model_state = ckpt['model'] if 'model' in ckpt else ckpt

# 检查一个具体的线性层（以第1个Block的MLP层为例）
# 正常的 DeiT-Base 维度应该是 3072
for key, value in model_state.items():
    if 'blocks.0.mlp.fc1.weight' in key:
        print(f"层名称: {key}")
        print(f"当前形状: {value.shape}") # 如果这里还是 3072，说明物理上没变
        
        # 统计非零元素
        non_zero = torch.count_nonzero(value)
        total = value.numel()
        print(f"非零参数比例: {non_zero/total:.2%}")
        break