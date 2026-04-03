#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import json
import time
import numpy as np
from hmr2.models import load_hmr2
from tqdm import tqdm

def benchmark_layer(layer, input_shape, device='cuda', runs=100):
    """
    测量单个层的耗时
    """
    # 创建 dummy input
    # Linear 层输入通常是 (B, N, C) 或 (B, C)
    if isinstance(layer, torch.nn.Linear):
        # 假设输入是 Token 序列 (Batch, Tokens, In_Features)
        # HMR2 ViT 的 tokens 数量通常是 197 (14x14+1)
        dummy_input = torch.randn(1, 197, layer.in_features).to(device)
    elif isinstance(layer, torch.nn.Conv2d):
        dummy_input = torch.randn(1, layer.in_channels, 32, 32).to(device)
    else:
        return 0.0

    layer.to(device)
    layer.eval()

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = layer(dummy_input)
    
    # 测量
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = layer(dummy_input)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time_ms = ((end - start) / runs) * 1000
    return avg_time_ms

def main():
    # 1. 设置路径
    checkpoint_path = "/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt"
    output_file = "latency.json"
    device = 'cuda'

    print(f"🔄 Loading HMR2 from {checkpoint_path}...")
    model, _ = load_hmr2(checkpoint_path)
    model.to(device)
    model.eval()

    # 2. 获取所有 Linear 层 (Pruning 主要针对 Linear)
    latency_table = {}
    
    print("🚀 Starting latency measurement (Layer-wise)...")
    
    # 遍历 backbone 和 head
    # 这里的 name 需要和 pruning_config.json 里的名字对应
    for name, module in tqdm(model.named_modules()):
        # NViT 主要剪枝 Linear 层 (qkv, proj, fc1, fc2)
        if isinstance(module, torch.nn.Linear):
            # 测量该层耗时
            latency = benchmark_layer(module, input_shape=None, device=device)
            latency_table[name] = latency
            
    # 3. 补充一些总耗时信息 (可选)
    print("📏 Measuring Total Inference Latency...")
    dummy_img = torch.randn(1, 3, 256, 256).to(device)
    batch = {'img': dummy_img}
    
    # 预热
    for _ in range(20):
        with torch.no_grad():
            _ = model(batch)
            
    # 测总耗时
    runs = 200
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(batch)
    torch.cuda.synchronize()
    t1 = time.time()
    
    total_latency = ((t1 - t0) / runs) * 1000
    print(f"✅ Total Model Latency: {total_latency:.2f} ms")

    # 4. 保存文件
    with open(output_file, "w") as f:
        json.dump(latency_table, f, indent=4)
    
    print(f"💾 Latency table saved to: {output_file}")
    print(f"📊 Captured {len(latency_table)} layers.")

if __name__ == "__main__":
    main()