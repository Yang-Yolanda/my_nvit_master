#!/home/yangz/.conda/envs/4D-humans/bin/python
import json
import os

def create_config_from_latency():
    # 1. 输入和输出路径
    latency_path = "latency.json"  # 确保这个文件就在当前目录下
    output_path = "/home/yangz/NViT-master/nvit/configs/hmr2_pruning_auto.json"

    if not os.path.exists(latency_path):
        print(f"❌ 错误: 找不到 {latency_path}，请先运行 get_latency_hmr2.py 生成它！")
        return

    print(f"📖 读取模型真实层名: {latency_path}...")
    with open(latency_path, 'r') as f:
        latency_data = json.load(f)

    # 2. 构建 Pruning Config
    # "Global" 是 NViT 必须的参数，随便给个默认值，命令行参数会覆盖它
    pruning_config = {
        "Global": 5120
    }

    count = 0
    skipped = 0
    
    for layer_name in latency_data.keys():
            if "backbone" in layer_name or "head" in layer_name:
                # ✅ 修复方案：填充 NViT 引擎预期的结构
                pruning_config[layer_name] = {
                    "compute_criteria_from": [
                        {
                            "parameter": "weight",
                            "type": "linear" # ViT 的 block 大多是 Linear 层
                        }
                    ]
                }
                count += 1
            else:
                skipped += 1

    # 3. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pruning_config, f, indent=4)

    print(f"✅ 成功生成配置: {output_path}")
    print(f"📊 共提取了 {count} 个可剪枝层 (跳过了 {skipped} 个无关层)")
    print("-" * 50)
    print("🚀 下一步：请更新你的训练命令：")
    print(f"   --pruning_config {output_path}")

if __name__ == '__main__':
    create_config_from_latency()