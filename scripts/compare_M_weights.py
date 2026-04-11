import torch
import os

# ================= 配置路径 =================
import pyrootutils
PROJECT_ROOT = pyrootutils.find_root()
BASE_CKPT = os.path.expanduser("~/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt")
EXP_ROOT = str(PROJECT_ROOT / "output/ch5_prior_compare")
DEVICE = "cuda:0" # 在卡0运行
# ============================================

def get_sd(path):
    try:
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        return d.get('state_dict', d)
    except: return None

def verify():
    print(f"🚀 [GPU 0] 启动路径深度对齐校验...")
    base_sd = get_sd(BASE_CKPT)
    
    # 基础模型的 Key
    base_key = "backbone.blocks.0.attn.qkv.weight"
    if base_key not in base_sd:
        print(f"❌ 基础模型中找不到 {base_key}")
        return
    base_weight = base_sd[base_key]

    # 你的模型对应的 Key (根据刚才的 log 发现多了 .block)
    # 尝试两种可能的路径
    possible_exp_keys = [
        "nvit_backbone.blocks.0.block.attn.qkv.weight",
        "nvit_backbone.blocks.0.attn.qkv.weight",
        "model.nvit_backbone.blocks.0.block.attn.qkv.weight"
    ]

    model_dirs = sorted([d for d in os.listdir(EXP_ROOT) if d.startswith('M')])

    print("\n" + "="*110)
    print(f"{'实验组':<20} | {'匹配到的 Key':<55} | {'结果'}")
    print("-" * 110)

    for m_dir in model_dirs:
        ckpt_path = os.path.join(EXP_ROOT, m_dir, "checkpoints/last.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"{m_dir:<20} | {'[文件缺失]':<55} | -")
            continue
            
        exp_sd = get_sd(ckpt_path)
        if exp_sd is None: continue

        found_key = None
        for k in possible_exp_keys:
            if k in exp_sd:
                found_key = k
                break
        
        if found_key:
            exp_weight = exp_sd[found_key]
            if exp_weight.shape == base_weight.shape:
                diff = torch.abs(base_weight - exp_weight).mean().item()
                if diff == 0:
                    res = "✅ 完美匹配(Frozen)"
                elif diff < 1e-3:
                    res = f"🆗 已继承(Fine-tuned, Diff:{diff:.6f})"
                else:
                    res = f"❌ 权重偏移(Diff:{diff:.4f})"
            else:
                res = f"⚠️ 维度不符({list(exp_weight.shape)} vs {list(base_weight.shape)})"
            print(f"{m_dir:<20} | {found_key:<55} | {res}")
        else:
            # 自动搜索包含 qkv.weight 的任何键
            alt_key = next((k for k in exp_sd.keys() if "blocks.0." in k and "qkv.weight" in k), "None")
            print(f"{m_dir:<20} | {'未找到匹配Key (建议检查: ' + alt_key + ')':<55} | ❓ 路径错误")

    print("="*110)

if __name__ == "__main__":
    verify()
    