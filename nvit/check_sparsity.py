#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import torch.nn as nn
from model_manager import ModelManager
import sys
from pathlib import Path

# Fix paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append('/home/yangz/4D-Humans')

def check_sparsity():
    checkpoint_path = '/home/yangz/NViT-master/nvit/output/hmr2_ddp_v1/checkpoint_sparsity_Global_0.2.pth'
    print(f"Loading {checkpoint_path}...")
    
    # Load state dict directly first to avoid model init overhead if possible
    # But we need structure to know which are weights.
    # Let's load full model.
    try:
        mm = ModelManager({'device': 'cpu'})
        mm.load_model(checkpoint_path=checkpoint_path)
        model = mm.model
    except Exception as e:
        print(f"ModelManager load failed, trying direct state_dict: {e}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Just analyze state dict tensors
        total_params = 0
        zero_params = 0
        for k, v in state_dict.items():
            if 'weight' in k or 'bias' in k:
                if isinstance(v, torch.Tensor):
                    num = v.numel()
                    zeros = torch.sum(v == 0).item()
                    total_params += num
                    zero_params += zeros
        print(f"Total Params: {total_params}")
        print(f"Zero Params: {zero_params}")
        print(f"Global Sparsity: {zero_params/total_params:.4%}")
        return

    # If ModelManager works
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            num = param.numel()
            zeros = torch.sum(param == 0).item()
            total_params += num
            zero_params += zeros
            if zeros/num > 0.01:
                print(f"Layer {name}: Sparsity {zeros/num:.2%}")
    
    print("-" * 30)
    print(f"Total Parameters: {total_params}")
    print(f"Zero Parameters: {zero_params}")
    print(f"Global Weight Sparsity: {zero_params/total_params:.4%}")

if __name__ == "__main__":
    check_sparsity()
