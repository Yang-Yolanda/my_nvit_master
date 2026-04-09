import torch
import sys
sys.path.append("/home/yangz/4D-Humans")
try:
    ckpt = torch.load('/home/yangz/NViT-master/nvit/Paper1_Diagnostics/checkpoints/ft_T2-A-H-Baseline.ckpt', map_location='cpu')
    s = ckpt['state_dict']
    blocks = [k for k in s.keys() if 'backbone.blocks' in k]
    nums = [int(k.split('backbone.blocks.')[1].split('.')[0]) for k in blocks if 'backbone.blocks.' in k]
    if nums:
        print("Max block index:", max(nums))
        print("Total blocks:", max(nums) + 1)
    else:
        print("No backbone.blocks found.")
except Exception as e:
    print(e)
