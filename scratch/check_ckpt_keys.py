import torch
ckpt_path = '/cpfs_infra/shared/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
keys = list(ckpt['state_dict'].keys())
print(f"Total keys: {len(keys)}")
print("First 20 keys:")
for k in keys[:20]:
    print(k)

# Check for 'backbone' or 'nvit_backbone'
backbone_keys = [k for k in keys if 'backbone' in k]
print(f"\nBackbone keys: {len(backbone_keys)}")
if backbone_keys:
    print("First 10 backbone keys:")
    for k in backbone_keys[:10]:
        print(k)
