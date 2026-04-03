#!/home/yangz/.conda/envs/4D-humans/bin/python
import torch
import torch.nn as nn
import argparse
import sys
from pathlib import Path
import warnings

# Add paths to ensure we can import ModelManager and 4D-Humans deps
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append('/home/yangz/4D-Humans')

try:
    from model_manager import ModelManager
except ImportError:
    # Fallback usually needed if running from weird directory
    sys.path.append('/home/yangz/NViT-master/nvit')
    try:
        from model_manager import ModelManager
    except ImportError:
        print("Error: Could not import ModelManager. Check paths.")
        sys.exit(1)

class PrunedONNXModel(nn.Module):
    def __init__(self, checkpoint_path, cfg):
        super().__init__()
        # Initialize with CPU device
        self.mm = ModelManager({'device': 'cpu'})
        print(f"Loading Pruned Model from: {checkpoint_path}")
        self.mm.load_model(checkpoint_path=checkpoint_path)
        self.model = self.mm.model
        self.model.eval()
        
    def forward(self, x):
        batch = {'img': x}
        if hasattr(self.model, 'forward'):
            out = self.model(batch)
        
        pred_smpl_params = out['pred_smpl_params']
        return (pred_smpl_params['body_pose'], 
                pred_smpl_params['betas'], 
                pred_smpl_params['global_orient'],
                out['pred_cam'],
                out['pred_keypoints_3d'])

def export_onnx(args):
    device = torch.device('cpu')
    wrapper = PrunedONNXModel(args.checkpoint, None)
    wrapper.to(device)
    
    # Dummy Input (B, 3, 256, 256)
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    output_onnx = args.output
    
    print(f"Exporting to {output_onnx}...")
    torch.onnx.export(
        wrapper, dummy_input, output_onnx, verbose=False,
        input_names=['img'],
        output_names=['body_pose', 'betas', 'global_orient', 'pred_cam', 'pred_joints'],
        opset_version=12,
        dynamic_axes={'img': {0: 'batch'}, 'body_pose': {0: 'batch'}}
    )
    print(f"Success! Model saved to {output_onnx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='hmr2_pruned.onnx')
    args = parser.parse_args()
    export_onnx(args)
