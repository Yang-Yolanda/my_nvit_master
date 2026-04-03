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
sys.path.append('/home/yangz/4D-Humans')  # Standard path on this server

try:
    from model_manager import ModelManager
except ImportError:
    print("Error: Could not import ModelManager. Check sys.path.")
    sys.exit(1)

class PrunedONNXModel(nn.Module):
    """
    Minimal wrapper for ONNX export. 
    Strips away PHALP/Dict overhead and returns pure tensors.
    """
    def __init__(self, checkpoint_path, cfg):
        super().__init__()
        self.mm = ModelManager({'device': 'cpu'}) # Export on CPU is fine/safer
        print(f"Loading Pruned Model from: {checkpoint_path}")
        self.mm.load_model(checkpoint_path=checkpoint_path)
        self.model = self.mm.model
        self.model.eval()
        
    def forward(self, x):
        """
        Input: x (Batch, 3, 256, 256) Normalized Image
        Output: (pred_pose, pred_shape, pred_cam, pred_joints)
        """
        # Create input dict expected by HMR2
        # HMR2 typically expects check for 'img' and 'mask' keys in batch
        # But we want to see what self.model actually is. 
        # Usually it's the Pruned Model which might expect a tensor or dict.
        # Based on run_diagnosis, it expects a dict with 'img'.
        
        batch = {'img': x}
        
        # We might need to handle mask if the model strictly strictly requires it
        # But for inference usually mask is optional or we can fake it
        if hasattr(self.model, 'forward'):
            out = self.model(batch)
        
        # Extract meaningful tensors
        # smpl_params is usually a dict {body_pose, betas, global_orient...}
        # We want to flatten this for ONNX
        
        pred_smpl_params = out['pred_smpl_params']
        pred_cam = out['pred_cam']
        pred_keys = out['pred_keypoints_3d']
        
        # Return flattened tuple
        # body_pose: (B, 69) or (B, 23, 3) - depending on rotation format (aa)
        body_pose = pred_smpl_params['body_pose'] 
        betas = pred_smpl_params['betas']
        global_orient = pred_smpl_params['global_orient']
        
        return body_pose, betas, global_orient, pred_cam, pred_keys

def export_onnx(args):
    # cfg is needed for ModelManager init? Usually it takes a dict or path.
    # In run_diagnosis we passed cfg object. Here we try minimal.
    
    device = torch.device('cpu')
    wrapper = PrunedONNXModel(args.checkpoint, None)
    wrapper.to(device)
    
    # Dummy Input
    # Standard HMR input is 256x256
    dummy_input = torch.randn(1, 3, 256, 256, device=device)
    
    output_onnx = args.output
    
    print(f"Exporting to {output_onnx}...")
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_onnx,
        verbose=False,
        input_names=['img'],
        output_names=['body_pose', 'betas', 'global_orient', 'pred_cam', 'pred_joints'],
        opset_version=12,
        dynamic_axes={
            'img': {0: 'batch_size'},
            'body_pose': {0: 'batch_size'},
            'betas': {0: 'batch_size'},
            'global_orient': {0: 'batch_size'},
            'pred_cam': {0: 'batch_size'},
            'pred_joints': {0: 'batch_size'}
        }
    )
    
    # Verify
    import onnx
    onnx_model = onnx.load(output_onnx)
    onnx.checker.check_model(onnx_model)
    print(f"Success! Model saved to {output_onnx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pruned .pth')
    parser.add_argument('--output', type=str, default='hmr2_pruned.onnx')
    args = parser.parse_args()
    
    export_onnx(args)
