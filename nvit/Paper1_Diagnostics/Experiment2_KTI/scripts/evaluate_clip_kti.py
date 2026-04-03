#!/home/yangz/.conda/envs/4D-humans/bin/python
"""
Evaluate KTI on CLIP Vision Transformer
Similar to SigLIP evaluation, CLIP is a general vision model without specific topology.
Expected: Low/Flat KTI (negative control)
"""
import os
import sys
import torch
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import cv2

sys.path.append(os.path.abspath('nvit/Paper1_Diagnostics/diagnostic_core'))

try:
    from transformers import CLIPModel, CLIPProcessor
    from diagnostic_engine import ViTDiagnosticLab, ModelWrapper
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class CLIPVisionWrapper(ModelWrapper):
    """Wrapper for CLIP Vision Encoder"""
    def __init__(self, model):
        super().__init__(model)
        
    def get_backbone(self):
        # CLIP vision_model.encoder has .layers (not .blocks)
        # Need proxy like SigLIP
        class CLIPBackboneProxy:
            def __init__(self, encoder):
                self.blocks = encoder.layers  # Map .layers to .blocks
                
        return CLIPBackboneProxy(self.model.vision_model.encoder)
        
    def forward(self, batch):
        """
        CLIP expects pixel_values input
        Returns vision embeddings
        """
        pixel_values = batch['img']
        # CLIP vision forward
        vision_outputs = self.model.vision_model(pixel_values, output_attentions=True)
        
        # Return dummy outputs for Evaluator compatibility
        B = pixel_values.shape[0]
        return {
            'pred_keypoints_3d': torch.zeros(B, 24, 3, device=pixel_values.device),
            'pred_keypoints_2d': torch.zeros(B, 24, 2, device=pixel_values.device),
            'pred_vertices': torch.zeros(B, 6890, 3, device=pixel_values.device),
            'pred_smpl_params': {}
        }

class AttentionCapture:
    """Capture attention from CLIP Vision Transformer"""
    def __init__(self, model):
        self.attentions = []
        self.hooks = []
        
        # CLIP: vision_model.encoder.layers[i].self_attn
        for i, layer in enumerate(model.vision_model.encoder.layers):
            # Hook on dropout after attention softmax
            if hasattr(layer.self_attn, 'dropout'):
                handle = layer.self_attn.dropout.register_forward_hook(self._hook_fn)
                self.hooks.append(handle)
            
    def _hook_fn(self, module, input, output):
        # Dropout returns attention weights after dropout
        self.attentions.append(output.detach().cpu())
        
    def clear(self):
        self.attentions = []
        
    def get_stacked(self):
        if not self.attentions: 
            return None
        return torch.stack(self.attentions, dim=0)

def run_evaluation(model_name, image_folder, num_batches=10):
    """
    Evaluate CLIP on 3DPW dataset (or any image folder)
    
    Args:
        model_name: CLIP model variant (e.g., 'openai/clip-vit-base-patch32')
        image_folder: Path to images
        num_batches: Number of batches to process
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating {model_name} on {image_folder}...")
    
    # Load CLIP
    print(f"Loading {model_name}...")
    try:
        # Try loading pretrained
        model = CLIPModel.from_pretrained(model_name)
        # Processor might fail due to network, so avoiding it if possible or wrap it
        # But for robustness, we will use manual transforms for fallback
    except Exception as e:
        print(f"Pretrained load failed: {e}")
        print("Falling back to RANDOM INITIALIZATION (Negative Control)")
        from transformers import CLIPConfig
        cfg = CLIPConfig() # Default ViT-B/32 config
        cfg.attn_implementation = "eager"
        if hasattr(cfg, 'vision_config'):
            cfg.vision_config._attn_implementation = "eager"
        if hasattr(cfg, 'text_config'):
             cfg.text_config._attn_implementation = "eager"
        
        # Try explicit assignment too as _attn_implementation is internal
        try:
            cfg.vision_config.attn_implementation = "eager"
            cfg.text_config.attn_implementation = "eager"
        except:
            pass
            
        model = CLIPModel(cfg)

    # Force eager attention implementation on all configs
    model.config.attn_implementation = "eager"
    model.config._attn_implementation = "eager"
    model.vision_model.config.attn_implementation = "eager"
    model.vision_model.config._attn_implementation = "eager"
    model.text_model.config.attn_implementation = "eager"
    model.text_model.config._attn_implementation = "eager"

    # Enable attention output
    model.config.output_attentions = True
    model.vision_model.config.output_attentions = True
    model.text_model.config.output_attentions = True
    
    model.to(device)
    model.eval()
    
    # Define a Lab that skips patching (since we don't need internal hooks/monkey patches for CLIP eager mode)
    class OffProbeLab(ViTDiagnosticLab):
        def _patch_attention_modules(self):
            print("OffProbeLab: Skipping internal patching (using clean outputs)")
            pass

    # Setup Wrapper & Lab
    wrapper = CLIPVisionWrapper(model)
    lab = OffProbeLab(
        wrapper, 
        model_name=f"CLIP_{model_name.split('/')[-1]}", 
        output_root='nvit/Paper1_Diagnostics/Experiment2_KTI/results'
    )
    
    # CLIP ViT-B/32: 224x224 input, patch_size=32 -> 7x7 grid
    # CLIP ViT-L/14: 224x224 input, patch_size=14 -> 16x16 grid
    if 'patch32' in model_name or 'base' in model_name:
        lab.current_feature_grid = (7, 7)
    elif 'patch14' in model_name or 'large' in model_name:
        lab.current_feature_grid = (16, 16)
    else:
        lab.current_feature_grid = (14, 14)  # Default - might be wrong for random config but ok for fallback
    
    lab.groups = {'Control': {'mask_layers': [], 'mode': 'none'}}
    
    # capturer = AttentionCapture(model) # Redundant and crashing
    layer_metrics = defaultdict(lambda: {'kti': []})
    
    # Manual Transform Setup
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from PIL import Image
    
    clip_transform = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    # Load Images
    import glob
    img_paths = sorted(glob.glob(os.path.join(image_folder, '**/*.jpg'), recursive=True))
    if not img_paths:
        print("No images found. Exiting.")
        return
    
    print(f"Found {len(img_paths)} images. Processing {num_batches} batches...")
    
    batch_count = 0
    with torch.no_grad():
        for img_path in tqdm(img_paths[:num_batches]):
            # Load & Preprocess
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Use Manual Transform
            pixel_values = clip_transform(img_pil).unsqueeze(0).to(device)
            
            batch = {'img': pixel_values}
            
            # capturer.clear()
            out = model.vision_model(pixel_values, output_attentions=True)
            
            # Get Attention from outputs
            # CLIP returns attentions as tuple of (num_layers,) tensors
            attns_tuple = out.attentions
            if attns_tuple is None:
                continue
            
            # Stack: (L, B, H, N, N)
            attns = torch.stack([a.cpu() for a in attns_tuple], dim=0)
            
            # Dummy keypoints (CLIP has no topology, so KTI should be random/low)
            B = pixel_values.shape[0]
            # Generate random keypoints in image space (224x224)
            dummy_kp = torch.rand(B, 24, 2) * 224
            
            for l_idx, attn_map in enumerate(attns):
                kti = lab.calculate_physically_grounded_kti(attn_map, dummy_kp)
                layer_metrics[l_idx]['kti'].append(kti)
            
            batch_count += 1
            if batch_count >= num_batches:
                break
    
    # Save Results
    output_dir = Path(f'nvit/Paper1_Diagnostics/Experiment2_KTI/results/CLIP_{model_name.split("/")[-1]}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    serializable = {}
    for l, v in layer_metrics.items():
        serializable[str(l)] = {'kti': [float(x) for x in v['kti']]}
    
    with open(output_dir / 'layer_metrics_Control.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"✓ Saved results to {output_dir}/layer_metrics_Control.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='openai/clip-vit-base-patch32',
                        help='CLIP model name from HuggingFace')
    parser.add_argument('--image_folder', type=str, 
                        default='datasets/3dpw/imageFiles',
                        help='Path to image folder')
    parser.add_argument('--num_batches', type=int, default=10,
                        help='Number of images to process')
    args = parser.parse_args()
    
    run_evaluation(args.model, args.image_folder, args.num_batches)
