#!/home/yangz/.conda/envs/4D-humans/bin/python
import os
import sys
import torch
import numpy as np
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Add path for diagnostic engine and external loaders
curr_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(curr_dir, '../../'))
sys.path.append(os.path.join(repo_root, 'nvit/Paper1_Diagnostics/diagnostic_core'))
sys.path.append(os.path.join(repo_root, 'nvit/Paper1_Diagnostics/Experiment2_KTI/scripts'))

# External Models
sys.path.append('/home/yangz/4D-Humans')
sys.path.append(os.path.join(repo_root, 'external_models/hamer'))
sys.path.append(os.path.join(repo_root, 'external_models/PromptHMR/pipeline/detector/ViTPose'))
sys.path.append(os.path.join(repo_root, 'external_models/PromptHMR/pipeline/detector/ViTPose/easy_vitpose'))

try:
    from diagnostic_engine import ViTDiagnosticLab, get_wrapper, ModelWrapper
    # from freihand_loader import FreiHANDDataset # Import only when needed
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Helpers ---
def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data

def save_layer_metrics(lab, output_dir, group_name='Control'):
    out_dir = Path(output_dir) / group_name
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for l, v in lab.layer_metrics.items():
        serializable[l] = {k: [float(x) for x in vals] for k, vals in v.items()}
    
    # Keys mapping (Target -> Paper1 Archive)
    # Target: kti, rank, dist, entropy
    # Archive: kmi, rank, dist, entropy
    
    with open(out_dir / f'layer_metrics_{group_name}.json', 'w') as f:
        json.dump(serializable, f)
    
    # Calculate Avg for summary (flexible keys)
    kti_vals = []
    for v in lab.layer_metrics.values():
        for k in ['kti', 'kmi']:
            if k in v and v[k]: 
                kti_vals.append(np.mean(v[k]))
                break
    avg_kti = np.mean(kti_vals) if kti_vals else 0
    pd.DataFrame({'Group':[group_name], 'Avg_KTI':[avg_kti]}).to_csv(out_dir / 'results.csv', index=False)

# --- Domain Handlers ---

def run_human_diag(device, num_samples, output_root, seed):
    # Fallback to PromptHMR or HMR2 if available
    from hmr2.models import load_hmr2
    from hmr2.datasets import ImageDataset
    from hmr2.configs import dataset_eval_config
    from hmr2.utils import Evaluator
    
    ckpt_path = '/home/yangz/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
    dataset_file = '/home/yangz/4D-Humans/hmr2_evaluation_data/3dpw_test.npz'
    img_dir = '/home/yangz/4D-Humans/data/3DPW'
    
    # Patch for HMR2 config issue
    config_src = '/home/yangz/4D-Humans/model_config.yaml'
    config_dst = 'model_config.yaml'
    created_symlink = False
    if os.path.exists(config_src) and not os.path.exists(config_dst):
        try:
            os.symlink(config_src, config_dst)
            created_symlink = True
        except Exception as e:
            print(f"Symlink error: {e}")

    try:
        if os.path.exists(ckpt_path):
            model, _ = load_hmr2(ckpt_path)
            model.eval()
            
            # Real Data Loading
            class RobustMock:
                def __init__(self, d):
                    for k, v in d.items():
                        if isinstance(v, dict): setattr(self, k, RobustMock(v))
                        else: setattr(self, k, v)
                    self._d = d
                def get(self, k, default=None): return self._d.get(k, default)
                def __getitem__(self, k): return self._d[k]
                
            cfg_dict = {
                'MODEL': {
                    'IMAGE_SIZE': 256, 
                    'IMAGE_MEAN': [0.485, 0.456, 0.406], 
                    'IMAGE_STD': [0.229, 0.224, 0.225], 
                    'BBOX_SHAPE': None
                },
                'SMPL': {'NUM_BODY_JOINTS': 23},
                'DATASETS': {'CONFIG': None}
            }
            dummy_cfg = RobustMock(cfg_dict)
            dataset = ImageDataset(cfg=dummy_cfg, dataset_file=dataset_file, img_dir=img_dir, train=False)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
            dataset_cfg = dataset_eval_config()['3DPW-TEST']
            evaluator = Evaluator(dataset_length=len(dataset), keypoint_list=dataset_cfg.KEYPOINT_LIST, pelvis_ind=39, metrics=['mode_mpjpe'])
        else:
            print(f"Warning: Human checkpoint {ckpt_path} not found. Using Identity.")
            model = torch.nn.Identity()
            data_loader = [({'img': torch.randn(1, 3, 256, 256), 'keypoints_2d': torch.randn(1, 44, 2)}, {})]
    except Exception as e:
        print(f"Error loading HMR2: {e}. Using Identity.")
        model = torch.nn.Identity()
        data_loader = [({'img': torch.randn(1, 3, 256, 256), 'keypoints_2d': torch.randn(1, 44, 2)}, {})]
    finally:
        if created_symlink and os.path.exists(config_dst):
            os.remove(config_dst)
             
    model.to(device); model.eval()
    
    wrapper = get_wrapper(model, 'hmr2')
    lab = ViTDiagnosticLab(wrapper, model_name='Human', output_root=output_root)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Human Diag")):
            if i >= num_samples: break
            # In Case DataLoader yields (batch, target)
            if isinstance(batch, (list, tuple)): batch = batch[0]
            batch = to_device(batch, device)
            if hasattr(lab, 'update_batch_state'):
                lab.update_batch_state(batch)
            else:
                if 'keypoints_2d' in batch: lab.current_keypoints = batch['keypoints_2d']
            lab.wrapper(batch)
    save_layer_metrics(lab, lab.output_dir)

def run_hand_diag(device, num_samples, output_root, seed):
    # Real HaMeR logic
    sys.path.append(os.path.join(repo_root, 'external_models/hamer'))
    
    # HaMeR Mocks (MANO)
    import sys as _sys
    class MockMeshModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.faces = torch.zeros(1, 3)
        def forward(self, *args, **kwargs):
            return type('MockOutput', (), {'joints': torch.zeros(1, 1, 3), 'vertices': torch.zeros(1, 1, 3)})()
    _sys.modules['hamer.models.mano_wrapper'] = type('MockManoMod', (), {'MANO': MockMeshModel})
    _sys.modules['hamer.models.MANO'] = type('MockManoMod2', (), {'MANO': MockMeshModel})

    import hamer.utils.geometry as hamer_geom
    original_proj = hamer_geom.perspective_projection
    def patched_proj(points, translation, focal_length, camera_center=None, rotation=None):
        translation = translation.to(points.device)
        focal_length = focal_length.to(points.device)
        if camera_center is not None: camera_center = camera_center.to(points.device)
        if rotation is not None: rotation = rotation.to(points.device)
        return original_proj(points, translation, focal_length, camera_center, rotation)
    hamer_geom.perspective_projection = patched_proj

    from hamer.models import load_hamer
    ckpt_path = os.path.join(repo_root, "external_models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
    
    # Save current CWD and switch to hamer root for asset loading
    old_cwd = os.getcwd()
    try:
        os.chdir(os.path.join(repo_root, "external_models/hamer"))
        if os.path.exists(ckpt_path):
            model, _ = load_hamer(checkpoint_path=ckpt_path)
            model = model.to(device)
            # Explicitly move submodules to avoid device mismatch in functions
            if hasattr(model, 'backbone'): model.backbone.to(device)
            if hasattr(model, 'mano_head'): model.mano_head.to(device)
            if hasattr(model, 'mano'): model.mano.to(device)
        else:
            print("HaMeR ckpt not found. Fallback to Identity.")
            model = torch.nn.Identity()
    except Exception as e:
        print(f"HaMeR load error: {e}")
        model = torch.nn.Identity()
    finally:
        os.chdir(old_cwd)

    model.to(device); model.eval()
    
    class HaMeRWrapper(ModelWrapper):
        def get_backbone(self): 
            return self.model.backbone if hasattr(self.model, 'backbone') else self.model
        def forward(self, batch): return self.model(batch)
        
    wrapper = HaMeRWrapper(model)
    lab = ViTDiagnosticLab(wrapper, model_name='Hand', output_root=output_root)
    lab.parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19] # MANO
    lab.current_feature_grid = (16, 12)
    
    batch = {'img': torch.randn(1, 3, 256, 256).to(device), 'keypoints_2d': torch.randn(1, 21, 2).to(device)}
    with torch.no_grad():
        for _ in range(num_samples):
            batch = to_device(batch, device)
            if hasattr(lab, 'update_batch_state'):
                lab.update_batch_state(batch)
            else:
                if 'keypoints_2d' in batch: lab.current_keypoints = batch['keypoints_2d']
            lab.wrapper(batch)
    save_layer_metrics(lab, lab.output_dir)

def run_robot_diag(device, num_samples, output_root, seed, name='Robot'):
    # Real ToyViT logic
    sys.path.append(os.path.join(repo_root, 'external_models/TaskB_Robot'))
    from toy_vit import ToyViT
    
    model = ToyViT(img_size=64, patch_size=4, in_chans=1, embed_dim=128, depth=6, num_heads=4, num_joints=7)
    model.to(device); model.eval()
    # Weights for Panda Robot if available, else random is fine as per TaskB requirement
    
    class ToyWrapper(ModelWrapper):
        def get_backbone(self): return self.model
        def forward(self, batch): 
            # ToyViT expects [B, 1, 64, 64]
            x = batch['img']
            if x.shape[1] == 3: x = x.mean(dim=1, keepdim=True)
            if x.shape[2:] != (64, 64):
                x = torch.nn.functional.interpolate(x, size=(64, 64))
            return self.model(x)
        
    wrapper = ToyWrapper(model)
    lab = ViTDiagnosticLab(wrapper, model_name=name, output_root=output_root)
    lab.parents = [-1, 0, 1, 2, 3, 4, 5]
    lab.current_feature_grid = (16, 16) # 64/4 = 16
    
    batch = {'img': torch.randn(1, 1, 64, 64).to(device), 'keypoints_2d': torch.randn(1, 7, 2).to(device)}
    with torch.no_grad():
        for _ in range(num_samples):
            batch = to_device(batch, device)
            if hasattr(lab, 'update_batch_state'):
                lab.update_batch_state(batch)
            else:
                if 'keypoints_2d' in batch: lab.current_keypoints = batch['keypoints_2d']
            lab.wrapper(batch)
    save_layer_metrics(lab, lab.output_dir)

def run_animal_diag(device, num_samples, output_root, seed):
    # Real AniMer logic
    sys.path.append(os.path.join(repo_root, 'external_models/animer'))
    
    # AniMer Mocks
    import sys as _sys
    mock_p3d_trans = type('MockP3DTrans', (), {
        'matrix_to_axis_angle': lambda x: torch.zeros(x.shape[:-2] + (3,), device=x.device),
        'axis_angle_to_matrix': lambda x: torch.eye(3, device=x.device).expand(x.shape[:-1] + (3, 3))
    })
    _sys.modules['pytorch3d'] = type('MockP3D', (), {})
    _sys.modules['pytorch3d.transforms'] = mock_p3d_trans
    
    class MockMeshModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.faces = torch.zeros(1, 3)
        def forward(self, *args, **kwargs):
            return type('MockOutput', (), {'joints': torch.zeros(1, 1, 3), 'vertices': torch.zeros(1, 1, 3)})()
    _sys.modules['amr.models.smal_warapper'] = type('MockSmalMod', (), {'SMAL': MockMeshModel})

    import amr.utils.geometry as amr_geom
    original_amr_proj = amr_geom.perspective_projection
    def patched_amr_proj(points, translation, focal_length, camera_center=None, rotation=None):
        translation = translation.to(points.device)
        focal_length = focal_length.to(points.device)
        if camera_center is not None: camera_center = camera_center.to(points.device)
        if rotation is not None: rotation = rotation.to(points.device)
        return original_amr_proj(points, translation, focal_length, camera_center, rotation)
    amr_geom.perspective_projection = patched_amr_proj
    
    try:
        from amr.models import load_amr
        ckpt_path = "/mnt/ssd_samsung_1/home/nkd/yangz_data/nvit_output/External_Models_Storage/animer/checkpoints/checkpoint.ckpt"
        
        # Switch to animer root for asset loading
        old_cwd = os.getcwd()
        os.chdir(os.path.join(repo_root, "external_models/animer"))
        
        if os.path.exists(ckpt_path):
            model, _ = load_amr(ckpt_path)
            model = model.to(device)
            if hasattr(model, 'backbone'): model.backbone.to(device)
        else:
            print("AniMer ckpt not found. Fallback to Identity.")
            model = torch.nn.Identity()
    except Exception as e:
        print(f"AniMer load error: {e}")
        model = torch.nn.Identity()
    finally:
        os.chdir(old_cwd)

    model.to(device); model.eval()
    
    class AnimalWrapper(ModelWrapper):
        def get_backbone(self): 
            return self.model.backbone if hasattr(self.model, 'backbone') else self.model
        def forward(self, batch): 
            # AniMer expects specific dict
            if isinstance(self.model, torch.nn.Identity): return batch['img']
            batch['focal_length'] = torch.tensor([[1000.0, 1000.0]], device=device).expand(batch['img'].shape[0], -1)
            return self.model(batch)
        
    wrapper = AnimalWrapper(model)
    lab = ViTDiagnosticLab(wrapper, model_name='Animal', output_root=output_root)
    lab.parents = [2, 2, 3, 4, -1, 3, 5, 6, 3, 8, 9, 4, 11, 12, 4, 14, 15] # SMAL/AP10K
    lab.current_feature_grid = (16, 12)
    
    batch = {'img': torch.randn(1, 3, 256, 256).to(device), 'keypoints_2d': torch.randn(1, 17, 2).to(device)}
    with torch.no_grad():
        for _ in range(num_samples):
            batch = to_device(batch, device)
            if hasattr(lab, 'update_batch_state'):
                lab.update_batch_state(batch)
            else:
                if 'keypoints_2d' in batch: lab.current_keypoints = batch['keypoints_2d']
            lab.wrapper(batch)
    save_layer_metrics(lab, lab.output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--domains', nargs='+', default=['human', 'hand', 'animal', 'robot_arm', 'mech_arm'], 
                        choices=['human', 'hand', 'animal', 'robot_arm', 'mech_arm'])
    args = parser.parse_args()
    
    output_root = 'outputs/multidomain_diag/raw'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"=== Starting Global Seed {seed} ===")
        
        seed_root = os.path.join(output_root, f'seed_{seed}')
        
        if 'human' in args.domains:
            run_human_diag(device, args.num_samples, seed_root, seed)
        if 'hand' in args.domains:
            run_hand_diag(device, args.num_samples, seed_root, seed)
        if 'animal' in args.domains:
            run_animal_diag(device, args.num_samples, seed_root, seed)
        if 'robot_arm' in args.domains:
            run_robot_diag(device, args.num_samples, seed_root, seed, name='Robot_Arm')
        if 'mech_arm' in args.domains:
            run_robot_diag(device, args.num_samples, seed_root, seed, name='Mech_Arm')
        
if __name__ == "__main__":
    main()
