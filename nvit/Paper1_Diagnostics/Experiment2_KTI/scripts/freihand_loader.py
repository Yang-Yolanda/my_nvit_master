
import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class FreiHANDDataset(Dataset):
    def __init__(self, root_dir, split='evaluation'):
        """
        FreiHAND Dataset Loader for HaMeR KTI Evaluation
        
        Args:
            root_dir (str): Path to folder containing 'evaluation' folder and json files
            split (str): 'evaluation' or 'training'
        """
        self.root = root_dir
        self.split = split
        
        # FreiHAND structure:
        # root/evaluation/rgb/*.jpg
        # root/evaluation_K.json (Note underscores)
        # root/evaluation_xyz.json
        
        self.img_dir = os.path.join(self.root, split, 'rgb')
        # Check if rgb exists, otherwise maybe directly in split
        if not os.path.exists(self.img_dir):
             self.img_dir = os.path.join(self.root, split)
             
        self.img_paths = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        
        # Load JSONs
        # Filename format: evaluation_K.json
        k_path = os.path.join(self.root, f"{split}_K.json")
        scale_path = os.path.join(self.root, f"{split}_scale.json")
        xyz_path = os.path.join(self.root, f"{split}_xyz.json")
        
        with open(k_path, 'r') as f:
            self.Ks = json.load(f)
            
        with open(scale_path, 'r') as f:
            self.scales = json.load(f)
            
        # Try to load XYZ (If available)
        self.xyz = None
        if os.path.exists(xyz_path):
            with open(xyz_path, 'r') as f:
                self.xyz = json.load(f)
        else:
            print(f"Warning: {xyz_path} not found. KTI calculation needs GT keypoints!")
            
    def __len__(self):
        return len(self.img_paths)
        
    def project_3d_to_2d(self, xyz, K):
        """Project 3D keypoints to 2D using camera matrix"""
        xyz = np.array(xyz)
        K = np.array(K)
        uv = np.matmul(K, xyz.T).T
        return uv[:, :2] / uv[:, 2:3]
        
    def get_bbox_from_kp2d(self, kp_2d, pad_factor=1.2):
        """Calculate BBox from keypoints with padding"""
        min_x = np.min(kp_2d[:, 0])
        max_x = np.max(kp_2d[:, 0])
        min_y = np.min(kp_2d[:, 1])
        max_y = np.max(kp_2d[:, 1])
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y
        
        # Square BBox
        size = max(width, height) * pad_factor
        
        x1 = center_x - size / 2
        y1 = center_y - size / 2
        return [x1, y1, size, size] # xywh

    def crop_and_resize(self, img, kp_2d, bbox, target_size=256):
        """Crop image and transform keypoints"""
        x1, y1, w, h = bbox
        
        # Transformation Matrix
        # 1. Translate (-x1, -y1)
        # 2. Scale (target / w)
        scale = target_size / w
        
        T = np.array([
            [scale, 0, -x1 * scale],
            [0, scale, -y1 * scale],
            [0, 0, 1]
        ])
        
        # Warp Image
        img_crop = cv2.warpAffine(img, T[:2], (target_size, target_size), flags=cv2.INTER_LINEAR)
        
        # Transform Keypoints
        n_kp = kp_2d.shape[0]
        ones = np.ones((n_kp, 1))
        kp_homo = np.hstack([kp_2d, ones]) # (N, 3)
        kp_trans = (T @ kp_homo.T).T # (N, 3) --> (N, 2) effectively
        
        return img_crop, kp_trans[:, :2]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path) # BGR
        if img is None:
            print(f"Error reading {img_path}")
            return self.__getitem__((idx + 1) % len(self))
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        K = self.Ks[idx]
        scale = self.scales[idx]
        
        sample = {'img_orig': img, 'img_path': img_path}
        
        if self.xyz:
            xyz = self.xyz[idx]
            kp_2d = self.project_3d_to_2d(xyz, K)
            
            # Calculate BBox & Crop
            bbox = self.get_bbox_from_kp2d(kp_2d)
            img_crop, kp_crop = self.crop_and_resize(img, kp_2d, bbox, target_size=256)
            
            # Create Normalized Tensor for HaMeR (0-1, then Normalize)
            # HaMeR uses: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float() / 255.0
            
            from torchvision import transforms
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = normalize(img_tensor)
            
            sample['img'] = img_tensor # This is what model needs
            sample['keypoints_2d'] = torch.tensor(kp_crop, dtype=torch.float32) # For KTI (in 256x256 space)
            sample['raw_keypoints'] = kp_2d
            
        return sample
