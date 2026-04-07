#!/home/yangz/.conda/envs/4D-humans/bin/python
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import Dataset
from torchvision import transforms

# --- 常量定义 ---
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
SMPL_JOINTS = 24
TOTAL_JOINTS = 44  # HMR2 通常需要 44 个点 (24 SMPL + Extra)

class ThreeDPWDataset(Dataset):
    def __init__(self, config: Any, split: str = 'test', transform=None):
        """
        Args:
            config: 包含 data_path 等配置的对象
            split: 'train', 'validation', or 'test'
            transform: 图像预处理
        """
        super().__init__()
        self.split = split
        
        # 路径配置 (优先读取 config, 兜底使用默认路径)
        default_path = Path('/home/yangz/4D-Humans/data')
        self.data_root = Path(getattr(config, 'data_path', default_path))
        self.sequence_dir = self.data_root / "3DPW" / "sequenceFiles" / split
        self.image_dir = self.data_root / "3DPW" / "imageFiles"
        
        self.img_size = (256, 256)
        
        # 图像增强
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
            ])
        else:
            self.transform = transform

        self.data_dict = {}
        self.data_len = 0
        self._load_index()
        print(f"✅ [ThreeDPW] {split} 集加载完成: 共 {self.data_len} 个样本")

    def _load_index(self):
        """扫描所有 .npz 文件并建立索引"""
        if not self.sequence_dir.exists():
            raise FileNotFoundError(f"目录不存在: {self.sequence_dir}")

        print(f"🔄 正在扫描序列文件: {self.sequence_dir}...")
        npz_files = sorted(list(self.sequence_dir.glob("*.npz")))
        
        image_paths = []
        npz_paths = []
        frame_idxs = []
        seq_names = []
        
        for npz_file in npz_files:
            try:
                with np.load(npz_file, allow_pickle=True) as data:
                    # 1. 处理序列名 (解决 numpy bytes/str 编码问题)
                    seq_name = str(data['seq_name'])
                    if 'b\'' in seq_name:
                         seq_name = data['seq_name'].item().decode('utf-8')

                    # 2. 获取帧数
                    num_frames = data['jointPositions'].shape[1] 
                    
                    # 3. 建立索引
                    for frame_idx in range(num_frames):
                        # 检查图片是否存在 (可选：为了速度可以注释掉 os.path.exists)
                        img_name = f"image_{frame_idx:05d}.jpg"
                        img_path = self.image_dir / seq_name / img_name
                        
                        image_paths.append(str(img_path))
                        npz_paths.append(str(npz_file))
                        frame_idxs.append(frame_idx)
                        seq_names.append(seq_name)
            except Exception as e:
                print(f"⚠️ 跳过损坏文件 {npz_file.name}: {e}")
                
        # 转化为 Numpy Array，以切断 Python 多进程 List[Dict] 引发的引用计数 (Copy-on-Write 泄漏)
        self.data_dict = {
            'image_path': np.array(image_paths, dtype=np.string_),
            'npz_path': np.array(npz_paths, dtype=np.string_),
            'frame_idx': np.array(frame_idxs, dtype=np.int32),
            'seq_name': np.array(seq_names, dtype=np.string_)
        }
        self.data_len = len(frame_idxs)

    def __len__(self):
        return self.data_len

    def _get_smpl_params(self, data: Any, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        🔥 核心适配器：将数据集中的 Pose/Shape 转换为模型需要的格式
        返回: (global_orient, body_pose, betas, has_real_smpl)
        """
        # 1. 提取 Pose (72维)
        # 优先顺序: norm_poses -> poses -> 全0
        pose = np.zeros(72, dtype=np.float32)
        has_pose = False
        
        if 'norm_poses' in data:
            pose = data['norm_poses'][0, idx]
            has_pose = True
        elif 'poses' in data:
            pose = data['poses'][0, idx]
            has_pose = True
            
        # 2. 提取 Betas (10维)
        beta = np.zeros(10, dtype=np.float32)
        if 'betas' in data:
            raw_beta = data['betas']
            # 处理 shape (1, 10) 或 (1, N, 10) 的情况
            if raw_beta.ndim == 2: # (1, 10)
                beta = raw_beta[0]
            elif raw_beta.ndim == 3: # (1, N, 10)
                beta = raw_beta[0, idx]

        # 3. 切分数据
        global_orient = torch.from_numpy(pose[:3]).float()
        body_pose = torch.from_numpy(pose[3:]).float()
        betas = torch.from_numpy(beta[:10]).float()
        
        # 4. 标记位：如果数据全是 0，说明是假的，Loss 计算时权重设为 0
        has_real_smpl = 1.0 if has_pose else 0.0
        
        return global_orient, body_pose, betas, has_real_smpl

    def __getitem__(self, idx):
        # Decode numpy strings back to python str without affecting reference counts of parent memory
        img_path = self.data_dict['image_path'][idx].decode('utf-8')
        npz_path = self.data_dict['npz_path'][idx].decode('utf-8')
        f_idx = int(self.data_dict['frame_idx'][idx])
        
        # --- A. 读取图像 ---
        img_cv2 = cv2.imread(img_path)
        if img_cv2 is None:
            # 容错：如果读图失败，递归读取下一个
            print(f"🚨 读图失败: {img_path}, 尝试下一个...")
            return self.__getitem__((idx + 1) % len(self))
            
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_cv2.shape[:2]

        # --- B. 读取并处理 NPZ 数据 ---
        with np.load(npz_path) as data:
            
            # 1. 调用适配器获取 SMPL 参数 (你的核心逻辑在这里)
            global_orient, body_pose, betas, has_smpl = self._get_smpl_params(data, f_idx)

            # 2. 获取 3D 关节 (World Coordinate)
            if 'jointPositions' not in data:
                raise ValueError("No jointPositions found")
            gt_world = data['jointPositions'][0, f_idx].reshape(-1, 3)[:24] # 只取前24个

            # 3. 相机参数
            cam_ext = data['extrinsics'][f_idx] # Extrinsics
            cam_int = data['intrinsics']        # Intrinsics

        # --- C. 几何投影 (World -> Camera -> Image) ---
        # World -> Camera
        ones = np.ones((gt_world.shape[0], 1))
        gt_cam = (np.concatenate([gt_world, ones], axis=1) @ cam_ext.T)[:, :3]

        # Camera -> Image (2D Projection)
        z = np.maximum(gt_cam[:, 2], 0.1) # 避免除以 0
        u = cam_int[0, 0] * (gt_cam[:, 0] / z) + cam_int[0, 2]
        v = cam_int[1, 1] * (gt_cam[:, 1] / z) + cam_int[1, 2]
        keypoints_2d = np.stack([u, v, np.ones_like(u)], axis=1) # [24, 3]

        # --- D. 裁剪 (Cropping) ---
        # 计算 Bbox
        xmin, ymin = np.min(keypoints_2d[:, 0]), np.min(keypoints_2d[:, 1])
        xmax, ymax = np.max(keypoints_2d[:, 0]), np.max(keypoints_2d[:, 1])
        center, scale = self._box_to_center_scale([xmin, ymin, xmax-xmin, ymax-ymin], img_w, img_h)
        
        # 执行裁剪
        img_crop = self._crop_image(img_cv2, center, scale, self.img_size)
        img_tensor = self.transform(img_crop)

        # --- E. 最终数据打包 (Padding & Formatting) ---
        
        # Transform keypoints to crop coordinates
        keypoints_2d_crop = self._transform_keypoints(keypoints_2d, center, scale, self.img_size)

        # 1. 填充 2D 关键点到 44 个
        kp_2d_padded = np.zeros((TOTAL_JOINTS, 3), dtype=np.float32)
        kp_2d_padded[:24] = keypoints_2d_crop
        
        # 2. 填充 3D 关键点 (Root-Relative)
        kp_3d_padded = np.zeros((TOTAL_JOINTS, 4), dtype=np.float32)
        root_joint = gt_cam[0:1] # Root (Pelvis)
        gt_cam_relative = gt_cam - root_joint
        
        kp_3d_padded[:24, :3] = gt_cam_relative
        kp_3d_padded[:24, 3] = 1.0 # Confidence = 1
        
        # 3. Add Pelvis to index 39 (HMR2 Evaluator Expectation)
        kp_3d_padded[39, :3] = gt_cam_relative[0] # Joint 0 is Pelvis
        kp_3d_padded[39, 3] = 1.0
        
        # Also map 2D
        kp_2d_padded[39] = keypoints_2d_crop[0]

        # 3. 构造返回字典
        return {
            'img': img_tensor,
            'keypoints_2d': torch.from_numpy(kp_2d_padded).float(),
            'keypoints_3d': torch.from_numpy(kp_3d_padded).float(),
            'smpl_params': {
                'global_orient': global_orient,
                'body_pose': body_pose,
                'betas': betas
            },
            # 告诉 Loss 函数：如果是假数据，就不要算 SMPL Loss
            'has_smpl_params': {
                'global_orient': torch.tensor(has_smpl),
                'body_pose': torch.tensor(has_smpl),
                'betas': torch.tensor(has_smpl)
            },
            'smpl_params_is_axis_angle': {
                'global_orient': torch.tensor(True),
                'body_pose': torch.tensor(True),
                'betas': torch.tensor(False)
            }
        }

    # --- Utils (封装在类内部作为静态方法) ---
    @staticmethod
    def _box_to_center_scale(box, img_width, img_height):
        x, y, w, h = box
        center = np.array([x + w / 2, y + h / 2])
        aspect_ratio = 1.0
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = h / 200.0 * 1.2
        return center, scale

    @staticmethod
    def _crop_image(img, center, scale, crop_size):
        scale_px = scale * 200.0
        ul = (center - scale_px / 2).astype(int)
        br = (center + scale_px / 2).astype(int)
        crop_h, crop_w = int(br[1] - ul[1]), int(br[0] - ul[0])
        
        new_img = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        
        ul_img = [max(0, ul[0]), max(0, ul[1])]
        br_img = [min(img.shape[1], br[0]), min(img.shape[0], br[1])]
        ul_new = [ul_img[0] - ul[0], ul_img[1] - ul[1]]
        
        img_w_valid = br_img[0] - ul_img[0]
        img_h_valid = br_img[1] - ul_img[1]
        
        if img_w_valid > 0 and img_h_valid > 0:
            new_img[ul_new[1]:ul_new[1]+img_h_valid, ul_new[0]:ul_new[0]+img_w_valid] = \
                img[ul_img[1]:br_img[1], ul_img[0]:br_img[0]]
                
        return cv2.resize(new_img, crop_size)

    @staticmethod
    def _transform_keypoints(keypoints, center, scale, crop_size):
        # keypoints: (N, 3) or (N, 2)
        scale_px = scale * 200.0
        ul = (center - scale_px / 2)
        
        # (x - ul_x) * (crop_w / scale_px)
        # Assuming square crop for simplicity as is standard in HMR
        ratio = crop_size[0] / scale_px
        
        new_kp = keypoints.copy()
        new_kp[:, 0] = (keypoints[:, 0] - ul[0]) * ratio
        new_kp[:, 1] = (keypoints[:, 1] - ul[1]) * ratio
        
        return new_kp

def create_dataset(args, split='train'):
    dataset = ThreeDPWDataset(config=args, split=split)
    return dataset