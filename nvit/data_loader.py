#!/home/yangz/.conda/envs/4D-humans/bin/python
"""
数据加载模块
目标：加载和管理3DPW数据集
"""

import os
import json
import pickle
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path

class DataLoader:
    """
    数据加载器 - 统一的数据加载接口
    
    职责：
    - 加载3DPW数据集
    - 提供校准和测试数据接口
    - 支持量化器和剪枝器的数据需求
    """    
    def __init__(self, config=None, model_cfg=None):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典
            model_cfg: 模型配置 (用于ViTDetDataset预处理)
        """
        data_root = config.get('data_root', '/home/yangz/4D-Humans/data') if config else '/home/yangz/4D-Humans/data'
        self.data_root = Path(data_root)
        self.model_cfg = model_cfg
        
        # 3DPW数据结构
        self.sequence_dir = self.data_root / "3DPW" / "sequenceFiles" / "test"
        self.image_dir = self.data_root / "3DPW" / "imageFiles"
        
        # 数据列表
        self.data_list = []
        self.samples = []
        
        # 统计信息
        self.num_sequences = 0
        self.num_frames = 0
        self.num_persons = 0
        
        print(f"✅ 数据加载器初始化完成")
    
    def _process_image(self, sample):
        """预处理单张图片"""
        if self.model_cfg is None:
            raise ValueError("Data loader needs model_cfg to process images!")
            
        import cv2
        import numpy as np
        from hmr2.datasets.vitdet_dataset import ViTDetDataset
        
        img_path = sample['image_path']
        img_cv2 = cv2.imread(img_path)
        
        if img_cv2 is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        # 使用全图框 (假设单人或主要人物)
        h, w = img_cv2.shape[:2]
        bbox = np.array([[0, 0, w, h]])
        
        # 使用ViTDetDataset处理
        dataset = ViTDetDataset(self.model_cfg, img_cv2, bbox)
        batch = dataset[0] # 获取第一个(也是唯一的)样本
        
        # 转换为Tensor并添加Batch维度
        processed_batch = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            processed_batch[k] = v.unsqueeze(0) # [1, ...]
            
        return processed_batch

    def load_samples(self, num_samples=100, split='test'):
        """加载数据样本 (仅元数据)"""
        print(f"🔄 开始加载数据集: {split}")
        self.sequence_dir = self.data_root / "3DPW"/ "sequenceFiles" / split
        npz_files = list(self.sequence_dir.glob("*.npz"))

        for npz_file in npz_files:
            try:
                sequence_name = npz_file.stem
                npz_path = self.sequence_dir / f"{sequence_name}.npz"
                data = np.load(npz_path)
                
                try:
                    seq_name = str(data['seq_name'])
                except KeyError:
                    seq_name = sequence_name
                    
                num_frames = data['jointPositions'].shape[1]
                
                for frame_idx in range(num_frames):
                    img_name = f"image_{frame_idx:05d}.jpg"
                    img_path = self.image_dir / seq_name / img_name                
                    
                    if not img_path.exists() or img_path.stat().st_size == 0:
                        continue
                    
                    sample = {
                        'npz_path': str(npz_path),
                        'seq_name': seq_name,
                        'frame_idx': frame_idx,
                        'image_path': str(img_path),
                    }
                    self.data_list.append(sample)
                
                data.close()
            except Exception as e:
                print(f"⚠️ 跳过文件 {npz_file}: {e}")
                continue
            
        import random
        print(f"找到 {len(self.data_list)} 个有效样本")
        # 随机采样
        if len(self.data_list) > 0:
            self.samples = random.sample(self.data_list, min(num_samples, len(self.data_list)))
        return self.samples
    
    def get_calibration_data(self, num_samples=10):
        """获取校准数据 (处理后的Tensor)"""
        if not self.samples:
            self.load_samples(num_samples=num_samples)
            
        calib_data = []
        count = 0
        for sample in self.samples:
            if count >= num_samples: break
            try:
                processed = self._process_image(sample)
                calib_data.append(processed)
                count += 1
            except Exception as e:
                print(f"⚠️ 处理样本失败: {e}")
                continue
                
        return calib_data
    
    def get_test_samples(self, num_samples=5):
        """获取测试数据 (元数据)"""
        if len(self.samples) < num_samples:
            self.load_samples(num_samples=num_samples + 5)
        return self.samples[-num_samples:]

# dataLoader = DataLoader(config={'data_root': '/home/yangz/4D-Humans/data/3DPW'})
# dataLoader.load_samples(num_samples=100)
