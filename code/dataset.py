# dataset.py
"""
AD诊断数据集
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
from scipy.ndimage import zoom

class ADDiagnosisDataset(Dataset):
    """
    阿尔茨海默病诊断数据集
    
    支持多数据集（ADNI, PUSH, SMHC）
    """
    
    def __init__(self, manifest_path, split='train', 
                 num_slices=64, transform=None,
                 dataset_name='ADNI'):
        """
        Args:
            manifest_path: manifest.csv路径，包含以下列：
                - subject_id: 被试ID
                - t1w_path: T1w图像路径
                - mind_path: MIND矩阵路径
                - label: 标签 (0=NC, 1=MCI, 2=AD)
                - age: 年龄
                - sex: 性别 (0=F, 1=M)
                - education: 教育年限
                - apoe4: APOE4基因型
                - ... 其他人口学变量
            split: 'train', 'val', or 'test'
            num_slices: 切片数量
            transform: 数据增强
            dataset_name: 数据集名称
        """
        self.manifest = pd.read_csv(manifest_path)
        self.manifest = self.manifest[self.manifest['split'] == split]
        self.split = split
        self.num_slices = num_slices
        self.transform = transform
        self.dataset_name = dataset_name
        
        # 加载覆盖矩阵
        self.cover_ctx = np.load('./data/cover_matrix_ctx.npy')  # [S, 360]
        self.cover_sub = np.load('./data/cover_matrix_sub.npy')  # [S, 66]
        
        print(f"Loaded {len(self.manifest)} samples from {dataset_name} ({split})")
        print(f"  NC: {(self.manifest['label']==0).sum()}")
        print(f"  MCI: {(self.manifest['label']==1).sum()}")
        print(f"  AD: {(self.manifest['label']==2).sum()}")
        
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # 1. 加载T1w图像
        t1w_path = row['t1w_path']
        slices = self._load_slices(t1w_path)  # [S, C, H, W]
        
        # ✓ 确保是 float32 类型
        slices = slices.astype(np.float32)

        # 数据增强
        if self.transform and self.split == 'train':
            slices = self.transform(slices)
        
        # 2. 加载MIND矩阵
        mind_path = row['mind_path']
        #mind = np.load(mind_path).astype(np.float32)  # [360, 360]
        mind_df = pd.read_csv(mind_path, index_col=0)
        mind = mind_df.values.astype(np.float32)  # 转换为numpy数组 [360, 360]
        # ✓ 确保是 float32 类型
        mind = mind.astype(np.float32)
        
        # 3. 加载人口学信息
        demographics = self._extract_demographics(row)
        
        # 4. 标签
        label = int(row['label'])
        
        return {
            'slices': torch.from_numpy(slices),
            'mind': torch.from_numpy(mind),
            'demographics': torch.from_numpy(demographics),
            'label': label,
            'text_class': label,  # 用于索引文本描述
            'cover_ctx': torch.from_numpy(self.cover_ctx).float(),
            'cover_sub': torch.from_numpy(self.cover_sub).float(),
            'subject_id': row['subject_id'],
            'dataset': self.dataset_name
        }
    
    def _load_slices(self, t1w_path):
        """
        加载并预处理T1w图像，提取轴向切片
        
        Args:
            t1w_path: NIfTI图像路径
            
        Returns:
            slices: [S, 1, H, W] - 轴向切片
        """
        # 加载NIfTI
        nii = nib.load(t1w_path)
        img = nii.get_fdata()  # [X, Y, Z]
        
        # 标准化方向（RAS）
        # 假设已经通过HCP pipeline处理过
        
        # 提取轴向切片（假设Z轴是上下方向）
        Z = img.shape[2]
        
        # 选择中间的num_slices个切片
        start_z = (Z - self.num_slices) // 2
        end_z = start_z + self.num_slices
        
        slices = img[:, :, start_z:end_z]  # [X, Y, S]
        slices = slices.transpose(2, 0, 1)  # [S, X, Y]
        
        # 调整大小到固定尺寸（例如224x224）
        target_size = (224, 224)
        slices_resized = []
        for s in range(slices.shape[0]):
            slice_2d = slices[s]
            # 计算缩放因子
            zoom_factors = [target_size[0] / slice_2d.shape[0],
                          target_size[1] / slice_2d.shape[1]]
            slice_resized = zoom(slice_2d, zoom_factors, order=1)
            slices_resized.append(slice_resized)
        
        slices_resized = np.stack(slices_resized, axis=0)  # [S, H, W]
        
        # 归一化到[0, 1]
        slices_resized = (slices_resized - slices_resized.min()) / \
                        (slices_resized.max() - slices_resized.min() + 1e-8)
        
        # 添加通道维度
        slices_resized = slices_resized[:, np.newaxis, :, :]  # [S, 1, H, W]
        
        return slices_resized.astype(np.float32)
    
    def _extract_demographics(self, row):
        """
        提取并标准化人口学特征
        
        特征：
        - age: 年龄（标准化）
        - sex: 性别（0=女, 1=男）
        - edu_years: 教育年限（标准化）
        - race: 种族（one-hot编码，7类）
        """
        # 1. 年龄（标准化：均值70，std 10）
        age = float(row.get('age', 70.0))
        age_norm = (age - 70.0) / 10.0
        
        # 2. 性别（已编码：0=女, 1=男）
        sex = float(row.get('sex', 0))
        
        # 3. 教育年限（标准化：均值12，std 4）
        edu_years = float(row.get('edu_years', 12.0))
        edu_norm = (edu_years - 12.0) / 4.0
        
        # 4. 种族（one-hot编码，7类）
        race = int(row.get('race', 1))  # 默认为1
        race_onehot = np.zeros(7, dtype=np.float32)
        if 1 <= race <= 7:
            race_onehot[race - 1] = 1.0
        
        # 5. 年龄分组（额外特征）
        # 根据AD风险年龄段划分
        age_group = np.zeros(3, dtype=np.float32)
        if age < 65:
            age_group[0] = 1.0  # 年轻组
        elif age < 75:
            age_group[1] = 1.0  # 中年组
        else:
            age_group[2] = 1.0  # 老年组
        
        # 拼接所有特征
        demographics = np.concatenate([
            [age_norm],           # 1维
            [sex],                # 1维
            [edu_norm],           # 1维
            race_onehot,          # 7维
            age_group,            # 3维
        ])  # 总共13维
        
        return demographics.astype(np.float32)


class CustomTransform:
    """数据增强"""
    
    def __init__(self, split='train'):
        self.split = split
        
    def __call__(self, slices):
        """
        Args:
            slices: [S, 1, H, W]
            
        Returns:
            augmented: [S, 1, H, W]
        """
        if self.split != 'train':
            return slices
        
        # 随机水平翻转
        if np.random.rand() > 0.5:
            slices = slices[:, :, :, ::-1].copy()
        
        # 随机亮度调整
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.9, 1.1)
            slices = np.clip(slices * factor, 0, 1)
        
        # 随机对比度调整
        if np.random.rand() > 0.5:
            mean = slices.mean()
            factor = np.random.uniform(0.9, 1.1)
            slices = np.clip((slices - mean) * factor + mean, 0, 1)
        
        # 随机高斯噪声
        if np.random.rand() > 0.5:
            noise = np.random.randn(*slices.shape) * 0.01
            slices = np.clip(slices + noise, 0, 1)
        
        return slices
