#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EdgeFlowNet 自定义模型类
用于 Axelera SDK 的光流模型部署
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Generator

# Axelera SDK 导入
try:
    from ax_models import base_onnx
    from axelera import types
    from axelera.app import logging_utils
except ImportError:
    # 本地测试时跳过
    pass


class EdgeFlowNetModel(base_onnx.AxONNXModel):
    """
    EdgeFlowNet 光流模型
    输入: 两帧 RGB 图像拼接 [H, W, 6]
    输出: 光流场 [H, W, 2] (u, v)
    """
    
    # 类属性（不需要在 __init__ 中初始化）
    prev_frame = None
    input_height = 540
    input_width = 960
    
    def init_model_deploy(self, model_info, dataset_config, **kwargs):
        """
        SDK 在实例化后调用此方法
        在这里初始化自定义属性
        """
        # 调用父类方法加载 ONNX 模型
        super().init_model_deploy(model_info, dataset_config, **kwargs)
        # 初始化帧缓存
        self.prev_frame = None
        self.input_height = 540
        self.input_width = 960
    
    def override_preprocess(self, img) -> np.ndarray:
        """
        预处理函数
        将当前帧与前一帧拼接为6通道输入
        支持 PIL Image 或 numpy 数组输入
        
        注意：返回 Rank 3 tensor [H, W, 6]，不带 batch 维度！
        SDK 会自动添加 batch 维度。
        """
        # 将 PIL Image 转换为 numpy 数组
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # 确保图像是 numpy 数组（兼容 torch tensor）
        if hasattr(img, 'numpy'):
            img = img.numpy()
        
        # 如果是 RGBA，转换为 RGB
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]
        
        # 调整图像大小
        img_resized = cv2.resize(img, (self.input_width, self.input_height))
        
        # 归一化到 0-1 范围
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 如果没有前一帧，使用当前帧代替
        if self.prev_frame is None:
            self.prev_frame = img_normalized.copy()
        
        # 拼接两帧为6通道 [H, W, 6]
        combined = np.concatenate([self.prev_frame, img_normalized], axis=-1)
        
        # 更新前一帧缓存
        self.prev_frame = img_normalized.copy()
        
        # 转换为 torch.Tensor，保持 NHWC 格式 [H, W, 6]
        # 不添加 batch 维度，SDK 会自动添加
        import torch
        tensor = torch.from_numpy(combined)
        
        # 返回 Rank 3 tensor [H, W, 6]
        # 模型 input_tensor_layout 是 NHWC，SDK 会处理
        return tensor
    
    def reset_frame_buffer(self):
        """重置帧缓存"""
        self.prev_frame = None


class OpticalFlowDataAdapter(types.DataAdapter):
    """
    光流校准数据集适配器
    支持 FlyingThings3D 目录结构:
    data_dir/
    ├── 0000/left/*.png
    ├── 0001/left/*.png
    └── ...
    """
    
    def __init__(self, dataset_config, model_info):
        """
        SDK 要求的构造函数签名
        """
        self.dataset_config = dataset_config
        self.model_info = model_info
        self.input_height = 540
        self.input_width = 960
        # 从配置中获取路径（支持多种配置方式）
        self.calib_data_path = dataset_config.get('calib_data_path', '')
        self.data_dir_path = dataset_config.get('data_dir_path', '')
        self.repr_imgs_dir_path = dataset_config.get('repr_imgs_dir_path', '')
        self.color_format = dataset_config.get('repr_imgs_dataloader_color_format', 'RGB')
    
    def _find_all_sequences(self, data_dir):
        """
        查找所有序列文件夹
        返回每个序列中排序后的帧列表
        """
        sequences = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            return sequences
        
        # 遍历子文件夹 (0000, 0001, ...)
        for subdir in sorted(data_path.iterdir()):
            if not subdir.is_dir():
                continue
            
            # 查找 left 文件夹
            left_dir = subdir / 'left'
            if left_dir.exists():
                frames = sorted(left_dir.glob('*.png'))
                if len(frames) >= 2:
                    sequences.append(frames)
            else:
                # 如果没有 left 子文件夹，直接查找 PNG
                frames = sorted(subdir.glob('*.png'))
                if len(frames) >= 2:
                    sequences.append(frames)
        
        # 如果没有子文件夹，尝试直接在根目录查找
        if not sequences:
            frames = sorted(data_path.glob('*.png'))
            if len(frames) >= 2:
                sequences.append(frames)
        
        return sequences
    
    def _generate_frame_pairs(self, data_dir):
        """
        生成所有帧对的列表
        """
        frame_pairs = []
        sequences = self._find_all_sequences(data_dir)
        
        for frames in sequences:
            for i in range(len(frames) - 1):
                frame_pairs.append((frames[i], frames[i + 1]))
        
        return frame_pairs
    
    def _load_and_process_pair(self, frame1_path, frame2_path):
        """
        加载并处理一对帧，返回 6 通道拼接
        """
        frame1 = cv2.imread(str(frame1_path))
        frame2 = cv2.imread(str(frame2_path))
        
        if frame1 is None or frame2 is None:
            return None
        
        # BGR → RGB
        if self.color_format == 'RGB':
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        frame1 = cv2.resize(frame1, (self.input_width, self.input_height))
        frame2 = cv2.resize(frame2, (self.input_width, self.input_height))
        
        # 归一化
        frame1 = frame1.astype(np.float32) / 255.0
        frame2 = frame2.astype(np.float32) / 255.0
        
        # 拼接为 6 通道
        combined = np.concatenate([frame1, frame2], axis=-1)
        
        return combined
    
    def __iter__(self):
        """
        支持直接迭代（用于校准）
        """
        # 优先使用 calib_data_path，然后 data_dir_path，最后 repr_imgs_dir_path
        data_dir = self.calib_data_path or self.data_dir_path or self.repr_imgs_dir_path
        
        # 如果是相对路径，尝试相对于 AXELERA_FRAMEWORK
        if data_dir and not Path(data_dir).is_absolute():
            import os
            framework_path = os.environ.get('AXELERA_FRAMEWORK', '')
            if framework_path:
                data_dir = Path(framework_path) / data_dir
        
        frame_pairs = self._generate_frame_pairs(data_dir)
        
        if not frame_pairs:
            print(f"警告: 未找到校准图片于 {data_dir}，使用随机数据")
            for _ in range(100):
                yield np.random.uniform(0, 1, (1, self.input_height, self.input_width, 6)).astype(np.float32)
            return
        
        print(f"找到 {len(frame_pairs)} 对校准帧对")
        for frame1_path, frame2_path in frame_pairs:
            combined = self._load_and_process_pair(frame1_path, frame2_path)
            if combined is not None:
                # 返回 numpy 数组，不需要 batch 维度，DataLoader 会处理
                yield combined

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        """
        创建校准数据加载器
        覆盖 SDK 默认行为，使用自定义逻辑加载 6 通道数据
        """
        import torch
        
        class CalibrationIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self, adapter):
                self.adapter = adapter
            
            def __iter__(self):
                # 适配器产生 numpy [H, W, 6]
                for combined in self.adapter:
                    # 转换为 Tensor [H, W, 6] (NHWC)，保持模型输入格式
                    # 不要进行 permute，因为模型 layout 是 NHWC
                    tensor = torch.from_numpy(combined)
                    print(f"DEBUG: Yielding tensor shape: {tensor.shape}")
                    yield tensor
        
        # 使用 batch_size=1，因为我们的数据已经是成对的
        # 使用 batch_size=None 禁用自动 batching，直接返回 dataset yield 的 tensor
        # 这样避免 DataLoader 自动添加 batch 维度导致 Rank 5 问题
        return torch.utils.data.DataLoader(
            CalibrationIterableDataset(self),
            batch_size=batch_size or 1,
            num_workers=0,
            collate_fn=lambda x: torch.stack(x)
        )


def flow_to_color(flow: np.ndarray, max_flow: float = None) -> np.ndarray:
    """
    将光流可视化为颜色图
    使用 HSV 色轮: 色相表示方向，亮度表示幅度
    
    参数:
        flow: [H, W, 2] 光流场 (u, v)
        max_flow: 最大流幅度，用于归一化
    
    返回:
        [H, W, 3] RGB 颜色图
    """
    # 获取 u, v 分量
    u = flow[..., 0]
    v = flow[..., 1]
    
    # 计算幅度和角度
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    
    # 归一化幅度
    if max_flow is None:
        max_flow = np.max(magnitude) + 1e-6
    magnitude = np.clip(magnitude / max_flow, 0, 1)
    
    # 将角度转换为色相 (0-180 for OpenCV)
    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    
    # 饱和度固定为 255
    saturation = np.ones_like(hue) * 255
    
    # 亮度 = 幅度
    value = (magnitude * 255).astype(np.uint8)
    
    # 组合 HSV 并转换为 RGB
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb
