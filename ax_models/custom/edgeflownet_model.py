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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_frame = None  # 缓存前一帧
        self.input_height = 540  # 输入高度
        self.input_width = 960   # 输入宽度
    
    def override_preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        预处理函数
        将当前帧与前一帧拼接为6通道输入
        """
        # 确保图像是 numpy 数组
        if hasattr(img, 'numpy'):
            img = img.numpy()
        
        # 调整图像大小
        img_resized = cv2.resize(img, (self.input_width, self.input_height))
        
        # 归一化到 0-1 范围
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 如果没有前一帧，使用当前帧代替
        if self.prev_frame is None:
            self.prev_frame = img_normalized.copy()
        
        # 拼接两帧为6通道
        combined = np.concatenate([self.prev_frame, img_normalized], axis=-1)
        
        # 更新前一帧缓存
        self.prev_frame = img_normalized.copy()
        
        # 添加 batch 维度
        combined = np.expand_dims(combined, axis=0)
        
        return combined
    
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
        # 从配置中获取路径
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
        data_dir = self.repr_imgs_dir_path
        frame_pairs = self._generate_frame_pairs(data_dir)
        
        if not frame_pairs:
            print(f"警告: 未找到校准图片于 {data_dir}，使用随机数据")
            for _ in range(100):
                yield np.random.uniform(0, 1, (1, self.input_height, self.input_width, 6)).astype(np.float32)
            return
        
        for frame1_path, frame2_path in frame_pairs:
            combined = self._load_and_process_pair(frame1_path, frame2_path)
            if combined is not None:
                yield np.expand_dims(combined, axis=0)


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
