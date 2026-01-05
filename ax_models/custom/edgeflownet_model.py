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


class OpticalFlowDataAdapter:
    """
    光流校准数据集适配器
    支持 FlyingThings3D 目录结构:
    data_dir/
    ├── 0000/left/*.png
    ├── 0001/left/*.png
    └── ...
    """
    
    def __init__(self, repr_imgs_dir_path: str, repr_imgs_dataloader_color_format: str = 'RGB'):
        self.data_dir = Path(repr_imgs_dir_path)
        self.color_format = repr_imgs_dataloader_color_format
        self.input_height = 540  # 输入高度
        self.input_width = 960   # 输入宽度
    
    def _find_all_sequences(self):
        """
        查找所有序列文件夹
        返回每个序列中排序后的帧列表
        """
        sequences = []
        
        # 遍历子文件夹 (0000, 0001, ...)
        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            
            # 查找 left 文件夹
            left_dir = subdir / 'left'
            if left_dir.exists():
                # 获取所有 PNG 文件并排序
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
            frames = sorted(self.data_dir.glob('*.png'))
            if len(frames) >= 2:
                sequences.append(frames)
        
        return sequences
    
    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        生成校准数据
        返回: [1, H, W, 6] 形状的数组
        """
        # 查找所有序列
        sequences = self._find_all_sequences()
        
        if not sequences:
            # 如果没有找到帧，生成随机数据
            print(f"警告: 未找到校准图片，使用随机数据")
            for _ in range(100):
                yield np.random.uniform(0, 1, (1, self.input_height, self.input_width, 6)).astype(np.float32)
            return
        
        # 遍历每个序列中的连续帧对
        for frames in sequences:
            for i in range(len(frames) - 1):
                # 读取连续两帧
                frame1_path = frames[i]
                frame2_path = frames[i + 1]
                
                frame1 = cv2.imread(str(frame1_path))
                frame2 = cv2.imread(str(frame2_path))
                
                if frame1 is None or frame2 is None:
                    continue
                
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
                
                # 拼接为6通道
                combined = np.concatenate([frame1, frame2], axis=-1)
                
                # 添加 batch 维度
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
