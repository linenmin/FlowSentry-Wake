#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
校准数据预处理脚本：将帧对水平拼接为单张图片

用于 Axelera SDK 的校准数据准备。
将 FlyingThings3D 等数据集中的连续帧对拼接成 [H, 2W, 3] 的 RGB 图片。
SDK 的 ImageReader 可以正常读取这些图片，然后在 override_preprocess 中切分。

使用方法:
    python prepare_calib_data.py --input data/calib_edgeflownet --output data/calib_merged

输入目录结构:
    input/
    ├── 0000/left/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    └── 0001/left/
        └── ...

输出目录结构:
    output/
    ├── merged_0000_0000.png  # 576 x 2048 x 3
    ├── merged_0000_0001.png
    └── ...
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse

# 目标分辨率
TARGET_HEIGHT = 576
TARGET_WIDTH = 1024


def find_frame_pairs(input_dir):
    """
    在输入目录中查找帧对
    返回 [(frame1_path, frame2_path), ...]
    """
    input_path = Path(input_dir)
    pairs = []
    
    # 查找所有子目录
    for subdir in sorted(input_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        # 查找 left 子目录
        left_dir = subdir / 'left'
        if not left_dir.exists():
            left_dir = subdir  # 直接使用子目录
        
        # 获取所有图片
        images = sorted([f for f in left_dir.iterdir() 
                        if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        # 生成帧对 (连续两帧)
        for i in range(len(images) - 1):
            pairs.append((images[i], images[i + 1]))
    
    # 如果没有子目录，直接在输入目录中查找
    if not pairs:
        images = sorted([f for f in input_path.iterdir() 
                        if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        for i in range(len(images) - 1):
            pairs.append((images[i], images[i + 1]))
    
    return pairs


def merge_frame_pair(frame1_path, frame2_path, target_h=TARGET_HEIGHT, target_w=TARGET_WIDTH):
    """
    将两帧图片水平拼接为单张图片
    输出: [H, 2W, 3] 的 RGB 图片
    """
    # 读取图片 (BGR)
    img1 = cv2.imread(str(frame1_path))
    img2 = cv2.imread(str(frame2_path))
    
    if img1 is None or img2 is None:
        print(f"警告: 无法读取 {frame1_path} 或 {frame2_path}")
        return None
    
    # Resize 到目标分辨率
    img1 = cv2.resize(img1, (target_w, target_h))
    img2 = cv2.resize(img2, (target_w, target_h))
    
    # 转换为 RGB (SDK 通常使用 RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # 水平拼接 [H, 2W, 3]
    merged = np.hstack((img1, img2))
    
    return merged


def main():
    parser = argparse.ArgumentParser(description='准备 EdgeFlowNet 校准数据')
    parser.add_argument('--input', '-i', default='data/calib_edgeflownet',
                       help='输入目录 (包含帧序列)')
    parser.add_argument('--output', '-o', default='data/calib_merged',
                       help='输出目录 (拼接后的图片)')
    parser.add_argument('--max-pairs', '-n', type=int, default=200,
                       help='最大帧对数量 (默认 200)')
    parser.add_argument('--height', type=int, default=TARGET_HEIGHT,
                       help=f'目标高度 (默认 {TARGET_HEIGHT})')
    parser.add_argument('--width', type=int, default=TARGET_WIDTH,
                       help=f'目标宽度 (默认 {TARGET_WIDTH})')
    args = parser.parse_args()
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"目标分辨率: {args.width}x{args.height}")
    print(f"最大帧对数: {args.max_pairs}")
    print("-" * 50)
    
    # 查找帧对
    pairs = find_frame_pairs(args.input)
    print(f"找到 {len(pairs)} 个帧对")
    
    if not pairs:
        print("错误: 未找到任何帧对!")
        return
    
    # 限制数量
    pairs = pairs[:args.max_pairs]
    
    # 处理每个帧对
    count = 0
    for i, (frame1, frame2) in enumerate(pairs):
        merged = merge_frame_pair(frame1, frame2, args.height, args.width)
        if merged is None:
            continue
        
        # 保存 (转回 BGR 给 OpenCV)
        output_file = output_path / f"merged_{i:04d}.png"
        merged_bgr = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_file), merged_bgr)
        count += 1
        
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(pairs)} 个帧对...")
    
    print("-" * 50)
    print(f"完成! 共生成 {count} 张拼接图片")
    print(f"输出目录: {output_path.absolute()}")
    print(f"图片尺寸: {args.width * 2}x{args.height} (W x H)")


if __name__ == '__main__':
    main()
