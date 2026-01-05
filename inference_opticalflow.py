#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EdgeFlowNet RTSP 光流推理脚本
基于 Axelera SDK 的实时光流估计

用法:
    ./inference_opticalflow.py edgeflownet-opticalflow "rtsp://user:pass@ip:port/stream"
"""

import os
import sys
import time
import cv2
import numpy as np

# 检查 Axelera 环境
if not os.environ.get('AXELERA_FRAMEWORK'):
    sys.exit("请激活 Axelera 环境: source venv/bin/activate")

from tqdm import tqdm
from axelera.app import (
    config,
    create_inference_stream,
    display,
    inf_tracers,
    logging_utils,
    statistics,
    yaml_parser,
)

# 日志配置
LOG = logging_utils.getLogger(__name__)
PBAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

# Logo 路径
LOGO1 = os.path.join(config.env.framework, "axelera/app/voyager-sdk-logo-white.png")
LOGO2 = os.path.join(config.env.framework, "axelera/app/axelera-ai-logo.png")
LOGO_POS = '95%, 95%'


def flow_to_color(flow: np.ndarray, max_flow: float = 50.0) -> np.ndarray:
    """
    将光流可视化为颜色图
    色相 = 方向，亮度 = 幅度
    """
    # 获取 u, v 分量
    u = flow[..., 0]
    v = flow[..., 1]
    
    # 计算幅度和角度
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)
    
    # 归一化幅度
    magnitude = np.clip(magnitude / max_flow, 0, 1)
    
    # 转换为 HSV
    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    saturation = np.ones_like(hue, dtype=np.uint8) * 255
    value = (magnitude * 255).astype(np.uint8)
    
    # HSV → BGR
    hsv = np.stack([hue, saturation, value], axis=-1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr


class OpticalFlowProcessor:
    """
    光流处理器
    管理帧缓存和光流可视化
    """
    
    def __init__(self, width: int = 960, height: int = 540):
        self.width = width        # 目标宽度
        self.height = height      # 目标高度
        self.prev_frame = None    # 前一帧缓存
    
    def prepare_input(self, frame: np.ndarray) -> np.ndarray:
        """
        准备模型输入
        将当前帧与前一帧拼接为6通道
        """
        # 调整大小
        frame_resized = cv2.resize(frame, (self.width, self.height))
        
        # 归一化
        frame_norm = frame_resized.astype(np.float32) / 255.0
        
        # 处理第一帧
        if self.prev_frame is None:
            self.prev_frame = frame_norm.copy()
        
        # 拼接为6通道
        combined = np.concatenate([self.prev_frame, frame_norm], axis=-1)
        
        # 更新缓存
        self.prev_frame = frame_norm.copy()
        
        return combined
    
    def visualize(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        可视化光流结果
        左: 原始帧, 右: 光流颜色图
        """
        # 调整原始帧大小
        frame_resized = cv2.resize(frame, (self.width, self.height))
        
        # 光流转颜色
        flow_color = flow_to_color(flow)
        
        # 水平拼接
        combined = np.hstack([frame_resized, flow_color])
        
        return combined
    
    def reset(self):
        """重置帧缓存"""
        self.prev_frame = None


def inference_loop(args, log_file_path, stream, app, wnd, tracers=None):
    """
    主推理循环
    """
    # 创建光流处理器
    processor = OpticalFlowProcessor(width=960, height=540)
    
    # 配置窗口
    if len(stream.sources) > 1:
        for sid, source in stream.sources.items():
            wnd.options(sid, title=f"#{sid} - {source}")
    
    wnd.options(-1, speedometer_smoothing=args.speedometer_smoothing)
    
    # 显示 Logo
    logo1 = wnd.image(LOGO_POS, LOGO1, anchor_x='right', anchor_y='bottom', scale=0.3)
    logo2 = wnd.image(LOGO_POS, LOGO2, anchor_x='right', anchor_y='bottom', scale=0.3, fadeout_from=0.0)
    supported = logo1 and logo2
    logo_start = time.time()
    logo_period = 10.0
    
    # 推理循环
    for event in tqdm(
        stream.with_events(),
        desc="光流推理...",
        unit='frames',
        leave=False,
        bar_format=PBAR,
        disable=None,
    ):
        # 检查事件
        if not event.result:
            LOG.warning(f"未知事件: {event!r}")
            continue
        
        frame_result = event.result
        
        # Logo 切换
        now = time.time()
        if supported and ((now - logo_start) > logo_period):
            logo1.hide(now, 1.0)
            logo2.show(now, 1.0)
            logo1, logo2 = logo2, logo1
            logo_start = now
        
        # 获取图像和元数据
        image, meta = frame_result.image, frame_result.meta
        
        if image is None and meta is None:
            if wnd.is_closed:
                break
            continue
        
        # 显示结果
        if image is not None:
            wnd.show(image, meta, frame_result.stream_id)
        
        # 检查窗口关闭
        if wnd.is_closed:
            break
    
    # 显示统计
    if log_file_path:
        print(statistics.format_table(log_file_path, tracers))
    inf_tracers.display_tracers(tracers)


if __name__ == "__main__":
    # 解析网络配置
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_inference_argparser(
        network_yaml_info, 
        description='EdgeFlowNet 光流推理 (Axelera M.2)'
    )
    args = parser.parse_args()
    
    # 创建 tracers
    tracers = inf_tracers.create_tracers_from_args(args)
    
    try:
        # 初始化日志
        log_file, log_file_path = None, None
        if args.show_stats:
            log_file, log_file_path = statistics.initialise_logging()
        
        # 创建推理流
        stream = create_inference_stream(
            config.SystemConfig.from_parsed_args(args),
            config.InferenceStreamConfig.from_parsed_args(args),
            config.PipelineConfig.from_parsed_args(args),
            config.LoggingConfig.from_parsed_args(args),
            config.DeployConfig.from_parsed_args(args),
            tracers=tracers,
        )
        
        # 创建显示窗口
        with display.App(
            renderer=args.display,
            opengl=stream.hardware_caps.opengl,
            buffering=not stream.is_single_image(),
        ) as app:
            wnd = app.create_window('EdgeFlowNet 光流推理', size=args.window_size)
            
            # 启动推理线程
            app.start_thread(
                inference_loop,
                (args, log_file_path, stream, app, wnd, tracers),
                name='OpticalFlowThread',
            )
            
            # 运行应用
            app.run(interval=1 / 30)  # 30fps 刷新
            
    except KeyboardInterrupt:
        LOG.exit_with_error_log()
    except logging_utils.UserError as e:
        LOG.exit_with_error_log(e.format())
    except Exception as e:
        LOG.exit_with_error_log(e)
    finally:
        if 'stream' in locals():
            stream.stop()
