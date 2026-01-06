#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 模型验证脚本
检查 EdgeFlowNet 模型是否正确修复

在 OrangePi 上运行:
    python verify_onnx.py

或指定模型路径:
    python verify_onnx.py --model /path/to/model.onnx
"""

import argparse
import sys

try:
    import onnx
except ImportError:
    print("请先安装 onnx: pip install onnx")
    sys.exit(1)


def verify_model(model_path):
    """验证 ONNX 模型"""
    print("=" * 60)
    print(f"验证模型: {model_path}")
    print("=" * 60)
    
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"❌ 无法加载模型: {e}")
        return False
    
    # 1. 检查输入形状
    print("\n[1] 输入形状检查")
    for inp in model.graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"    输入: {inp.name}")
        print(f"    形状: {dims}")
        
        # 检查是否为 16 的倍数
        if len(dims) >= 4:
            h, w = dims[1], dims[2]  # 假设 NHWC
            h_ok = h % 16 == 0
            w_ok = w % 16 == 0
            status = "✅" if (h_ok and w_ok) else "⚠️"
            print(f"    16倍数检查: H={h} ({h_ok}), W={w} ({w_ok}) {status}")
    
    # 2. 检查 ConvTranspose 节点
    print("\n[2] ConvTranspose 节点检查")
    conv_transpose_nodes = [n for n in model.graph.node if n.op_type == "ConvTranspose"]
    print(f"    找到 {len(conv_transpose_nodes)} 个 ConvTranspose 节点")
    
    asymmetric_count = 0
    for node in conv_transpose_nodes:
        pads = None
        for attr in node.attribute:
            if attr.name == "pads":
                pads = list(attr.ints)
                break
        
        if pads:
            half = len(pads) // 2
            is_symmetric = (pads[:half] == pads[half:])
            status = "✅ 对称" if is_symmetric else "❌ 非对称"
            if not is_symmetric:
                asymmetric_count += 1
            print(f"    - {node.name[:60]}...")
            print(f"      pads: {pads} {status}")
    
    if asymmetric_count > 0:
        print(f"\n    ⚠️ 警告: 有 {asymmetric_count} 个非对称 padding 的 ConvTranspose!")
        print("    这将导致 Axelera 编译失败。请重新运行 extract_onnx.py 并上传。")
    
    # 3. 检查 Crop 节点 (修复后应该有)
    print("\n[3] Shifted Conv 裁剪层检查")
    crop_nodes = [n for n in model.graph.node if "_crop" in n.name.lower()]
    print(f"    找到 {len(crop_nodes)} 个 Crop 节点")
    
    if len(crop_nodes) == 0:
        print("    ⚠️ 警告: 没有找到 Crop 节点，模型可能未修复!")
    else:
        for node in crop_nodes[:5]:
            print(f"    - {node.op_type}: {node.name[:60]}...")
    
    # 4. 总结
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)
    
    all_ok = True
    
    # 检查项 1: 输入形状是否正确
    if len(dims) >= 4 and dims[1] == 576 and dims[2] == 1024:
        print("✅ 输入分辨率: 576×1024 (正确)")
    else:
        print(f"⚠️ 输入分辨率: {dims[1]}×{dims[2]} (期望 576×1024)")
        all_ok = False
    
    # 检查项 2: ConvTranspose padding 是否对称
    if asymmetric_count == 0:
        print("✅ ConvTranspose padding: 全部对称")
    else:
        print(f"❌ ConvTranspose padding: {asymmetric_count} 个非对称")
        all_ok = False
    
    # 检查项 3: 是否有 Crop 节点
    if len(crop_nodes) == 4:
        print("✅ Crop 节点: 4 个 (修复完成)")
    elif len(crop_nodes) > 0:
        print(f"⚠️ Crop 节点: {len(crop_nodes)} 个 (期望 4 个)")
    else:
        print("❌ Crop 节点: 0 个 (模型未修复)")
        all_ok = False
    
    print()
    if all_ok:
        print("🎉 模型验证通过! 可以部署。")
    else:
        print("⛔ 模型存在问题，请重新生成并上传。")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description='验证 EdgeFlowNet ONNX 模型')
    parser.add_argument('--model', '-m', 
                       default='/home/orangepi/.cache/axelera/weights/edgeflownet/edgeflownet_576_1024.onnx',
                       help='ONNX 模型路径')
    args = parser.parse_args()
    
    verify_model(args.model)


if __name__ == '__main__':
    main()
