#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 模型验证脚本 (增强版)
检查 EdgeFlowNet 模型是否正确修复，输出所有中间节点详细形状

在 OrangePi 上运行:
    python verify_onnx.py

或指定模型路径:
    python verify_onnx.py --model /path/to/model.onnx
"""

import argparse
import sys

try:
    import onnx
    import onnx.shape_inference
except ImportError:
    print("请先安装 onnx: pip install onnx")
    sys.exit(1)


def get_all_tensor_shapes(model):
    """获取所有张量的形状信息"""
    shapes = {}
    
    # 输入
    for inp in model.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value:
                dims.append(d.dim_value)
            else:
                dims.append(d.dim_param or "?")
        shapes[inp.name] = dims
    
    # 输出
    for out in model.graph.output:
        dims = []
        for d in out.type.tensor_type.shape.dim:
            if d.dim_value:
                dims.append(d.dim_value)
            else:
                dims.append(d.dim_param or "?")
        shapes[out.name] = dims
    
    # 中间值
    for vi in model.graph.value_info:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_value:
                dims.append(d.dim_value)
            else:
                dims.append(d.dim_param or "?")
        shapes[vi.name] = dims
    
    return shapes


def check_suspicious_sizes(shapes):
    """检查可疑尺寸"""
    # 预期的合法尺寸
    expected = set([1, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 
                    576, 288, 144, 72, 36, 18, 9])
    
    suspicious = []
    for name, shape in shapes.items():
        for dim in shape:
            if isinstance(dim, int) and dim > 1 and dim not in expected:
                suspicious.append((name, shape, dim))
                break
    
    return suspicious


def verify_model(model_path, verbose=False):
    """验证 ONNX 模型"""
    print("=" * 70)
    print(f"验证模型: {model_path}")
    print("=" * 70)
    
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"❌ 无法加载模型: {e}")
        return False
    
    # 运行形状推断
    print("\n[0] 运行形状推断...")
    try:
        model = onnx.shape_inference.infer_shapes(model)
        print("    ✅ 形状推断成功")
    except Exception as e:
        print(f"    ⚠️ 形状推断失败: {e}")
    
    # 获取所有形状
    shapes = get_all_tensor_shapes(model)
    
    # 1. 输入/输出形状
    print("\n[1] 输入/输出形状")
    for inp in model.graph.input:
        dims = shapes.get(inp.name, [])
        print(f"    输入: {inp.name}")
        print(f"           形状: {dims}")
    
    for out in model.graph.output:
        dims = shapes.get(out.name, [])
        print(f"    输出: {out.name}")
        print(f"           形状: {dims}")
    
    # 2. 检查可疑尺寸
    print("\n[2] 可疑尺寸检查 (非 2^n 或非预期尺寸)")
    suspicious = check_suspicious_sizes(shapes)
    if suspicious:
        print(f"    ⚠️ 发现 {len(suspicious)} 个可疑尺寸节点:")
        for name, shape, bad_dim in suspicious[:30]:  # 显示前30个
            short_name = name if len(name) < 60 else name[:57] + "..."
            print(f"       {short_name}")
            print(f"         形状: {shape}, 问题尺寸: {bad_dim}")
    else:
        print("    ✅ 未发现可疑尺寸")
    
    # 3. 统计算子类型
    print("\n[3] 关键算子统计")
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    
    key_ops = ["Conv", "ConvTranspose", "Resize", "Slice", "Concat", "Add", "Sub", "BatchNormalization"]
    for op in key_ops:
        if op in op_counts:
            print(f"    {op}: {op_counts[op]}")
    
    # 4. ConvTranspose 详情
    print("\n[4] ConvTranspose 节点详情")
    conv_transpose_nodes = [n for n in model.graph.node if n.op_type == "ConvTranspose"]
    print(f"    共 {len(conv_transpose_nodes)} 个 ConvTranspose")
    
    asymmetric_count = 0
    for node in conv_transpose_nodes:
        pads = None
        strides = None
        kernel_shape = None
        for attr in node.attribute:
            if attr.name == "pads":
                pads = list(attr.ints)
            if attr.name == "strides":
                strides = list(attr.ints)
            if attr.name == "kernel_shape":
                kernel_shape = list(attr.ints)
        
        out_shape = shapes.get(node.output[0], "未知")
        
        is_symmetric = True
        if pads:
            half = len(pads) // 2
            is_symmetric = (pads[:half] == pads[half:])
            if not is_symmetric:
                asymmetric_count += 1
        
        status = "✅" if is_symmetric else "❌ 非对称"
        
        short_name = node.name if len(node.name) < 50 else node.name[:47] + "..."
        print(f"\n    - {short_name}")
        print(f"      输出形状: {out_shape}")
        print(f"      pads: {pads} {status}")
        print(f"      strides: {strides}, kernel: {kernel_shape}")
    
    # 5. Resize 节点详情
    print("\n[5] Resize 节点详情")
    resize_nodes = [n for n in model.graph.node if n.op_type == "Resize"]
    print(f"    共 {len(resize_nodes)} 个 Resize")
    
    for node in resize_nodes:
        out_shape = shapes.get(node.output[0], "未知")
        mode = "unknown"
        for attr in node.attribute:
            if attr.name == "mode":
                mode = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
        
        short_name = node.name if len(node.name) < 50 else node.name[:47] + "..."
        print(f"    - {short_name}")
        print(f"      输出形状: {out_shape}, mode: {mode}")
    
    # 6. Slice 节点详情
    print("\n[6] Slice 节点详情")
    slice_nodes = [n for n in model.graph.node if n.op_type == "Slice"]
    print(f"    共 {len(slice_nodes)} 个 Slice")
    
    if verbose:
        for node in slice_nodes[:20]:  # 显示前20个
            out_shape = shapes.get(node.output[0], "未知")
            short_name = node.name if len(node.name) < 50 else node.name[:47] + "..."
            print(f"    - {short_name} -> {out_shape}")
    
    # 7. 总结
    print("\n" + "=" * 70)
    print("验证结果总结")
    print("=" * 70)
    
    all_ok = True
    
    # 检查 1: 可疑尺寸
    if len(suspicious) == 0 or all(s[2] == 6 for s in suspicious):  # 6 是输入通道，允许
        print("✅ 尺寸检查: 所有中间张量尺寸正常")
    else:
        non_6_suspicious = [s for s in suspicious if s[2] != 6]
        print(f"❌ 尺寸检查: 发现 {len(non_6_suspicious)} 个可疑尺寸")
        all_ok = False
    
    # 检查 2: ConvTranspose padding
    if asymmetric_count == 0:
        print("✅ ConvTranspose: 所有 padding 对称")
    else:
        print(f"❌ ConvTranspose: {asymmetric_count} 个非对称 padding")
        all_ok = False
    
    # 检查 3: 算子兼容性
    if len(conv_transpose_nodes) == 0 and len(resize_nodes) > 0:
        print("✅ 上采样方式: 使用 Resize (更兼容)")
    elif len(conv_transpose_nodes) > 0:
        print(f"⚠️ 上采样方式: 使用 ConvTranspose ({len(conv_transpose_nodes)} 个)")
    
    print()
    if all_ok:
        print("🎉 模型验证通过! 可以尝试部署。")
    else:
        print("⛔ 模型存在问题，可能导致编译失败。")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(description='验证 EdgeFlowNet ONNX 模型 (增强版)')
    parser.add_argument('--model', '-m', 
                       default='/home/orangepi/.cache/axelera/weights/edgeflownet/edgeflownet_576_1024.onnx',
                       help='ONNX 模型路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示更多详细信息')
    args = parser.parse_args()
    
    verify_model(args.model, args.verbose)


if __name__ == '__main__':
    main()
