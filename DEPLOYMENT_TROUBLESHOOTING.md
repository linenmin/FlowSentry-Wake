# EdgeFlowNet Axelera 部署问题排查与解决方案

本文档详细记录了将 EdgeFlowNet 光流模型部署到 OrangePi + Axelera M.2 加速器过程中遇到的问题及解决方案。

---

## 目录

1. [项目背景](#项目背景)
2. [问题一：DataAdapter 类继承错误](#问题一dataadapter-类继承错误)
3. [问题二：Model 类初始化参数冲突](#问题二model-类初始化参数冲突)
4. [问题三：SDK 忽略自定义 DataAdapter](#问题三sdk-忽略自定义-dataadapter)
5. [问题四：override_preprocess 返回值维度错误](#问题四override_preprocess-返回值维度错误)
6. [问题五：输入分辨率不是 16 的倍数](#问题五输入分辨率不是-16-的倍数)
7. [问题六：ConvTranspose 非对称 Padding](#问题六convtranspose-非对称-padding)
8. [问题七：Slice 操作硬件限制](#问题七slice-操作硬件限制)
9. [最终解决方案](#最终解决方案)
10. [经验总结](#经验总结)

---

## 项目背景

### 模型信息
- **模型**: EdgeFlowNet (光流估计模型)
- **输入**: 6 通道 (两帧 RGB 图像拼接) `[1, H, W, 6]`
- **输出**: 2 通道光流场 `[1, H, W, 2]` (u, v 速度分量)
- **原始分辨率**: 960×540 → **最终分辨率**: 1024×576

### 部署目标
- **硬件**: OrangePi 5 Plus + Axelera M.2 加速器
- **SDK**: Axelera Voyager SDK
- **任务**: 编译量化模型并部署推理

---

## 问题一：DataAdapter 类继承错误

### 错误信息
```
TypeError: OpticalFlowDataAdapter is not a subclass of DataAdapter
```

### 原因分析
自定义的 `OpticalFlowDataAdapter` 类没有继承 Axelera SDK 的 `types.DataAdapter` 基类。

### 错误代码
```python
# ❌ 错误
class OpticalFlowDataAdapter:
    def __init__(self, repr_imgs_dir_path, repr_imgs_dataloader_color_format='RGB'):
        ...
```

### 解决方案
```python
# ✅ 正确
from axelera import types

class OpticalFlowDataAdapter(types.DataAdapter):
    def __init__(self, dataset_config, model_info):  # SDK 要求的签名
        ...
```

### 关键点
- **必须继承** `axelera.types.DataAdapter`
- **构造函数签名必须是** `__init__(self, dataset_config, model_info)`
- SDK 通过配置传递参数，而不是直接传递路径

---

## 问题二：Model 类初始化参数冲突

### 错误信息
```
ERROR : ONNXModel.__init__() got an unexpected keyword argument 'name'
```

### 原因分析
`EdgeFlowNetModel` 自定义了 `__init__` 方法，但 SDK 在实例化时会传递额外参数（如 `name`），与父类签名冲突。

### 错误代码
```python
# ❌ 错误
class EdgeFlowNetModel(base_onnx.AxONNXModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 父类不接受 'name' 参数
        self.prev_frame = None
```

### 解决方案
```python
# ✅ 正确：使用 init_model_deploy 而不是 __init__
class EdgeFlowNetModel(base_onnx.AxONNXModel):
    prev_frame = None  # 类属性
    
    def init_model_deploy(self, model_info, dataset_config, **kwargs):
        super().init_model_deploy(model_info, dataset_config, **kwargs)
        self.prev_frame = None  # 实例属性初始化
```

### 关键点
- **不要自定义 `__init__`**，SDK 会调用 `init_model_deploy` 进行初始化
- 自定义属性应在 `init_model_deploy` 中初始化

---

## 问题三：SDK 忽略自定义 DataAdapter

### 现象
```
INFO : Using representative images from /home/orangepi/.cache/axelera/data/coco2017_repr400
```

即使配置了自定义 DataAdapter，SDK 始终使用 COCO 数据集进行量化校准。

### 尝试的解决方案

#### 方案 A：配置 repr_imgs_dir_path 指向自定义目录
```yaml
datasets:
  EdgeFlowNetCalibrationDataset:
    repr_imgs_dir_path: data/calib_edgeflownet/0000/left
```
**结果**：`num_samples=0`，ImageReader 不递归搜索子目录

#### 方案 B：移除 repr_imgs_dir_path
```yaml
datasets:
  EdgeFlowNetCalibrationDataset:
    # repr_imgs_dir_path 被移除
    calib_data_path: data/calib_edgeflownet
```
**结果**：SDK 回退到 COCO 数据

#### 方案 C：实现 create_calibration_data_loader
```python
def create_calibration_data_loader(self, batch_size, num_samples, **kwargs):
    return DataLoader(CustomDataset(), batch_size=batch_size)
```
**结果**：SDK 仍然优先使用 COCO

### 根本原因
SDK 的数据加载逻辑优先级固定：
1. 如果有 `repr_imgs_dir_path` → 使用内置 `ImageReader.PIL`
2. 如果没有或为空 → 下载并使用 COCO
3. **不会自动调用** 自定义的 `create_calibration_data_loader`

### 最终解决方案：预处理拼接法

将帧对**水平拼接**成标准 RGB 图片 `[H, 2W, 3]`，让 SDK 的 ImageReader 正常读取：

```
Frame1 [576, 1024, 3] + Frame2 [576, 1024, 3]
                    ↓ np.hstack
           Merged [576, 2048, 3] → 保存为 PNG
```

然后在 `override_preprocess` 中切分并重组为 6 通道：

```python
def override_preprocess(self, img):
    h, w, c = img.shape
    is_merged = (w > h * 2.5)  # 检测是否为拼接图片
    
    if is_merged:
        half_w = w // 2
        img1 = img[:, :half_w, :]
        img2 = img[:, half_w:, :]
        combined = np.concatenate([img1, img2], axis=-1)  # [H, W, 6]
    ...
```

---

## 问题四：override_preprocess 返回值维度错误

### 错误信息
```
ERROR : Invalid rank for input: input_frames:0 Got: 5 Expected: 4
```

### 原因分析
`override_preprocess` 返回了带 batch 维度的 tensor `[1, H, W, 6]`，SDK 内部又添加了一层 batch 维度，导致 Rank 5。

### 错误代码
```python
# ❌ 错误：返回 Rank 4
combined = np.expand_dims(combined, axis=0)  # [1, H, W, 6]
return torch.from_numpy(combined)
```

### 解决方案
```python
# ✅ 正确：返回 Rank 3，不带 batch 维度
combined = np.concatenate([img1, img2], axis=-1)  # [H, W, 6]
return torch.from_numpy(combined)  # SDK 会自动添加 batch
```

### 关键点
- `override_preprocess` 应返回**单个样本**，不含 batch 维度
- SDK 会自行处理 batching

---

## 问题五：输入分辨率不是 16 的倍数

### 错误信息
```
ERROR : The Relay type checker is unable to show the following types match:
  Tensor[(270, 484, 4, 64), int8]
  Tensor[(270, 480, 1, 64), int8]
dimension 1 conflicts: 484 does not match 480
```

### 原因分析
原始分辨率 960×540 中，**540 不是 16 的倍数**：
- 960 / 16 = 60 ✓
- 540 / 16 = 33.75 ✗

EdgeFlowNet 有 4 次下采样（stride=2），需要输入是 2^4 = 16 的倍数。

### 解决方案
将分辨率改为 **1024×576**（16:9，两者都是 16 的倍数）：
- 1024 / 16 = 64 ✓
- 576 / 16 = 36 ✓

需要修改的文件：
- `extract_onnx.py`: INPUT_HEIGHT, INPUT_WIDTH
- `edgeflownet_model.py`: input_height, input_width
- `edgeflownet-opticalflow.yaml`: input_tensor_shape

---

## 问题六：ConvTranspose 非对称 Padding

### 错误信息
```
WARNING : Unsatisfied constraint: pads is None or pads[:len(pads)//2] == pads[len(pads)//2:]
ERROR : The Relay type checker is unable to show the following types match...
```

### 原因分析
EdgeFlowNet 的 ConvTranspose 层使用**非对称 padding**：
```
pads: [0, 0, 1, 1]  # top=0, left=0, bottom=1, right=1
pads: [1, 1, 2, 2]  # top=1, left=1, bottom=2, right=2
pads: [2, 2, 3, 3]  # top=2, left=2, bottom=3, right=3
```

**Axelera 硬件要求 padding 必须对称**：`pads[:2] == pads[2:]`

### 尝试的解决方案

#### 方案 A：Padding + Slice 裁剪
将 pads 设为 0，然后用 Slice 裁剪多余部分。

**失败原因**：Axelera 的 Slice 有严格限制：
- `axes is not None and len(axes) == 1` (只能单轴操作)
- `(axes[0] != 1 and data.shape[1] % 64 == 0) or axes[0] == 1` (通道数必须是 64 倍数)

模型中间层通道数为 32、16，不满足 64 倍数约束。

#### 方案 B：Resize + Conv 替代 ConvTranspose
将 ConvTranspose 替换为 Resize + Conv 组合。

**未采用原因**：需要较复杂的图转换，且数学上是近似等价。

#### 方案 C：通道对齐法
先 Concat 凑通道到 64 倍数，再 Slice，再还原。

**未采用原因**：算子多、图结构复杂，容易出错。

### 最终解决方案：Shifted Depthwise Conv

核心思想：使用 **NxN Depthwise Conv** 替代 Slice 进行裁剪。

| pads      | 总 Pad | Kernel Size | 热点位置 | 效果                         |
| --------- | ------ | ----------- | -------- | ---------------------------- |
| [0,0,1,1] | 1      | 2×2         | [0,0]    | 保留左上，裁剪右下 1 行 1 列 |
| [1,1,2,2] | 3      | 4×4         | [1,1]    | 跳过开头 1，裁剪结尾 2       |
| [2,2,3,3] | 5      | 6×6         | [2,2]    | 跳过开头 2，裁剪结尾 3       |

**卷积核权重设计**：只在热点位置设为 1，其他为 0
```python
kernel = np.zeros((C, 1, k_size, k_size))
kernel[:, 0, crop_top, crop_left] = 1.0
```

**实现代码**：
```python
# 修改 ConvTranspose pads 为 0
node.attrs["pads"] = [0, 0, 0, 0]

# 插入 Shifted Depthwise Conv
crop_node = gs.Node(
    op="Conv",
    attrs={
        "pads": [0, 0, 0, 0],      # Valid padding
        "kernel_shape": [k_size, k_size],
        "group": C                  # Depthwise
    }
)
```

---

## 问题七：Slice 操作硬件限制

### Axelera Slice 约束
```
rule: axes is not None and len(axes) == 1
rule: steps is None or (len(steps) == 1 and (steps == 1 or axes == 1))
rule: (axes[0] != 1 and data.shape[1] % 64 == 0) or axes[0] == 1
```

### 含义
| 约束                                        | 说明                                 |
| ------------------------------------------- | ------------------------------------ |
| `len(axes) == 1`                            | 只能沿单个轴切片                     |
| `axes[0] == 1` 或 `data.shape[1] % 64 == 0` | 非通道轴切片时，通道数必须是 64 倍数 |

### 影响
由于模型中间层通道数为 128/64/32/16，只有 128 和 64 满足约束，32 和 16 不满足。

这导致 **Slice 裁剪方案不可行**，最终采用 Shifted Depthwise Conv 方案。

---

## 最终解决方案

### 1. 分辨率对齐
```
960×540 → 1024×576 (16 的倍数)
```

### 2. ConvTranspose 修复
在 `extract_onnx.py` 中集成 `fix_convtranspose_for_axelera()` 函数，导出后自动修复：
- 将非对称 pads 改为 [0,0,0,0]
- 插入 Shifted Depthwise Conv 进行裁剪

### 3. 校准数据预处理
使用 `prepare_calib_data.py` 将帧对水平拼接为 [576, 2048, 3] 的 PNG 图片。

### 4. override_preprocess 适配
自动检测拼接图片（宽 > 高×2.5），切分并重组为 6 通道。

### 部署流程
```bash
# 1. 导出 ONNX（自动修复 ConvTranspose）
python extract_onnx.py

# 2. 准备校准数据
python prepare_calib_data.py --input <帧序列目录> --output data/calib_merged

# 3. 上传到 OrangePi
scp edgeflownet_576_1024.onnx orangepi@host:~/.cache/axelera/weights/edgeflownet/
scp -r data/calib_merged orangepi@host:~/.cache/axelera/data/
scp ax_models/custom/* orangepi@host:~/voyager-sdk/ax_models/custom/

# 4. 部署
./deploy.py edgeflownet-opticalflow
```

---

## 经验总结

### SDK 接口规范
- 自定义类**必须严格遵循** SDK 的签名和继承规范
- 使用 `init_model_deploy` 而非 `__init__` 进行初始化
- `override_preprocess` 返回**不带 batch 维度**的单样本

### 硬件约束
- **ConvTranspose**: padding 必须对称
- **Slice**: 只能单轴操作，非通道轴需要通道数是 64 倍数
- **分辨率**: 输入尺寸应是 2^(下采样次数) 的倍数

### 模型修改技巧
- **Shifted Depthwise Conv**: 利用卷积核热点位置同时实现开头和结尾的裁剪
- **预处理拼接法**: 将多帧输入转换为标准 RGB 图片，绕过自定义 DataAdapter 的困难

### 调试建议
- 使用 Netron 可视化 ONNX 模型结构
- 检查 SDK 日志中的 WARNING 信息
- 利用 onnx-graphsurgeon 进行图手术

---

## 相关文件

| 文件                                            | 说明                                |
| ----------------------------------------------- | ----------------------------------- |
| `extract_onnx.py`                               | TF → ONNX 转换 + ConvTranspose 修复 |
| `prepare_calib_data.py`                         | 校准数据预处理（帧对拼接）          |
| `ax_models/custom/edgeflownet_model.py`         | 模型类和 DataAdapter                |
| `ax_models/custom/edgeflownet-opticalflow.yaml` | 部署配置                            |

---

*文档更新日期: 2026-01-06*
