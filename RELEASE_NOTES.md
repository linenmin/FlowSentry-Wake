![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Voyager SDK release notes v1.3.3

- [Voyager SDK release notes v1.3.3](#voyager-sdk-release-notes-v133)
  - [v1.3.3 Release Description](#v133-release-description)
  - [Fixed issues (since v1.3.1)](#fixed-issues-since-v131)
  - [New Features / Support](#new-features--support)
  - [New and Updated Documentation](#new-and-updated-documentation)
- [Voyager SDK release notes v1.3.1](#voyager-sdk-release-notes-v131)
  - [v1.3.1 Release Description](#v131-release-description)
    - [Release Qualification](#release-qualification)
    - [Host Environment](#host-environment)
  - [New Features / Support](#new-features--support-1)
    - [New Axelera AI Cards and Systems](#new-axelera-ai-cards-and-systems)
    - [New Platforms](#new-platforms)
    - [New Networks Supported](#new-networks-supported)
      - [Object Detection](#object-detection)
      - [Instance Segmentation](#instance-segmentation)
      - [Keypoint Detection](#keypoint-detection)
      - [Monocular Depth Estimation](#monocular-depth-estimation)
      - [Image Enhancement / Super Resolution](#image-enhancement--super-resolution)
      - [License Plate Recognition](#license-plate-recognition)
      - [Large Language Models](#large-language-models)
  - [Supported models (not offered in Model Zoo)](#supported-models-not-offered-in-model-zoo)
  - [NN Operator Support](#nn-operator-support)
  - [End-to-end Pipelines](#end-to-end-pipelines)
  - [General Features](#general-features)
    - [AI Pipeline Builder](#ai-pipeline-builder)
    - [Inference Visualization](#inference-visualization)
    - [Compiler](#compiler)
    - [Runtime](#runtime)
    - [Firmware](#firmware)
    - [Monitoring \[Beta\]](#monitoring-beta)
  - [Removed Features](#removed-features)
  - [New and Updated Documentation](#new-and-updated-documentation-1)
  - [Fixed issues (since v1.2.5)](#fixed-issues-since-v125)
  - [Known Issues \& Limitations](#known-issues--limitations)

## v1.3.3 Release Description

This release addresses some of the issues found in v1.3.1 and improves documentation (detailed below).


## Fixed issues (since v1.3.1)

- Yolo11 (all sizes) not working when built locally using inference.py (SDK-6886, SDK-6977)
- Unstable behaviour on axrunmodel executions(SDK-6875)
- Remove trailing "<|end|>" string after each model response(SDK-6984)
- Incorrect md5 for llama-3-1-8b-1024-static download (SDK-6985)
- AxMonitor.md contains incorrect file location (SDK-6942)
- Some LLM YAML are pointing to single-core model artifacts (SDK-6991)
- Running inference with a small number of frames can cause segfault (SDK-6992)
- Not getting all of the detections from the full size frame when using tiling (SDK-6994)

## New Features / Support

- Integrate PCI-SIG certification changes (SDK-6988)

## New and Updated Documentation

- Added support details to the end of several documents (SDK-6888)
- Updated AxInferenceNet C++ example and tutorial (SDK-7081)

---

# Voyager SDK release notes v1.3.1

## v1.3.1 Release Description

This release expands the capabilities of Voyager SDK with new features, platform support and AI use cases. It is a public release which follows the same [installation](/docs/tutorials/install.md) and upgrade principles as v1.2.5, as well as the same [licensing](/LICENSE.txt) model. An upgrade of the boards' [firmware](#firmware) is strongly recommended, 

Most notably, it adds the following capabilities:

- Support for running precompiled Large Language Models and applications for LLM-based chatbots.
- Native runtime support for Microsoft Windows 10/11 and Microsoft Windows Server 2025 host operating system on x86-based systems.
- Support for hybrid CNNs with Attention layers such as YOLO11.
- Support for new AI use cases such as Depth Estimation, Super Resolution, License Plate Recognition and Person Re-identification.
- Thermal management enables a wider operating temperature range for our boards and include thermal protection and thermal control features.
- Support for 1-chip Metis PCIe 16GB card & the Metis Compute Board.
- Support for AMD Ryzen 7 based hosts and NXP iMX8-based hosts.
- A system and Graphical User Interface for monitoring Metis devices.
- Performance improvements for most models (10% average across all model zoo)


### Release Qualification

This is a production-ready release of Voyager SDK. Software components and features that are in development are marked “\[Beta\]” indicating tested functionality that will continue to grow in future releases or “\[Experimental\]” indicating early-stage feature with limited testing.

### Host Environment

For model compiling purposes, these are the host requirements:

| Requirement               | Detail                                                                 |
| :------------------------ | :--------------------------------------------------------------------- |
| OS                        | Linux Ubuntu 22.04, Docker (on Windows or Linux), Windows + WSL/Ubuntu |
| CPU architecture          | ARM64, x86, x86\_64                                                    |
| Recommended CPU           | Intel Core-i5 or equivalent                                            |
| Minimum System Memory     | 16GB (large models may require swap partition)                         |
| Recommended System Memory | 32 GB                                                                  |

## New Features / Support

### New Axelera AI Cards and Systems

- Support for 1-chip Metis 16GB PCIe card
- \[Beta\] Support for Metis Compute Board (AI SBC)
- \[Experimental\] Support for Metis Development Kit with Arduino Portenta X8

### New Platforms

- Lenovo M75t Gen2 (AMD Ryzen 7 PRO 5750G)
- Aetina FR68 (Intel Core i9-13900E)

### New Networks Supported
For a full list of supported models and data about their performance and accuracy see [here](/docs/reference/model_zoo.md)

#### Object Detection

| Model Name                                                                  | Resolution | Format                          |
| :-------------------------------------------------------------------------- | :--------- | :------------------------------ |
| [YOLOv5s-Relu](/ax_models/zoo/yolo/object_detection/yolov5s-relu-coco.yaml) | 640x640    | PyTorch *(ONNX already exists)* |
| [YOLOv7-tiny](/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco.yaml)   | 640x640    | ONNX *(Pytorch already exists)* |
| [YOLOv8n](/ax_models/zoo/yolo/object_detection/yolov8n-coco.yaml)           | 640x640    | PyTorch *(ONNX already exists)* |
| [YOLOv8s](/ax_models/zoo/yolo/object_detection/yolov8s-coco.yaml)           | 640x640    | PyTorch *(ONNX already exists)* |
| [YOLOv8m](/ax_models/zoo/yolo/object_detection/yolov8m-coco.yaml)           | 640x640    | PyTorch *(ONNX already exists)* |
| [YOLOv8l](/ax_models/zoo/yolo/object_detection/yolov8l-coco.yaml)           | 640x640    | PyTorch *(ONNX already exists)* |
| [YOLO11n](/ax_models/zoo/yolo/object_detection/yolo11n-coco-onnx.yaml)      | 640x640    | ONNX                            |
| [YOLO11s](/ax_models/zoo/yolo/object_detection/yolo11s-coco-onnx.yaml)      | 640x640    | ONNX                            |
| [YOLO11m](/ax_models/zoo/yolo/object_detection/yolo11m-coco-onnx.yaml)      | 640x640    | ONNX                            |
| [YOLO11l](/ax_models/zoo/yolo/object_detection/yolo11l-coco-onnx.yaml)      | 640x640    | ONNX                            |
| [YOLO11x](/ax_models/zoo/yolo/object_detection/yolo11x-coco-onnx.yaml)      | 640x640    | ONNX                            |

#### Instance Segmentation

| Model Name                                                                         | Resolution | Format        |
| :--------------------------------------------------------------------------------- | :--------- | :------------ |
| [YOLOv8n-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco.yaml)      | 640x640    | PyTorch, ONNX |
| [YOLOv8s-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco.yaml)      | 640x640    | PyTorch, ONNX |
| [YOLOv8l-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco.yaml)      | 640x640    | PyTorch, ONNX |
| [YOLO11n-seg](/ax_models/zoo/yolo/instance_segmentation/yolo11nseg-coco-onnx.yaml) | 640x640    | ONNX          |
| [YOLO11l-seg](/ax_models/zoo/yolo/instance_segmentation/yolo11lseg-coco-onnx.yaml) | 640x640    | ONNX          |

#### Keypoint Detection

| Model Name                                                                        | Resolution | Format                          |
| :-------------------------------------------------------------------------------- | :--------- | :------------------------------ |
| [YOLOv8n-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco.yaml)      | 640x640    | ONNX *(Pytorch already exists)* |
| [YOLOv8s-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco.yaml)      | 640x640    | ONNX *(Pytorch already exists)* |
| [YOLOv8l-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco.yaml)      | 640x640    | ONNX *(Pytorch already exists)* |
| [YOLO11n-pose](/ax_models/zoo/yolo/keypoint_detection/yolo11npose-coco-onnx.yaml) | 640x640    | ONNX                            |
| [YOLO11l-pose](/ax_models/zoo/yolo/keypoint_detection/yolo11lpose-coco-onnx.yaml) | 640x640    | ONNX                            |

#### Monocular Depth Estimation

| Model Name                                                       | Resolution | Format |
| :--------------------------------------------------------------- | :--------- | :----- |
| [FastDepth](/ax_models/zoo/torch/fastdepth-nyudepthv2-onnx.yaml) | 224x224    | ONNX   |

#### Image Enhancement / Super Resolution

| Model Name                                                              | Resolution | Format |
| :---------------------------------------------------------------------- | :--------- | :----- |
| [Real-ESRGAN-x4plus](/ax_models/zoo/torch/real-esrgan-x4plus-onnx.yaml) | 128x128    | ONNX   |

#### License Plate Recognition

| Model Name                                 | Resolution | Format        |
| :----------------------------------------- | :--------- | :------------ |
| [LPRNet](/ax_models/zoo/torch/lprnet.yaml) | 24x94      | PyTorch, ONNX |

#### Large Language Models

This release introduces support for running pre-compiled LLMs and provides a chatbot application ([inference_llm.py](/docs/tutorials/llm.md)) to try them. This application will also output performance metrics.

| Model Name                                                            | Context Window    | Format      |
| :-------------------------------------------------------------------- | :---------------- | :---------- |
| [Phi3-mini-512](/ax_models/zoo/llm/phi3-mini-512-static.yaml)         | Up to 512 tokens  | Precompiled |
| [Phi3-mini-1024](/ax_models/zoo/llm/phi3-mini-1024-4core-static.yaml) | Up to 1024 tokens | Precompiled |
| [Phi3-mini-2048](/ax_models/zoo/llm/phi3-mini-2048-4core-static.yaml) | Up to 2048 tokens | Precompiled |
| [Llama3.2-1B](/ax_models/zoo/llm/llama-3-2-1b-1024-4core-static.yaml) | Up to 1024 tokens | Precompiled |
| [Llama3.2-3B](/ax_models/zoo/llm/llama-3-2-3b-1024-4core-static.yaml) | Up to 1024 tokens | Precompiled |
| [Llama3.1-8B](/ax_models/zoo/llm/llama-3-1-8b-1024-4core-static.yaml) | Up to 1024 tokens | Precompiled |
| [Velvet-2B](/ax_models/zoo/llm/velvet-2b-1024-4core-static.yaml)      | Up to 1024 tokens | Precompiled |

## Supported models (not offered in Model Zoo)

The following classification models can be compiled and their accuracy has been verified. While they don't have dedicated YAML configurations in our model zoo yet, you can easily use them by adapting the existing [mobilenetv4_small-imagenet.yaml](/ax_models/zoo/timm/mobilenetv4_small-imagenet.yaml) template - simply update the timm_model_args.name field to your desired model and adjust the preprocessing configuration as needed.

| Model Name                            | Accuracy Drop (vs. FP32 model) |
| :------------------------------------ | :----------------------------- |
| dla34.in1k                            | 0.59                           |
| dla60.in1k                            | 0.55                           |
| dla60_res2net.in1k                    | 0.15                           |
| dla102.in1k                           | 0.03                           |
| dla169.in1k                           | 0.27                           |
| efficientnet_es.ra_in1k               | 0.02                           |
| efficientnet_es_pruned.in1k           | 0.13                           |
| efficientnet_lite0.ra_in1k            | 0.22                           |
| dla46_c.in1k                          | 1.54                           |
| fbnetc_100.rmsp_in1k                  | 0.24                           |
| gernet_m.idstcv_in1k                  | 0.05                           |
| gernet_s.idstcv_in1k                  | 0.18                           |
| mnasnet_100.rmsp_in1k                 | 0.28                           |
| mobilenetv2_050.lamb_in1k             | 0.92                           |
| mobilenetv2_120d.ra_in1k              | 0.44                           |
| mobilenetv2_140.ra_in1k               | 0.89                           |
| res2net50_14w_8s.in1k                 | 0.17                           |
| res2net50_26w_4s.in1k                 | 0.17                           |
| res2net50_26w_6s.in1k                 | 0.06                           |
| res2net50_48w_2s.in1k                 | 0.09                           |
| res2net50d.in1k                       | 0.00                           |
| res2net101_26w_4s.in1k                | 0.19                           |
| res2net101d.in1k                      | 0.08                           |
| resnet10t.c3_in1k                     | 1.61                           |
| resnet14t.c3_in1k                     | 0.85                           |
| resnet50c.gluon_in1k                  | 0.03                           |
| resnet50s.gluon_in1k                  | 0.19                           |
| resnet101c.gluon_in1k                 | 0.08                           |
| resnet101d.gluon_in1k                 | 0.1                            |
| resnet101s.gluon_in1k                 | 0.18                           |
| resnet152d.gluon_in1k                 | 0.15                           |
| selecsls42b.in1k                      | 0.25                           |
| selecsls60.in1k                       | 0.05                           |
| selecsls60b.in1k                      | 0.2                            |
| spnasnet_100.rmsp_in1k                | 0.25                           |
| tf_efficientnet_es.in1k               | 0.26                           |
| tf_efficientnet_lite0.in1k            | 0.33                           |
| tf_mobilenetv3_large_minimal_100.in1k | 1.68                           |
| wide_resnet101_2.tv2_in1k             | 0.26                           |

Additionally, we support **OSNet x1_0** for model compilation.

## NN Operator Support
Our operator coverage and documentation have expanded to support 7 ONNX opsets (opset11-opset17). To ensure optimal performance and stability, we specifically encourage using the following 4 opsets.
- [onnx-opset14](/docs/reference/onnx-opset14-support.md)
- [onnx-opset15](/docs/reference/onnx-opset15-support.md)
- [onnx-opset16](/docs/reference/onnx-opset16-support.md)
- [onnx-opset17](/docs/reference/onnx-opset17-support.md)

## End-to-end Pipelines
- Support for broadened cascade functionality to enable the selection of the top-K ROIs, prioritizing them by their proximity to the center (nearest first), detection score (highest first), or area (largest first). See also [documentation](/docs/reference/pipeline_operators.md#example-axtransformpostamble).
- New YAML files for all new models offered in our model zoo in this release (see tables above).

## General Features

### AI Pipeline Builder

- New user-facing tool for performing LLM inference (inference_llm.py).
- Support for returning the raw output tensor to the application (e.g., to enable application to implement custom decoding).
- Support for `AxTransformPostamble`, a new plug-in to simplify integrating customer models when building a model decoder (makes a complex process much simpler). See also [documentation](/docs/reference/pipeline_operators.md#example-axtransformpostamble).
- \[Limited\]Support for Tiled Inference for single-model pipelines.

### Inference Visualization

- New visualization options are available:
  - Set a per stream title, and adjust size/color/font
  - Enable displaying the video in grayscale to make annotations easier to see
  - Adjust color of bounding boxes and labels on a per-class-id basis
  - Configure the format used to label detections, for example to include 'Person 59%' or just '0.59' 

- Augment the visualization with labels and images
  - Add, update and remove text labels. For example to dynamically show number of detections in the window
  - Add, update and remove images. For example to show logos
  - These items can be set to smoothly fade in and out

### Compiler

- Expanded support for ONNX operators and opset versions.
- Support for the compiler CLI and expanded set of configuration parameters.

### Runtime

- Support for Microsoft Windows as a runtime environment.
  * We provide a native Windows driver and support for `AxRuntime`, with which native Windows applications can offload inference to Metis
  * At the time of the release, the Windows driver has not completed Microsoft certification so it requires manual installation as described in the [installation](/docs/tutorials/windows/installing_driver.md) guide. Once certification finishes this should not loger be needed.
- Support for cascaded models with `AxInferenceNet`

### Firmware

The following features have been added. To benefit from these features, a firmware upgrade of both Metis and board controller is required, please refer to this [firmware flash update](/docs/tutorials/firmware_flash_update.md) document.

- **Dynamic fan control**
On boards with a fan, the speed of the fan is determined by the temperature of the board. In low temperatures the fan no longer operates at 100% speed, resulting in a significant reduction in the noise generated by the card and, therefore, in an improved User Experience.

- **Increased temperature range for PCIe and M.2 boards**
REV1.1 boards are expected to operate within [-20°C, +70°C] without any loss of function, performance degradation or impact to the lifetime of the boards.
The default values of temperature thresholds have been finalized to the values documented in the [Thermal Guide](/docs/reference/thermal_guide.md)

### Monitoring \[Beta\]

- Launch of `AxMonitor` as system monitoring tool (see [documentation](/docs/tutorials/axmonitor.md) for more details)

## Removed Features

- Inference Server : This experimental feature has been removed.

## New and Updated Documentation

- Guide for SDK [installation](/docs/tutorials/windows/installing_driver.md) on Windows
- Getting Started [guide](/docs/tutorials/windows/windows_getting_started.md) for performing inference on Windows
- In-depth [tutorial](/ax_models/tutorials/general/tutorials.md) for model deployment
- Documentation on the Compiler [CLI](/docs/reference/compiler_cli.md) and [configuration parameters](/docs/reference/compiler_configs_full.md)
- [Tutorial](/docs/tutorials/llm.md) for running LLM inference
- [Tutorial](/docs/tutorials/axmonitor.md) for system monitoring using `AxMonitor`
- Updated [tutorial](/docs/tutorials/firmware_flash_update.md) on upgrading the firmware to the latest version
- Updated documentation on ONNX Opset support ([opset14](/docs/reference/onnx-opset14-support.md), [opset15](/docs/reference/onnx-opset15-support.md), [opset16](/docs/reference/onnx-opset16-support.md) and [opset17](/docs/reference/onnx-opset17-support.md)
- [Guide](/docs/reference/thermal_guide.md) on thermal management features

## Fixed issues (since v1.2.5)

- **Software configuration limits temperature operating range to [0°C ~ 50°C] (SDK-4943, AIS-806, AIS-821)**
  Now Metis-based M.2 and PCIe cards can operate in their full temperature range.

- **Memory Leaks on host application**
  Fixed memory leaks in the host CPU that could appear under specific usage patterns.

## Known Issues & Limitations

- **Device monitoring does not work on single-MSI hosts (SDK-6581)**
  Device monitoring (via `axsystemserver` and `axmonitor` does not work on hosts that only provide a single PCIe interrupt (“MSI”). Arduino Portenta x8 is one such host. 

- **Metis does not come up after reboot on RK3588 hosts (SDK-5176)**
  On RK3588-based hosts (e.g., Firefly and Aetina), it has been observed that the PCIe card and M.2 module is not powered on by the host upon reboot. Until the issue is solved by Rockchip, the issue can be prevented by powering the host off and on instead of rebooting. To recover from a system where the issue manifests, a host power cycle is required, too (i.e., power-off followed by power-on).

- **Accuracy measurement may fail on RK3588 hosts (SDK-5228)**
  The accuracy measurement benchmark driver performs inference on a large number of images (5000 or more). We have observed that on Arm RK3588 systems the accuracy measurement fails for some NNs. Currently, the only workaround is to run the accuracy measurement on an x86 host.

- **On entering a docker container, must run ‘make operators’ (SDK-5228)**
  When using docker, built Axelera operators are not retained after exiting the docker. As a workaround, it is necessary to run ‘make operators’ on entering the docker before running applications on Metis hardware.

- **No quantized accuracy for Real-ESRGAN-x4plus (SDK-6749)**
  It is not currently possible to measure the quantized accuracy of the Real-ESRGAN-x4plus super
  resolution model using `--pipe=torch-aipu`. However torch and metis end-to-end and accuracy can
  both be measured using the following commands:

```
 ./inference.py real-esrgan-x4plus-onnx dataset --pipe=torch --no-display
 ./inference.py real-esrgan-x4plus-onnx dataset --no-display
```
