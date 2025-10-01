![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Voyager SDK release notes v1.4.1

- [Voyager SDK release notes v1.4.1](#voyager-sdk-release-notes-v141)
  - [Release Description](#release-description)
    - [Release Qualification](#release-qualification)
    - [Host Environment](#host-environment)
  - [New Features / Support](#new-features--support)
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
    - [Monitoring](#monitoring)
  - [Removed Features](#removed-features)
  - [New and Updated Documentation](#new-and-updated-documentation)
  - [Fixed issues last release](#fixed-issues-last-release)
  - [Known Issues \& Limitations](#known-issues--limitations)

## Release Description


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


### New Platforms


### New Networks Supported
For a full list of supported models and data about their performance and accuracy see [here](/docs/reference/model_zoo.md)

#### Object Detection

| Model Name                                                                  | Resolution | Format                          |
| :-------------------------------------------------------------------------- | :--------- | :------------------------------ |
e.g...
| [Model name](/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco.yaml) | e.g. 640x640    | e.g. PyTorch *(ONNX already exists)* |

#### Instance Segmentation

| Model Name                                                                         | Resolution | Format        |
| :--------------------------------------------------------------------------------- | :--------- | :------------ |
|                                                                                    |            |               |
#### Keypoint Detection

| Model Name                                                                        | Resolution | Format                          |
| :-------------------------------------------------------------------------------- | :--------- | :------------------------------ |
|                                                                                   |            |                                 |

#### Monocular Depth Estimation

| Model Name                                                       | Resolution | Format |
| :--------------------------------------------------------------- | :--------- | :----- |
|                                                                  |            |        |

#### Image Enhancement / Super Resolution

| Model Name                                                              | Resolution | Format |
| :---------------------------------------------------------------------- | :--------- | :----- |
|                                                                         |            |        |

#### License Plate Recognition

| Model Name                                 | Resolution | Format        |
| :----------------------------------------- | :--------- | :------------ |
|                                            |            |               |

#### Large Language Models

For support for running pre-compiled LLMs and provides a chatbot application ([inference_llm.py](/docs/tutorials/llm.md)) to try them. This application will also output performance metrics.

| Model Name                                                            | Context Window    | Format      |
| :-------------------------------------------------------------------- | :---------------- | :---------- |
|                                                                       | e.g. Up to 512 tokens  | e.g. Precompiled |

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

## General Features

### AI Pipeline Builder

### Inference Visualization

### Compiler

### Runtime

### Firmware

### Monitoring

## Removed Features

## New and Updated Documentation

## Fixed issues last release

## Known Issues & Limitations
