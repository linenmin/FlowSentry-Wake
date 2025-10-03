![Axelera banner](docs/images/Ax_Page_Banner_2500x168_01.png)

# Voyager SDK v1.4 Release Notes

- [Voyager SDK Release Notes 1.4.1](#voyager-sdk-release-notes-141)
  - [Fixed issues (since v1.4.0)](#fixed-issues-since-v140)
  - [New Features / Support](#new-features--support)
- [Voyager SDK Release Notes 1.4.0](#voyager-sdk-release-notes-140)
  - [Release Description](#release-description)
    - [Metis Cards and Systems Support](#metis-cards-and-systems-support)
    - [Release Qualification](#release-qualification)
  - [Features](#features)
    - [New Platforms](#new-platforms)
    - [New Networks Supported](#new-networks-supported)
      - [Pre-compiled models](#pre-compiled-models)
      - [Model Zoo for Computer Vision](#model-zoo-for-computer-vision)
        - [New models for Object Detection](#new-models-for-object-detection)
        - [New models for Semantic Segmentation](#new-models-for-semantic-segmentation)
        - [New models for Person Re-Identification / Multi-Object Tracking](#new-models-for-person-re-identification--multi-object-tracking)
        - [New models for Face Recognition](#new-models-for-face-recognition)
      - [End-to-end Pipelines](#end-to-end-pipelines)
    - [General Features](#general-features)
      - [Installation](#installation)
      - [AI Pipeline Builder](#ai-pipeline-builder)
      - [[Beta] Model Compiler](#beta-model-compiler)
      - [Runtime](#runtime)
      - [Tools](#tools)
      - [Firmware](#firmware)
  - [System Requirement](#system-requirement)
    - [Development Environment](#development-environment)
    - [Runtime Environment](#runtime-environment)
  - [Fixed Issues](#fixed-issues)
  - [Known Issues & Limitations](#known-issues--limitations)

# Voyager SDK Release Notes 1.4.1

This release addresses some of the issues found in v1.4.0.

## Fixed issues (since v1.4.0)

- FastSAM app not working properly standalone as it does in
  inference.py (SDK-7503)
- Wrong msi info data registered when working in single msi bug
  detected on Intel N15 host CPU (SDK-7527)
- Build issues due to proto version check script path resolution issue
  (SDK-7608)
- Incorrect handing of tracers in stream response for UI e.g. Gradio
  for LLM Chatbot (SDK-7720)
- Error 'extended fdi already open' when running inference and double
  buffering and output dma bufs are disabled (SDK-7768)

## New Features / Support

- Double buffering for inference on Raspberry Pi 5 and Portenta X8
  improving performance (SDK-7107)

# Voyager SDK Release Notes 1.4.0

## Release Description

Voyager SDK v.1.4 release brings new models like YOLOv10 and new
applications like person re-identification and face recognition.
Significant performance improvements on Windows platforms and compiler
python API are included. User experience improvements include removing
the requirement of a token to install the SDK, a command-line tool to
query the SDK version and fixes to improve stability across systems.

### Metis Cards and Systems Support

- The new board supported in this release is [Metis Compute
  Board](https://store.axelera.ai/collections/ai-acceleration-cards/products/metis-compute-board-with-arm-based-rk3588)
  with 4 GB RAM

### Release Qualification

This is a production-ready release of Voyager SDK. Software components
and features that are in development are marked "[Beta]" indicating
tested functionality that will continue to grow in future releases or
"[Experimental]" indicating early-stage feature with limited testing.

## Features

### New Platforms

- Nvidia Jetson Orin NX (Arm Cortex-A78AE)

### New Networks Supported

Voyager SDK model zoo includes computer vision tasks and LLMs. For a
full list of supported models and data about their performance and
accuracy see
[here](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.4/docs/reference/model_zoo.md).

Models that are supported but not included in the model zoo are
documented here.

#### Pre-compiled models

- The `download_prebuilt.py` utility for downloading pre-compiled models
  has been deprecated in favor of `axdownloadmodel`.

#### Model Zoo for Computer Vision

##### New models for Object Detection

| Model Name                                                            | Resolution | Format |
| :-------------------------------------------------------------------- | :--------- | :----- |
| [YOLO10n](ax_models/zoo/yolo/object_detection/yolov10n-coco-onnx.yaml) | 640x640    | ONNX   |
| [YOLO10s](ax_models/zoo/yolo/object_detection/yolov10s-coco-onnx.yaml) | 640x640    | ONNX   |
| [YOLO10b](ax_models/zoo/yolo/object_detection/yolov10b-coco-onnx.yaml) | 640x640    | ONNX   |

##### New models for Semantic Segmentation

| Model Name                                                            | Resolution | Format |
| :-------------------------------------------------------------------- | :--------- | :----- |
| [FastSAM](ax_models/reference/apps/fastsam/fastsams-rn50x4-onnx.yaml) | 640x640    | ONNX   |

##### New models for Person Re-Identification / Multi-Object Tracking

| Model Name                                                                                                                                      | Resolution | Format |
| :---------------------------------------------------------------------------------------------------------------------------------------------- | :--------- | :----- |
| [OSNet x1_0](ax_models/zoo/torch/osnet-x1-0-market1501-onnx.yaml)                                                                              | 256x128    | ONNX   |
| [Deep-OC-Sort](ax_models/reference/cascade/with_tracker/yolox-deep-oc-sort-osnet.yaml) (combined with YOLOX but can be adapted to other object detectors) | 384x128    | ONNX   |

##### New models for Face Recognition

| Model Name                                                                 | Resolution | Format        |
| :------------------------------------------------------------------------- | :--------- | :------------ |
| [FaceNet - InceptionResnetV1](ax_models/zoo/torch/facenet-lfw.yaml)       | 160x160    | Pytorch, ONNX |

#### End-to-end Pipelines

- End-to-end application for 'segment anything' using FastSAM.
- New YAML files for all new models offered in our model zoo in this
  release (see tables above).

### General Features

#### Installation

- The installer has been updated to remove the requirement of
  token-based authentication. The SDK can be installed without
  providing a user identity (username/token).

#### AI Pipeline Builder

- An `InferenceStream` can be created on multiple different pipelines,
  and pipelines can be added dynamically using `add_pipeline()`.
- `AxInferenceNet` supports Top-k Region-Of-Interest (ROI) filtering
  based on area, size, classes, etc.
- Classification metadata have been simplified to make classification
  metadata objects easier to use. See also the
  [classification example](https://github.com/axelera-ai/internal-voyager-sdk/blob/release/v1.4.0-0-g29645cdd9/examples/classification_example.py).

#### [Beta] Model Compiler

- Support for a simplified [compilation API in
  Python](https://github.com/axelera-ai/internal-voyager-sdk/blob/ae36f8fe311ea1ea6be63795a6e5d89bd0a976b5/docs/reference/compiler_api.md)
  based on two API calls: `quantize()` and `compile()`. Compiled models
  can be run using `AxRunModel` tool or AxRuntime inference API.
- Support for concat along any non-batch axis of a 4d tensor, and
  grouped convolutions for symmetric kernels (kernel width equals
  kernel height).
- List of supported operators documented
  [here](https://github.com/axelera-ai/internal-voyager-sdk/blob/release/v1.4.0-0-g29645cdd9/docs/reference/onnx-opset17-support.md)
  will grow in future releases. For technical assistance on compiling
  your own model please turn to
  [Axelera Community](https://community.axelera.ai/).

#### Runtime

- Significant performance improvement for several models using
  AxRuntime inference API on Windows platforms.
- Support for multiple Axelera cards on Windows platforms.

#### Tools

- New `axversion` command-line utility to query the version of the SDK
  running.
- Enhanced functionality of `axmonitor`
  - New metrics — power usage for cards that feature on-board power
    sensors, kernels per second.
  - Added multi-device support.
  - Included configuration and threshold settings information.
  - Added monitoring messages for hardware throttling, software
    throttling, and version.
- Improvements in `axdevice`
  - Improved multi-device handling and PCIe bridge management.
  - Added reboot and thermal commands.

#### Firmware

The following features have been added.

- Improved version checking commands for board controller.
- Improved resiliency and stability across different host platforms.
- Updated firmware upgrade script documented in the guide
  [here](https://github.com/axelera-ai/internal-voyager-sdk/blob/ae36f8fe311ea1ea6be63795a6e5d89bd0a976b5/docs/tutorials/firmware_flash_update.md).

## System Requirement

### Development Environment

For model compiling purposes, these are the host requirements:

- OS: Linux Ubuntu 22.04, Docker (on Windows or Linux), Windows +
  WSL/Ubuntu
- CPU architecture: ARM64, x86, x86_64
- Recommended CPU: Intel Core-i5 or equivalent
- Minimum System Memory: 16GB (large models may require swap
  partition)
- Recommended System Memory: 32 GB

### Runtime Environment

This release is expected to work with Intel Core-i 12th and 13th
generations (x86), AMD Ryzen (x86) and Arm64 host CPUs. Please find the
list of platforms Axelera AI has tested with Metis M.2 Card
[here](https://support.axelera.ai/hc/en-us/articles/25437844422418-Metis-M-2-Tested-Host-PCs)
and Metis PCIe Card
[here](https://support.axelera.ai/hc/en-us/articles/25437554693138-Metis-PCIe-Tested-Host-PCs).

## Fixed Issues

- Incorrect Board Controller Firmware version returned (SDK-6710)
- Loading a model to DDR may fail (SDK-6708)
- Fixed unresponsive board controller shell issues (SDK-6803)

## Known Issues & Limitations

- **Higher RAM required for compiling** Real-ESRGAN-x4plus  
  Compiling the model Real-ESRGAN-x4plus requires a machine with at
  least 128GB of memory.
- **Device monitoring with AxMonitor is not supported on
  single-MSI hosts (SDK-6581)**  
  For some systems with single-MSI hosts device monitoring with
  AxMonitor does not display any data. An example of a host with this
  issue is Arduino Portenta X8 Mini.
- **Built operators are not retained in a docker environment
  (SDK-5228)**  
  On entering a docker container, it is required to run `make
  operators`. The built Axelera operators are not retained after
  exiting the docker. As a workaround, run `make operators` each time
  when entering a docker container.
