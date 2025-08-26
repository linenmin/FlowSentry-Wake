![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Voyager SDK release notes v1.4.0

- [Voyager SDK release notes v1.4.0](#voyager-sdk-release-notes-v140)
  - [Release Description](#release-description)
    - [Release Qualification](#release-qualification)
    - [Host Environment](#host-environment)
  - [New Features / Support](#new-features--support)
    - [New Axelera AI Cards and Systems](#new-axelera-ai-cards-and-systems)
    - [New Platforms](#new-platforms)
    - [New Networks Supported](#new-networks-supported)
      - [Object Detection](#object-detection)
      - [Semantic Segmentation](#semantic-segmentation)
      - [Face Recognition](#face-recognition)
      - [Re-identification](#re-identification)
    - [End-to-End Pipelines](#end-to-end-pipelines)
  - [New Features](#new-features)
    - [AI Pipeline Builder](#ai-pipeline-builder)
    - [Model Compiler](#mark\\[Beta\\]-model-compiler)
    - [Runtime](#runtime)
    - [Tools](#tools)
    - [Firmware](#firmware)
  - [Fixed Issues since Last Release](#fixed-issues-since-last-release)
  - [Known Issues and Limitations](#known-issues-and-limitations)
  - [Further Support](#further-support)

## Release Description
This release expands the capabilities of Voyager SDK with new features, platform support and AI use cases. It is a public release which follows the same installation and upgrade principles as v1.3, as well as the same licensing model. It adds the following capabilities:

* Support for new models including YOLO10, FastSAM, FaceNet, OSNet and Deep-OC-Sort
* Support for new AI use cases such as face recognition and person re-identification
* Python API for invoking the compiler
* Performance improvements on Windows platforms
* Command-line tool to query the SDK version
* Stability fixes
* Simplified installer with the requirement for a token removed

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

Compiled models are expected to run on Intel Core-i 12th and 13th generations (x86), AMD Ryzen (x86)
and ARM64 host CPUs. Please find the list of platforms Axelera AI has tested with Metis M.2 Card [here](https://support.axelera.ai/hc/en-us/articles/25437844422418-Metis-M-2-Tested-Host-PCs)
and Metis PCIe Card [here](https://support.axelera.ai/hc/en-us/articles/25437554693138-Metis-PCIe-Tested-Host-PCs).

## New Features / Support

### New Axelera AI Cards and Systems
- Support for Metis Compute Board (AI SBC) AIPU, 4GB of RAM

### New Platforms
- NVIDIA Jetson Orin NX (Arm Cortex-A78AE)

### New Networks Supported

The Voyager Model Zoo includes models for Computer Vision and Large Language Models. For a full list of supported models and data about their performance and accuracy see [here](/docs/reference/model_zoo.md).

Models that are supported but not included in the model zoo are documented [here](/docs/reference/additional_models.md).

> [!NOTE]
> The `download_prebuilt.py` utility for downloading pre-compiled models has been deprecated in favor of `axdownloadmodel`.

#### Object Detection

| Model Name                                                              | Resolution | Format |
| :---------------------------------------------------------------------- | :--------- | :----- |
| [YOLO10n](/ax_models/zoo/yolo/object_detection/yolov10n-coco-onnx.yaml) | 640x640    | ONNX   |
| [YOLO10s](/ax_models/zoo/yolo/object_detection/yolov10s-coco-onnx.yaml) | 640x640    | ONNX   |
| [YOLO10b](/ax_models/zoo/yolo/object_detection/yolov10b-coco-onnx.yaml) | 640x640    | ONNX   |

#### Semantic Segmentation

| Model Name                                                              | Resolution | Format |
| :---------------------------------------------------------------------- | :--------- | :----- |
| [FastSAM](/ax_models/reference/apps/fastsam/fastsams-rn50x4-onnx.yaml)   | 640x640    | ONNX   |

#### Face Recognition
| Model Name                                                           | Resolution | Format          |
| :------------------------------------------------------------------- | :--------- | :-------------- |
| [FaceNet - InceptionResnetV1](/ax_models/zoo/torch/facenet-lfw.yaml) | 160x160    | Pytorch, ONNX   |

#### Re-identification
| Model Name                                                           | Resolution | Format |
| :------------------------------------------------------------------- | :--------- | :----- |
| [OSNet x1_0](/ax_models/zoo/torch/osnet-x1-0-market1501-onnx.yaml)   | 256x128    |  ONNX  |
| [Deep-OC-Sort](/ax_models/reference/cascade/with_tracker/yolox-deep-oc-sort-osnet.yaml) (combined with YOLOX but can be adapted to other object detectors) | 384x128    |  ONNX  |

### End-to-End Pipelines
- End-to-end application for _segment anything_ using FastSAM
- New YAML files for all new models offered in our model zoo in this release (see tables above)

## New Features

### Installation
- The installer has been updated to remove the requirement of token-based authentication. The
SDK can now be installed without providing a user identity (username/token)

### AI Pipeline Builder
- An InferenceStream can be created on multiple different pipelines, and pipelines can be
added dynamically using `add_pipeline()`
- AxInferenceNet supports Top-k Region-Of-Interest (ROI) filtering based on area, size and classes
- Classification metadata has been simplified to make classification metadata objects easier to use (see
the [classification example](/examples/classification_example.py) for further details)

### [Beta] Model Compiler
- Support for a simplified [compilation API in Python](/docs/reference/compiler_api.md) based on two API calls:
`quantize()` and `compile()`. Compiled models can be run using the [AxRunModel tool](docs/reference/axrunmodel.md)
or [AxRuntime inference API](/docs/reference/axelera.runtime.md)
- Support for concat along any non-batch axis of a 4d tensor, and grouped convolutions for symmetric kernels (kernel width equals kernel height)
- The list of [supported operators](/docs/reference/onnx-opset17-support.md) will grow in future releases.
For technical assistance on compiling your own model please turn to the [Axelera Community](https://community.axelera.ai/)

### Runtime
- Significant performance improvement for several models using AxRuntime inference API on Windows platforms
- Support for multiple Axelera cards on Windows platforms

### Tools
- New `axversion` command-line utility to query the version of the SDK running
- Improvements to `axmonitor`
  * New metrics - power usage for cards that feature on-board power sensors, kernels per second
  * Added multi-device support
  * Included configuration and threshold settings information
  * Added monitoring messages for hardware throttling, software throttling and version
- Improvements to `axdevice`
  * Improved multi-device handling and PCIe bridge management 
  * Added reboot and thermal commands

### Firmware
- Improved version checking commands for board controller
- Improved resiliency and stability across different host platforms
- [Updated firmware upgrade script](/docs/tutorials/firmware_flash_update.md)

## Fixed Issues Since Last Release
- Incorrect Board Controller Firmware version returned (SDK-6710)
- Loading a model to DDR may fail (SDK-6708)
- Fixed unresponsive board controller shell issues (SDK-6803)

## Known Issues and Limitations
- Compiling the model Real-ESRGAN-x4plus requires a machine with at least 128GB of system RAM
- Device monitoring with `AxMonitor` is not supported on single-MSI hosts (SDK-6581): For some systems
with single-MSI, host device monitoring with `AxMonitor` does not display any data. An example of
a host with this issue is Arduino Portena X8 Mini.
- Built operators are not retained in a docker environment (SDK-5228): As a workaround, on entering
a docker container you must run `make operators`. 

## Further Support
For blog posts, projects and technical support please visit [Axelera AI Customer Portal](https://support.axelera.ai)
For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).
