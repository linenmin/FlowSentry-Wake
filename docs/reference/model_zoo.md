![](/docs/images/Ax_Page_Banner_2500x168_01.png)

# Voyager model zoo

- [Voyager model zoo](#voyager-model-zoo)
  - [Querying the supported models and pipelines](#querying-the-supported-models-and-pipelines)
  - [Working with models trained on non-redistributable datasets](#working-with-models-trained-on-non-redistributable-datasets)
  - [Supported models and performance characteristics](#supported-models-and-performance-characteristics)
    - [Image Classification](#image-classification)
    - [Object Detection](#object-detection)
    - [Semantic Segmentation](#semantic-segmentation)
    - [Instance Segmentation](#instance-segmentation)
    - [Keypoint Detection](#keypoint-detection)
    - [Depth Estimation](#depth-estimation)
    - [License Plate Recognition](#license-plate-recognition)
    - [Image Enhancement Super Resolution](#image-enhancement-super-resolution)
    - [Large Language Model (LLM)](#large-language-model-llm)
  - [Next Steps](#next-steps)

The Voyager model zoo provides a comprehensive set of industry-standard models for common tasks
such as classification, object detection, segmentation and keypoint detection. It also provides
examples of pipelines that utilize these models in different ways.

The Voyager SDK makes it easy to
[deploy](/docs/reference/deploy.md) and [evaluate](/docs/reference/inference.md)
any model or pipeline on the command-line. Furthermore, most model YAML files can be modified to
replace the default weights with your own [pretrained weights](/docs/tutorials/custom_weights.md).
Pipeline YAML files can be modified to replace any model with any other model with the same task
type.

## Querying the supported models and pipelines

To view a list of all models and pipelines supported by the current release of the Voyager SDK,
type the following command from the root of the Voyager SDK repository:

```bash
make
```

The Voyager SDK outputs information similar to the example fragment below.


```yaml
ZOO
  yolov8n-coco-onnx                yolov8n ultralytics v8.1.0, 640x640 (COCO), anchor free model
  ...
REFERENCE APPLICATION PIPELINES
  yolov8sseg-yolov8lpose           Cascade example - yolov8sseg cascaded into yolov8lpose
  ...
TUTORIALS
  t1-simplest-onnx                 ONNX Tutorial-1 - An example demonstrating how to deploy an ONNX
                                   model with minimal effort. The compiled model, located at
                                   build/t1-simplest-onnx/model1/1/model.json, can be utilized in
                                   AxRuntime to create your own pipeline.
  ...

```

The `MODELS` section lists all the basic models supported from the model zoo.

The `REFERENCE APPLICATION PIPELINES` section includes examples of more complex pipelines such as
[model cascading](/docs/tutorials/cascaded_model.md) and object tracking.

The `TUTORIALS` section provides examples referred to by the
[model deployment tutorials](/ax_models/tutorials/general/tutorials.md),
which covers many aspects of model deployment and evaluation.

You can build and run most models with a single command, for example:

```bash
./inference.py yolov8n-coco-onnx usb:0
```

This command first downloads and compiles the yolov8n-coco-onnx PyTorch model from the model zoo,
if necessary, and then runs the compiled model on an available Metis device using a USB camera as
input.

Axelera also provides precompiled versions of many models, which helps reduce deployment time on
many systems
with limited performance and memory. To use a precompiled model, first download it with a command
such as:

```bash
./download_prebuilt.py yolov8n-coco-onnx
```

Further introductory information on how to run and evaluate models on Metis hardware can be
found in the [quick start guide](/docs/tutorials/quick_start_guide.md).


## Working with models trained on non-redistributable datasets

Axelera provides pre-compiled binaries for most models, which you can use directly in inferencing
applications. Access to the dataset used to train or validate the model is required only when
compiling an ML model from source or validating and verifying the accuracy of a compiled model.

In most cases, running either [`deploy.py`](/deploy.py) or
[`inference.py`](/inference.py) with the `dataset` input option will download the
required dataset to your system, if it is not already present.
The compiler uses the dataset's validation images or representative images to calibrate
quantization, while the evaluation abilities use the dataset's test images to calculate model
accuracy.

Not all industry-standard models are trained using datasets that are publicly
redistributable. In these cases, you may need to register directly with the dataset provider
and download the dataset manually. The Voyager SDK raises an error if the dataset is
missing when needed, providing you with the expected location on your system and any
data preparation steps required. The table below summarises the datasets that require manual
download.

| Dataset  | Archive | Download location |
| :------- | :------ | :---- |
| [Cityscapes (val)](https://www.cityscapes-dataset.com/) | `gtFine_val.zip` | `data/cityscapes` |
| [Cityscapes (val)](https://www.cityscapes-dataset.com/) | `leftImg8bit_val.zip` | `data/cityscapes` |
| [Cityscapes (test)](https://www.cityscapes-dataset.com/) | `gtFine_test.zip` | `data/cityscapes` |
| [Cityscapes (test)](https://www.cityscapes-dataset.com/) | `leftImg8bit_test.zip` | `data/cityscapes` |
| [ImageNet (train)](https://www.image-net.org/download.php) | `ILSVRC2012_devkit_t12.tar.gz`  | `data/ImageNet` |
| [ImageNet (train)](https://www.image-net.org/download.php) | `ILSVRC2012_img_train.tar`  | `data/ImageNet` |
| [ImageNet (val)](https://www.image-net.org/download.php) | `ILSVRC2012_devkit_t12.tar.gz`  | `data/ImageNet` |
| [ImageNet (val)](https://www.image-net.org/download.php) | `ILSVRC2012_img_val.tar`  | `data/ImageNet` |
| WiderFace (train) | `widerface_train.zip` | `data/widerface` |
| WiderFace (val) | `widerface_val.zip` | `data/widerface` |

You are responsible for adhering to all terms and conditions of the dataset licenses.

## Supported models and performance characteristics

The tables below list all model zoo models supported by this release of the Voyager SDK. The models
are categorised by task type (such as classification or object detection) and the tables provide 
information including the accuracy of the original FP32 model, the accuracy loss following
compilation and quantization (FP32 accuracy minus Quantized model accuracy), and the host
throughput in frames per second (FPS) which is measured from the host side when running inference
on the following reference platform:

* Intel Core i9-13900K CPU with Metis 1x PCIe card
* Intel Core i5-1145G7E CPU with Metis 1x M.2 card

The accuracy for each model on Metis is determined using a pipeline where the pre-processing and post-processing elements are implemented using PyTorch/torchvision:

`inference.py <model> dataset --pipe=torch-aipu --no-display`

Because most models are originally trained using pre-processing and post-processing code implemented in PyTorch, this pipeline configuration most accurately isolates the quantization loss introduced by Metis, independent of the host, thereby enabling like-for-like comparison with other AI accelerators.

`inference.py <model> media/traffic2_720p.mp4 --pipe=gst --no-display`

This command measures both the host frame rate and end-to-end frame rate. The input video is h.264-encoded 720p consistent with many real-world deployments. The tables below report the host frame rate, thereby enabling like-for-like comparison with other AI accelerators. You can also modify the above command with different video sources to measure the end-to-end performance for your specific use case.

Additionally, you can modify the accuracy measurement command with the flag --pipe=gst to measure the end-to-end accuracy on your target platform. To the best of our knowledge, we are the only provider offering this comprehensive end-to-end accuracy measurement. We will be publishing a dedicated blog post explaining the significance of this approach and how it differs from standard industry practices.

The [benchmarking and performance evaluation guide](/docs/tutorials/benchmarking.md) explains how
to verify these results and how to perform many other evaluation tasks on all supported platforms.

### Image Classification
| Model                                                                                      | ONNX                                                                                      | Repo                                                             | Resolution | Dataset     | Ref FP32 Top1 | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- | :--------------------------------------------------------------- | :--------- | :---------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| [EfficientNet-B0](/ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.67         | 0.78          | 1448         | 1461        | BSD-3-Clause  |
| [EfficientNet-B1](/ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.60         | 0.46          | 979          | 982         | BSD-3-Clause  |
| [EfficientNet-B2](/ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.79         | 0.42          | 904          | 872         | BSD-3-Clause  |
| [EfficientNet-B3](/ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.54         | 0.42          | 784          | 729         | BSD-3-Clause  |
| [EfficientNet-B4](/ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 79.27         | 0.83          | 555          | 458         | BSD-3-Clause  |
| [MobileNetV2](/ax_models/zoo/torchvision/classification/mobilenetv2-imagenet.yaml)         | [&#x1F517;](/ax_models/zoo/torchvision/classification/mobilenetv2-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 71.87         | 1.19          | 3608         | 3594        | BSD-3-Clause  |
| [MobileNetV4-small](/ax_models/zoo/timm/mobilenetv4_small-imagenet.yaml)                   | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_small-imagenet-onnx.yaml)                     | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 224x224    | ImageNet-1K | 73.74         | 3.75          | 5086         | 4858        | Apache 2.0    |
| [MobileNetV4-medium](/ax_models/zoo/timm/mobilenetv4_medium-imagenet.yaml)                 | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_medium-imagenet-onnx.yaml)                    | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 224x224    | ImageNet-1K | 79.04         | 1.08          | 2501         | 2409        | Apache 2.0    |
| [MobileNetV4-large](/ax_models/zoo/timm/mobilenetv4_large-imagenet.yaml)                   | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_large-imagenet-onnx.yaml)                     | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 384x384    | ImageNet-1K | 82.92         | 0.72          | 731          | 466         | Apache 2.0    |
| [MobileNetV4-aa_large](/ax_models/zoo/timm/mobilenetv4_aa_large-imagenet.yaml)             | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_aa_large-imagenet-onnx.yaml)                  | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 384x384    | ImageNet-1K | 83.22         | 2.24          | 610          | 406         | Apache 2.0    |
| [SqueezeNet 1.0](/ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet.yaml)    | [&#x1F517;](/ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 58.10         | 3.22          | 997          | 836         | BSD-3-Clause  |
| [SqueezeNet 1.1](/ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet.yaml)    | [&#x1F517;](/ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 58.19         | 2.41          | 7414         | 7122        | BSD-3-Clause  |
| [ResNet-18](/ax_models/zoo/torchvision/classification/resnet18-imagenet.yaml)              | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet18-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 69.76         | 0.34          | 4052         | 3752        | BSD-3-Clause  |
| [ResNet-34](/ax_models/zoo/torchvision/classification/resnet34-imagenet.yaml)              | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet34-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 73.30         | 0.75          | 2285         | 2233        | BSD-3-Clause  |
| [ResNet-50 v1.5](/ax_models/zoo/torchvision/classification/resnet50-imagenet.yaml)         | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet50-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 76.15         | 0.47          | 1982         | 1894        | BSD-3-Clause  |
| [ResNet-101](/ax_models/zoo/torchvision/classification/resnet101-imagenet.yaml)            | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet101-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.37         | 0.71          | 1047         | 682         | BSD-3-Clause  |
| [ResNet-152](/ax_models/zoo/torchvision/classification/resnet152-imagenet.yaml)            | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet152-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.31         | 0.21          | 474          | 265         | BSD-3-Clause  |
| [ResNet-10t](/ax_models/zoo/timm/resnet10t-imagenet.yaml)                                  | [&#x1F517;](/ax_models/zoo/timm/resnet10t-imagenet-onnx.yaml)                             | [&#x1F517;](https://huggingface.co/timm/resnet10t.c3_in1k)       | 224x224    | ImageNet-1K | 68.22         | 1.75          | 5499         | 5025        | Apache 2.0    |

### Object Detection
| Model                                                                           | ONNX                                                                                       | Repo                                                                                                        | Resolution | Dataset   | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :--------- | :-------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| RetinaFace - Resnet50                                                           | [&#x1F517;](/ax_models/zoo/torch/retinaface-resnet50-widerface-onnx.yaml)                  | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 840x840    | WiderFace | 95.25        | 0.20          | 88           | 50          | MIT           |
| RetinaFace - mb0.25                                                             | [&#x1F517;](/ax_models/zoo/torch/retinaface-mobilenet0.25-widerface-onnx.yaml)             | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 640x640    | WiderFace | 89.44        | 0.99          | 903          | 756         | MIT           |
| SSD-MobileNetV1                                                                 | [&#x1F517;](/ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv1-coco-poc-onnx.yaml) | [&#x1F517;](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | 300x300    | COCO2017  | 25.70        | 0.20          | 3351         | 2969        | Apache 2.0    |
| SSD-MobileNetV2                                                                 | [&#x1F517;](/ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv2-coco-poc-onnx.yaml) | [&#x1F517;](https://github.com/tensorflow/models)                                                           | 300x300    | COCO2017  | 19.21        | 0.61          | 2280         | 2180        | Apache 2.0    |
| YOLOv3                                                                          | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov3-coco-onnx.yaml)                    | [&#x1F517;](https://github.com/ultralytics/yolov3)                                                          | 640x640    | COCO2017  | 46.61        | 0.87          | 157          | 104         | AGPL-3.0      |
| [YOLOv5s-Relu](/ax_models/zoo/yolo/object_detection/yolov5s-relu-coco.yaml)     | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-relu-coco-onnx.yaml)              | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 35.09        | 0.51          | 768          | 539         | AGPL-3.0      |
| [YOLOv5s-v5](/ax_models/zoo/yolo/object_detection/yolov5s-v5-coco.yaml)         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-v5-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 36.18        | 0.39          | 757          | 535         | AGPL-3.0      |
| [YOLOv5n](/ax_models/zoo/yolo/object_detection/yolov5n-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5n-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 27.72        | 0.84          | 984          | 666         | AGPL-3.0      |
| [YOLOv5s](/ax_models/zoo/yolo/object_detection/yolov5s-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 37.25        | 0.73          | 887          | 805         | AGPL-3.0      |
| [YOLOv5m](/ax_models/zoo/yolo/object_detection/yolov5m-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5m-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 44.94        | 0.72          | 449          | 326         | AGPL-3.0      |
| [YOLOv5l](/ax_models/zoo/yolo/object_detection/yolov5l-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5l-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 48.67        | 0.91          | 271          | 177         | AGPL-3.0      |
| [YOLOv7](/ax_models/zoo/yolo/object_detection/yolov7-coco.yaml)                 | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-coco-onnx.yaml)                    | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x640    | COCO2017  | 51.02        | 0.62          | 208          | 170         | GPL-3.0       |
| [YOLOv7-tiny](/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco.yaml)       | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco-onnx.yaml)               | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 416x416    | COCO2017  | 33.12        | 0.43          | 1421         | 1104        | GPL-3.0       |
| [YOLOv7 640x480](/ax_models/zoo/yolo/object_detection/yolov7-640x480-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-640x480-coco-onnx.yaml)            | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x480    | COCO2017  | 50.78        | 0.54          | 236          | 163         | GPL-3.0       |
| [YOLOv8n](/ax_models/zoo/yolo/object_detection/yolov8n-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 37.12        | 1.29          | 826          | 754         | AGPL-3.0      |
| [YOLOv8s](/ax_models/zoo/yolo/object_detection/yolov8s-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 44.80        | 0.93          | 641          | 523         | AGPL-3.0      |
| [YOLOv8m](/ax_models/zoo/yolo/object_detection/yolov8m-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 50.16        | 2.04          | 239          | 176         | AGPL-3.0      |
| [YOLOv8l](/ax_models/zoo/yolo/object_detection/yolov8l-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 52.83        | 1.16          | 181          | 141         | AGPL-3.0      |
| YOLOX-s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolox-s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017  | 39.24        | 0.21          | 625          | 416         | Apache-2.0    |
| YOLOX-m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolox-m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017  | 46.26        | 0.27          | 344          | 264         | Apache-2.0    |
| YOLOv9t                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9t-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 37.81        | 1.36          | 402          | 246         | AGPL-3.0      |
| YOLOv9s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 46.28        | 2.60          | 368          | 235         | AGPL-3.0      |
| YOLOv9m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 51.24        | 1.29          | 200          | 147         | AGPL-3.0      |
| YOLOv9c                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9c-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 52.67        | 2.46          | 191          | 143         | AGPL-3.0      |
| YOLO11n                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 39.17        | 0.72          | 745          | 573         | AGPL-3.0      |
| YOLO11s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 46.54        | 0.62          | 550          | 422         | AGPL-3.0      |
| YOLO11m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 51.31        | 0.50          | 264          | 194         | AGPL-3.0      |
| YOLO11l                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 53.23        | 0.36          | 180          | 123         | AGPL-3.0      |
| YOLO11x                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11x-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 54.67        | 0.41          | 53           | 31          | AGPL-3.0      |

### Semantic Segmentation
| Model                                                                    | ONNX                                                                      | Repo                                                                             | Resolution | Dataset    | Ref FP32 mIoU | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------- | :------------------------------------------------------------------------ | :------------------------------------------------------------------------------- | :--------- | :--------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| U-Net FCN 256                                                            | [&#x1F517;](/ax_models/zoo/mmlab/mmseg/unet_fcn_256-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 256x256    | Cityscapes | 57.75         | 0.23          | 241          | 190         | Apache 2.0    |
| [U-Net FCN 512](/ax_models/zoo/mmlab/mmseg/unet_fcn_512-cityscapes.yaml) |                                                                           | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 512x512    | Cityscapes | 66.62         | 0.19          | 33           | 18          | Apache 2.0    |

### Instance Segmentation
| Model                                                                         | ONNX                                                                             | Repo                                                    | Resolution | Dataset  | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| [YOLOv8n-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 54.12        | 1.84          | 644          | 426         | AGPL-3.0      |
| [YOLOv8s-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 63.13        | 1.40          | 482          | 341         | AGPL-3.0      |
| [YOLOv8l-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 70.50        | 1.84          | 155          | 118         | AGPL-3.0      |
| YOLO11n-seg                                                                   | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolo11nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 56.49        | 1.30          | 578          | 399         | AGPL-3.0      |
| YOLO11l-seg                                                                   | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolo11lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 71.76        | 0.18          | 151          | 107         | AGPL-3.0      |

### Keypoint Detection
| Model                                                                        | ONNX                                                                           | Repo                                                    | Resolution | Dataset  | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| [YOLOv8n-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 51.11        | 2.32          | 815          | 719         | AGPL-3.0      |
| [YOLOv8s-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 60.65        | 2.55          | 589          | 466         | AGPL-3.0      |
| [YOLOv8l-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 68.39        | 1.90          | 181          | 137         | AGPL-3.0      |
| YOLO11n-pose                                                                 | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolo11npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 51.15        | 3.96          | 723          | 525         | AGPL-3.0      |
| YOLO11l-pose                                                                 | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolo11lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 67.44        | 3.55          | 177          | 122         | AGPL-3.0      |

### Depth Estimation
| Model     | ONNX                                                             | Repo                                                                              | Resolution | Dataset    | Ref FP32 RMSE | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :-------- | :--------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------- | :--------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| FastDepth | [&#x1F517;](/ax_models/zoo/torch/fastdepth-nyudepthv2-onnx.yaml) | [&#x1F517;](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/146_FastDepth) | 224x224    | NYUDepthV2 | 0.6574        | -0.0034       | 971          | 871         | MIT           |

### License Plate Recognition
| Model                                      | ONNX | Repo                                                     | Resolution | Dataset       | Ref FP32 WLA | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------- | :--- | :------------------------------------------------------- | :--------- | :------------ | -----------: | ------------: | -----------: | ----------: | ------------: |
| [LPRNet](/ax_models/zoo/torch/lprnet.yaml) |      | [&#x1F517;](https://github.com/sirius-ai/LPRNet_Pytorch) | 94x24      | LPRNetDataset | 89.40        | 2.70          | 9690         | 9162        | Apache-2.0    |

### Image Enhancement Super Resolution
| Model              | ONNX                                                           | Repo                                                | Resolution | Dataset                         | Ref FP32 PSNR | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------- | :------------------------------------------------------------- | :-------------------------------------------------- | :--------- | :------------------------------ | ------------: | ------------: | -----------: | ----------: | ------------: |
| Real-ESRGAN-x4plus | [&#x1F517;](/ax_models/zoo/torch/real-esrgan-x4plus-onnx.yaml) | [&#x1F517;](https://github.com/xinntao/Real-ESRGAN) | 128x128    | SuperResolutionCustomSet128x128 | 24.77         | 0.22          | 11           | 8           | BSD-3-Clause  |

### Large Language Model (LLM)
For details of usage please see [SLM Inference on Axelera AI Platform](/docs/tutorials/llm.md).

| Model                                                                                      | Max Context Window (tokens) |
| :----------------------------------------------------------------------------------------- | --------------------------: |
| [meta-llama/Llama-3.1-8B-Instruct](/ax_models/zoo/llm/llama-3-1-8b-1024-static.yaml)       | 1024                        |
| [meta-llama/Llama-3.2-1B-Instruct](/ax_models/zoo/llm/llama-3-2-1b-1024-4core-static.yaml) | 1024                        |
| [meta-llama/Llama-3.2-3B-Instruct](/ax_models/zoo/llm/llama-3-2-3b-1024-4core-static.yaml) | 1024                        |
| [microsoft/Phi-3-mini-4k-instruct](/ax_models/zoo/llm/phi3-mini-512-static.yaml)           | 512                         |
| [microsoft/Phi-3-mini-4k-instruct](/ax_models/zoo/llm/phi3-mini-1024-4core-static.yaml)    | 1024                        |
| [microsoft/Phi-3-mini-4k-instruct](/ax_models/zoo/llm/phi3-mini-2048-static.yaml)          | 2048                        |
| [Almawave/Velvet-2B](/ax_models/zoo/llm/velvet-2b-1024-static.yaml)                        | 1024                        |

## Next Steps

You can quickly experiment with any of the above models following the
[quick start guide](/docs/tutorials/quick_start_guide.md), and replacing the name of the model in
the example commands given.

You can also evaluate your own pretrained weights for most model zoo models by following the
[custom weights tutorial](/docs/tutorials/custom_weights.md).
