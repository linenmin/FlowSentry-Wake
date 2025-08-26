![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Application integration tutorial

- [Application integration tutorial](#application-integration-tutorial)
  - [Example computer vision pipeline](#example-computer-vision-pipeline)
  - [Simple application integration example](#simple-application-integration-example)
  - [Advanced features](#advanced-features)
    - [Tracing](#tracing)
    - [Hardware caps](#hardware-caps)
    - [Frame rate control](#frame-rate-control)
    - [RTSP latency](#rtsp-latency)
  - [Customizing visualisation](#customizing-visualisation)
    - [Visualisation without metadata](#visualisation-without-metadata)
  - [Using Raw Tensor Output in Your Application](#using-raw-tensor-output-in-your-application)
    - [Pipeline configuration for raw tensor output](#pipeline-configuration-for-raw-tensor-output)
      - [Performance optimization (advanced)](#performance-optimization-advanced)
    - [Performance note](#performance-note)
    - [Example: Extracting the tensor in Python](#example-extracting-the-tensor-in-python)
  - [List of example applications](#list-of-example-applications)


Voyager application integration APIs cleanly separate the task of developing computer vision
pipelines from the task of developing inferencing applications based on these pipelines.

## Example computer vision pipeline

A Voyager YAML file describes an inferencing pipeline including image pre-processing,
a deep learning model and post-processing. This tutorial is based on a
[YOLOv5m-based pipeline with tracker](/ax_models/reference/cascade/with_tracker/yolov5m-v7-coco-tracker.yaml),
but it can be applied to any YAML pipeline that outputs tracked bounding boxes.
The code fragment below shows the complete pipeline definition.

```yaml

name: yolov5m-v7-coco-tracker

pipeline:
  - detections:
      model_name: yolov5m
      preprocess:
        - letterbox:
            width: 640
            height: 640
        - torch-totensor:
      postprocess:
        - decodeyolo:
            box_format: xywh
            normalized_coord: False
            label_filter: ['car', 'truck', 'motorcycle', 'bus', 'person', 'bicycle', 'cat', 'dog']
            conf_threshold: 0.3
            nms_iou_threshold: 0.5
        - tracker:
            algorithm: oc-sort
            history_length: 30
```

The pipeline includes a single YOLOv5m model and OC-SORT tracker. The input images are first pre-processed
by letterboxing and are then converted to tensor format required by the model. The output tensors generated
by the model are decoded to Voyager bounding box metadata for use
with Voyager libraries for tracking, visualising detections and performing application-level analysis.
The fields `label_filter`, `conf_threshold` and `nsm_iou_threshold` are all used to limit the
number of detections output from pipeline. Their values represent initial default values, which can
be subsequently modified at runtime by the application.

The Voyager toolchain generates optimized implementations of YAML pipelines for a target system
comprising a host processor and one or more Metis devices. You can use the application integration APIs
to configure and run pipelines from either Python or C++ applications and by writing only a few lines of code.

YAML pipelines do not include input operators. Instead, you specify your required 
[video sources](/docs/tutorials/video_sources.md) at runtime when instantiating a pipeline
in your application.


## Simple application integration example

The file [application.py](/examples/application.py) shows how to configure a pipeline
with multiple video input sources and then run it, obtaining a sequence of images and inference metadata.
The application renders the results visually in a window and outputs a basic analysis of
all tracked vehicles to the terminal.

The first few lines of code import the Voyager application integration libraries needed by this example.

```python
from axelera.app import config, display
from axelera.app.stream import create_inference_stream
```

The application defines a *stream* comprising the YOLOv5m-based pipeline and two input video files.

```python
stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=[
        str(config.env.framework / "media/traffic1_1080p.mp4"),
        str(config.env.framework / "media/traffic2_1080p.mp4"),
    ],
)
```

The first argument to `create_inference_stream` is the name of the pipeline as it appears in the YAML file (field: `name`).
If your YAML file is located outside of the Voyager SDK repository, you can provide an absolute path to it instead.

The second argument is an array of input sources, in this case two video files in the `media` directory of the
Voyager SDK repository. (The variable `config.env.framework` is the path to the Voyager repository for the activated environment.)
Many different input sources are supported.

| Source | Description | Example |
| :----- | :---------- | :------ |
| `/path/to/file` | [Path to an image or video file](/docs/tutorials/video_sources.md#local-files) | `str(config.env.framework / "media/traffic1_1080p.mp4")` |
| `usb:<device_id>` | [USB camera](/docs/tutorials/video_sources.md#usb-cameras) | `usb:0` |
| `rtsp://<ip_address>:<port>/<stream_name>` | [RTSP camera](/docs/tutorials/video_sources.md#rtsp-cameras) | `rtsp://user:pwd@127.0.0.1:8554/1` |

You can freely mix and match different video sources and input formats. Using the same type of source, color format and video resolution often results in the highest possible performance.

> [!TIP]
> You can download the sample videos used in this tutorial by running the command `./install.sh --media` from the root of the Voyager SDK repository.

The application creates a window to display rendered inference results. For performance reasons,
it runs the pipeline in a separate thread from the main application.
A final call to `stream.stop()` is needed to terminate the pipeline and
release all of its allocated resources.

```python
with display.App(visible=True) as app:
    wnd = app.create_window("Business logic demo", (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run()
stream.stop()
```

The main application arranges the display window with two tiles, one for each input video.
It defines a `VEHICLE` as a list of class labels, defines the function `center`
for analyzing object movement with center coordinates, and then starts iterating over
the inference stream using Voyager libraries to analyze and visualise the results.

```python
def main(window, stream):
    window.options(0, title="Traffic 1")
    window.options(1, title="Traffic 2")

    VEHICLE = ('car', 'truck', 'motorcycle')
    center = lambda box: ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
        for veh in frame_result.detections:
            if veh.is_a(VEHICLE):
                print(
                    f"{veh.label.name} {veh.track_id}: {center(veh.history[0])} → {center(veh.history[-1])} @ stream {frame_result.stream_id}"
                )
```

The `stream` object is an iterator that yields a `frame_result` for each input frame. This object contains:

- `frame_result.image`: The input source image at its original resolution
- `frame_result.meta`: All inference metadata
- `frame_result.detections`: The name `detections` is the name defined in the YAML pipeline and provides convenient access to detection results
- `frame_result.stream_id`: The input stream id that produced this result

The method `window.show()` overlays the frame metadata on the original image and renders the result.

The helper function `is_a()` makes it easy to filter objects by category. In this example, 
`veh.is_a(VEHICLE)` determines if a detection belongs to any of the defined vehicle categories.

Each detection includes tracking information with position history, making it easy to analyze object
movement with center coordinates. Sample output from running this application is shown below.

```
car 2: (398, 312) → (415, 312) @ stream 1
truck 70028: (720, 311) → (754, 310) @ stream 1
car 70027: (1647, 1039) → (1644, 1040) @ stream 0
```

> [!TIP]
> You can also implement your own methods to manipulate and display images. To obtain
> a PIL Image object, call `frame_result.image.aspil()`, and call `frame_result.image.color_format`
> to determine its color format.
> To obtain the image as a NumPy array use `frame_result.image.asarray()` optionally specifying the required color
> format as an argument (`RGB`, `BGR`, `GRAY` or `BGRA`). If unspecified, the colour format is determined
> based on the input video.

## Advanced features

The file [application_extended.py](/examples/application_extended.py) uses a number of
advanced integration features.

The first few lines of code import the Voyager application integration libraries needed by this example.

```python
import time

from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.stream import create_inference_stream

framework = config.env.framework
```

The application creates a YOLOv5m inferencing pipeline and with a number of advanced
options configured.

```python
tracers = inf_tracers.create_tracers('core_temp', 'end_to_end_fps', 'cpu_usage')

stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=[
        str(framework / "media/traffic1_1080p.mp4"),
        str(framework / "media/traffic2_1080p.mp4"),
    ],
    pipe_type='gst',
    log_level=logging_utils.INFO,  # INFO, DEBUG, TRACE
    hardware_caps=config.HardwareCaps(
        vaapi=config.HardwareEnable.detect,
        opencl=config.HardwareEnable.detect,
        opengl=config.HardwareEnable.detect,
    ),
    tracers=tracers,
    specified_frame_rate=10,
    # rtsp_latency=500,
)
```

The `pipe_type` argument to `create_inference_stream` specifies the type of pipeline
to build and run. The default `gst` option represents a pipeline running end-to-end
across the host processor and Metis deice. Other supported values include 
`torch` and `torch-aipu` (see [`--pipe` option](/docs/reference/deploy.md) for further
details).

The `log_level` argument to `create_inference_stream` controls the verbosity of output
to the terminal. The default verbosity level is `INFO` while `DEBUG` and `TRACE` levels
provide additional information useful for debugging.

> [!TIP]
> All options supported by [`inference.py`](/docs/reference/inference.md) can also be passed to
> `create_inference_stream`, enabling you to easily switch between between evaluation and
> development integration between evaluation and development environments.

### Tracing

Tracers provide real-time metrics that help you better understand
device resource utilization, performance and thermal characteristics.

The function `create_tracers` shown in the example below takes a list of metrics as its arguments and returns a list of
tracers to provide the `tracers` argument in the function `create_inference_stream`.  The available tracers are:

* `end_to_end_fps` - collects end-to-end fps metric
* `core_temp` - collects the temperature of the metis core
* `cpu_usage` - collects information about the host CPU usage
* `stream_timing`  - collects metrics about stream latency and jitter

The main application periodically queries these tracers using the method `stream.get_all_metrics()` and prints
the returned metrics to the terminal.

```python
def main(window, stream):
    # ... other code
    for frame_result in stream:
        # Process frames...
        
        # Get current metrics
        metrics = stream.get_all_metrics()
        core_temp = metrics['core_temp']
        end_to_end_fps = metrics['end_to_end_fps']
        cpu_usage = metrics['cpu_usage']
        
        # Report metrics periodically
        if (now := time.time()) - last_temp_report > 1:
            last_temp_report = now
            metrics = [
                f"Core temp: {core_temp.value}°C",
                f"End-to-end FPS: {end_to_end_fps.value:.1f}",
                f"CPU usage: {cpu_usage.value:.1f}%",
            ]
            print(' | '.join(metrics))
```

### Hardware caps

The `hardware_caps` argument to `create_inference_stream` is used to enable or disable use of the host runtime acceleration
libraries, which can accelerate image pre-processing and post-processing operations on GPU hardware embedded on the host
processor. Set to `detect` for automatic detection, `enable` to force usage and `disable` to prevent usage.

### Frame rate control

The `specified_frame_rate` argument to `create_inference_stream` lets you fine-tune pipeline frame rate control behavior.

| Value | Description |
| :---- | :---------- |
| Positive integer | The pipeline produces precisely the specified frames per second, dropping or duplicating frames as needed
| `0` | The pipeline produces frames at the a rate matching the input frame rate
| `-1` | The pipeline operates in downstream-leaky mode, dropping frames if the application is unable to consume them before the next frame is ready

In general, if the application can process frames faster than the input frame rate then a value of `0` can be specified. If
not and your application loop runs with predictable time, a positive value can be specified. If your application requires
periods of intense processing, downstream-leaky mode can help prevent your pipeline queues from filling up during these periods,
which would otherwise lead to a buildup of latency and potential instabilities.

### RTSP latency

The `rtsp_latency` argument to `create_inference_stream` lets you balance system latency
and stability when working with IP network cameras.
Setting a low latency enables the application to respond quicker to detections, but
may also lead to problems such as:

- inability to smooth out network jitter resulting in packet loss
- choppy and stuttering playback during transmission delays
- poor synchronization of audio and video streams
- loss of connection due to insufficient margin for packet retransmission

In general, specifying a higher latency value such as 2000ms can help avoid these
issues under poor network conditions, while the default latency value of 500ms provides
a good tradeoff between latency and reliability in many typical network conditions.

## Customizing visualisation

The file [fruit_demo.py](/examples/demos/fruit_demo.py) shows how to configure the visualiser in a number
of different ways.

```python
def main(window, stream):
    for n, source in stream.sources.items():
        window.options(
            n,
            title=f"Video {n} {source}",
            title_size=24,
            grayscale=0.9,
            bbox_class_colors={
                'banana': (255, 255, 0, 125),
                'apple': (255, 0, 0, 125),
                'orange': (255, 127, 0, 125),
            },
        )
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
```

The method `window.options()` supports the following configuration options:

- `title`: Adds a descriptive title to the video stream
- `grayscale`: Sets a grayness level for the original image. Inference metadata such as bounding boxes and segments continue to be rendered in color
- `bbox_class_colors`: Specifies the color to be used to render each specified class label
- `bbox_label_format`: See the [`Options` class definition](/axelera/app/display.py#L75) for further details

> [!TIP]
> Consider reducing rendering overheads by calling `window.show()` only when needed, or by setting `visible=False` to
> disable rendering if not required by your application.

### Visualisation without metadata

The file [application_extended](/examples/application_extended.py) shows how to render layers,
such as text and images, in the visualiser.

```python
def main(window, stream):
    ...
    counter = window.text(
        '20px, 10%',
        "Vehicles: 00",
    )
    ...
    for frame_result in stream:
        ...
        VEHICLE = ('car', 'truck', 'motorcycle')
        vehicles = sum(d.is_a(VEHICLE) for d in frame_result.detections)
        counter["text"] = f"Vehicles: {vehicles:02}"
```

The method `window.text` creates a text box on the window, and returns a handle to the text box which can be used for future modification.

- The text box will be located at the given css-style string position `'20px, 10%'`. This means the anchor point of the text box (default top left) will be located 20 screen pixels from the left of the window, and 10% down from the top. This illustrates the two methods of providing layer locations. In absolute pixel terms, or relative to the window/stream size.
  - Note that the counter example is constructed without a `stream_id`. If a `stream_id` is provided, the layer will be rendered to the stream, not the window. As such, the positions will be relative to the indicated `stream_id`, not relative to the window.
- The text box will contain the text "Vehicles: 00" after initialization, to serve as a counter of the number of vehicles currently in the frame.

Later, when processing the stream, we update the text in the text box to whatever the current vehicle count is for this frame. This is done by setting the field by name like a dictionary, `counter["text"]` in this case. This will update the text in-place on the next rendered frame.

> [!TIP]
> You can also update fields by kwarg with the `set` method on the handle, including multiple fields at once.
> Position coordinates can be provided in tuple form, like `('20px', '10%')`.
> Other layers, such as images, may be displayed in a similar fashion.
> There are many other customisations and methods provided by the handle. See the [`Window` and `LayerHandle` class definitions](/axelera/app/display.py) for further details.


## Using Raw Tensor Output in Your Application

In some cases, you may want your pipeline to return the raw output tensor from your model, rather than ready-to-use detection objects. This is useful if:
- You are working with a custom model or a model not supported by the standard postprocessing operators.
- You want full control over postprocessing (e.g., custom decoding, NMS, or analytics).
- You want to experiment with new postprocessing algorithms or integrate with your own pipeline.

### Pipeline configuration for raw tensor output

To receive the raw tensor output that matches your original ONNX or PyTorch model, configure your YAML pipeline like this:

```yaml
inference:
  handle_all: True  # Default: produces output matching your original model
postprocess:
  - get-raw-tensor:
```

This gives you the same tensor output as your original ONNX or PyTorch model - fully dequantized, with correct dimensions and layout.

#### Performance optimization (advanced)

For improved performance, especially with smaller models, you can handle processing steps yourself:

**Option 1 - Handle nothing (maximum performance potential):**
```yaml
inference:
  handle_all: False  # You handle all processing steps yourself
postprocess:
  - get-raw-tensor:
```

**Option 2 - Fine-grained control:**
```yaml
inference:
  # Don't set handle_all, configure individual steps
  handle_dequantization_and_depadding: True
  handle_transpose: False
  handle_postamble: False
postprocess:
  - get-raw-tensor:
```

With `handle_all: False`, you take responsibility for all processing steps yourself, but this can enable additional optimizations in the pipeline.

This will produce an `.axnet` where the output meta contains a tensor, not decoded detections.

### Performance note

The raw tensor output path is intended for flexibility, experimentation, and integration - not for maximum speed. If you want maximum performance in production applications, consider building a standard decoder as done for models in the Axelera model zoo, which allows the pipeline to use fully optimized postprocessing.

### Example: Extracting the tensor in Python

When you run inference, you can extract the tensor from the frame result like this:

```python
for frame_result in stream:
    tensor_wrapper = frame_result.meta['detections']
    tensor = tensor_wrapper.tensors[0]  # This is a numpy array
    # Now you can implement your own postprocessing, e.g.:
    # detections = postprocess_yolov8(tensor, tensor.shape, ...)
```

> **Note:**
> - The output is a tensor (e.g., shape `[1, 84, 8400]` for YOLOv8), not a list of detection objects.
> - You are responsible for all postprocessing, including decoding, NMS, and mapping to image coordinates.
> - **For InferenceStream, we highly recommend using the standard pipeline path for production and real applications.** The raw tensor output path is intended primarily for educational purposes, experimentation, or quick prototyping, and it significantly reduces performance compared to the standard pipeline.
> - You can use the raw tensor output path for quick prototyping or exploration. After that, you can either:
>   - Move to a production solution with AxInferenceNet ([AxInferenceNet C++ Integration Tutorial](/docs/tutorials/axinferencenet.md)), or
>   - Finish your implementation and use InferenceStream properly with a standard pipeline ([Tutorials](/ax_models/tutorials/general/tutorials.md)).


See the [axinferencenet_tensor.cpp](/examples/axinferencenet/axinferencenet_tensor.cpp) and [application_tensor.py](/examples/application_tensor.py) examples for more details on working with raw tensor output.


## List of example applications

| Example | Description |
| :------ | :---------- |
| [`/examples/application.py`](/examples/application.py) | Simple integration of vehicle tracker into an application including visualisation and basic analytics |
| [`/examples/application_extended.py`](/examples/application_extended.py) | Adds advanced customization and monitoring to the simple vehicle tracker example |
| [`/examples/demos/fruit_demo.py`](/examples/demos/fruit_demo.py) | Renders segmentation of fruits held by people in colour against a grayscale background |
| [`/examples/application_tensor.py`](/examples/application_tensor.py) | Demonstrates how to extract and postprocess the raw tensor output from a YOLOv8 model |
