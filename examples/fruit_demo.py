#!/usr/bin/env python
# Copyright Axelera AI, 2025
#
# This example shows how to run a complex pipeline involving 3 different models
#
# │ └─master_detections   - yolov8lpose-coco-onnx
# │ └─    │ segmentations - yolov8sseg-coco-onnx
#   └─object_detections   - yolov8s-fruit (this is yolov8s-coco-onnx with a filter for fruit)
#
# The purpose of the example is to show how using the ROI of the master_detections model makes
# detection of sub object (fruit in this case) more effective.
#
# The example also shows some other functionality, for example:
#   * how to configure rendering options tuned for this application
#   * how to use the inference.py argparser with a default network
#
from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.stream import create_inference_stream

NETWORK = 'fruit-demo'


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


if __name__ == '__main__':
    # Reuse the inference.py command line parameters
    parser = config.create_inference_argparser(
        default_network=NETWORK, description='Perform inference on an Axelera platform'
    )
    args = parser.parse_args()
    tracers = inf_tracers.create_tracers('core_temp', 'end_to_end_fps')
    stream = create_inference_stream(
        network=NETWORK,
        sources=args.sources,
        pipe_type='gst',
        log_level=logging_utils.get_config_from_args(args).console_level,
        hardware_caps=config.HardwareCaps.from_parsed_args(args),
        tracers=tracers,
        rtsp_latency=args.rtsp_latency,
        allow_hardware_codec=args.enable_hardware_codec,
        specified_frame_rate=args.frame_rate,
    )

    with display.App(
        visible=args.display,
        opengl=stream.hardware_caps.opengl,
        buffering=not stream.is_single_image(),
    ) as app:
        wnd = app.create_window("CES 2025 Fruit Demo", args.window_size)
        app.start_thread(main, (wnd, stream), name='InferenceThread')
        app.run()
    stream.stop()
