#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Extended app with additional config, showing advanced usage of metadata
import argparse

from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.stream import create_inference_stream

parser = argparse.ArgumentParser(description="Multiple pipelines demo")
config.HardwareCaps.add_to_argparser(parser)
config.add_display_arguments(parser)

args = parser.parse_args()
hwcaps = config.HardwareCaps.from_parsed_args(args)
framework = config.env.framework
tracers = inf_tracers.create_tracers('end_to_end_fps', 'latency')

stream = create_inference_stream(
    log_level=logging_utils.INFO,  # INFO, DEBUG, TRACE
    hardware_caps=hwcaps,
)

networks = ['yolov8n-coco', 'yolov8npose-coco', 'yolov8nseg-coco', 'yolov8s-coco']
sources = [f'rtsp://127.0.0.1:8554/{n}' for n in range(4)]

for n, s in zip(networks, sources):
    stream.add_pipeline(
        network=n,
        sources=[s],
        pipe_type='gst',
        allow_hardware_codec=False,
        tracers=tracers,
        rtsp_latency=50,
    )


def main(window, stream):
    for i, pipeline in enumerate(stream.pipelines):
        for src_id, src in pipeline.sources.items():
            title = f"Pipeline {i}: {pipeline.network.name} ({src.location})"
            window.options(src_id, title=title)
            print(src_id, title)
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)


with display.App(
    visible=args.display,
    opengl=stream.hardware_caps.opengl,
    buffering=not stream.is_single_image(),
) as app:
    wnd = app.create_window("Multiple pipeline demo", args.window_size)
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run()
stream.stop()
