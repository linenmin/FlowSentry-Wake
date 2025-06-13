#!/usr/bin/env python
# Copyright Axelera AI, 2025

import os

from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.stream import create_inference_stream

NETWORK = 'yolov8l-coco-onnx'
TITLE = "Axelera 8K Demo YOLOV8L Tiling"
LOGO = os.path.join(config.env.framework, "axelera/app/voyager-sdk-logo-white.png")
TITLE_FONT = 78
UNTILED_GRAYSCALE = 0.7  # 0.0 = no grayscale, 1.0 = full grayscale


def main(window, stream, args):
    if args.tile_position != 'none':
        opposites = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
        grayscale_area = opposites.get(args.tile_position, 'all')
        grayscale = UNTILED_GRAYSCALE
        if args.tile_position in ('left', 'right'):
            anchors = {'anchor_x': 'center', 'anchor_y': 'bottom'}
            labels = [('25%', '10%'), ('75%', '10%')]
        elif args.tile_position in ('top', 'bottom'):
            anchors = {'anchor_x': 'right', 'anchor_y': 'center'}
            labels = [('98%', '25%'), ('98%', '75%')]
        if args.tile_position in ('right', 'bottom'):
            labels.reverse()
    else:
        grayscale, grayscale_area, labels = 0.0, 'all', []

    window.options(
        -1,
        title=TITLE,
        title_position=('50%', '0%'),
        title_size=TITLE_FONT,
        title_anchor_x='center',
    )

    window.options(
        0,
        grayscale=grayscale,
        grayscale_area=grayscale_area,
        bbox_label_format="{label} {scorep:.0f}%",
    )
    if labels:
        window.text(labels[0], "Tiling enabled", **anchors)
        window.text(labels[1], "No tiling", **anchors)
    window.image(('98%', '98%'), LOGO, anchor_x='right', anchor_y='bottom')
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)


if __name__ == '__main__':
    parser = config.create_inference_argparser(
        default_network=NETWORK, description='Perform inference on an Axelera platform'
    )
    args = parser.parse_args()
    tracers = inf_tracers.create_tracers('core_temp', 'end_to_end_fps')
    stream = create_inference_stream(
        network=NETWORK,
        sources=args.sources,
        pipe_type=args.pipe,
        log_level=logging_utils.get_config_from_args(args).console_level,
        hardware_caps=config.HardwareCaps.from_parsed_args(args),
        tracers=tracers,
        allow_hardware_codec=args.enable_hardware_codec,
        rtsp_latency=args.rtsp_latency,
        device_selector=args.devices,
        specified_frame_rate=args.frame_rate,
        tile=config.determine_tile_options(args),
    )

    with display.App(
        visible=args.display,
        opengl=stream.manager.hardware_caps.opengl,
        buffering=not stream.is_single_image(),
    ) as app:
        wnd = app.create_window(TITLE, args.window_size)
        app.start_thread(main, (wnd, stream, args), name='InferenceThread')
        app.run(interval=1 / 10)
    stream.stop()
