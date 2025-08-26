#!/usr/bin/env python
import os
import time

from axelera.types import ColorFormat, Image
import cv2

from axelera.app import config, display
from axelera.app.stream import create_inference_stream

NETWORK = "yolov8s-coco"
SOURCE = config.env.framework / "media/traffic1_1080p.mp4"

TITLE = "Axelera Python Data Reader Example"
TITLE_FONT = 39

LOGO0 = os.path.join(config.env.framework, "axelera/app/axelera-ai-logo-logo-only.png")
LOGO1 = os.path.join(config.env.framework, "axelera/app/voyager-sdk-logo-white.png")
LOGO2 = os.path.join(config.env.framework, "axelera/app/axelera-ai-logo-text-only.png")


def frame_reader(src):
    cap = cv2.VideoCapture(src)
    n = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if n % 2 == 0:
                yield frame
            else:
                # Every other frame is returned as an axelera.types.Image, which
                # supports PIL, numpy and gstreamer buffers
                # The types may be used together as shown here
                yield Image.fromarray(frame, ColorFormat.BGR)
            n += 1
    finally:
        cap.release()


stream = create_inference_stream(network=NETWORK, sources=[frame_reader(SOURCE)])


def main(window, stream):
    window.options(
        -1,
        title=TITLE,
        title_position="50%, 0%",
        title_size=TITLE_FONT,
        title_anchor_x="center",
    )

    logo0 = wnd.image('52%, 88%', LOGO0, anchor_x='right', anchor_y='center', scale=0.4)
    logo1 = wnd.image('52%, 88%', LOGO1, anchor_x='left', anchor_y='center', scale=0.45)
    logo2 = wnd.image(
        '52%, 88%', LOGO2, anchor_x='left', anchor_y='center', scale=0.4, fadeout_from=0.0
    )
    supported = logo0 and logo1 and logo2
    start = time.time()
    period = 10.0

    for frame_result in stream:
        now = time.time()
        if supported and ((now - start) > period):
            logo1.hide(now, 1.0)
            logo2.show(now, 1.0)
            logo1, logo2 = logo2, logo1
            start = now

        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
        count = sum(x.is_car for x in frame_result.detections)
        print(f"Detected {count} cars\r", end="")


with display.App(visible=True) as app:
    wnd = app.create_window(TITLE, (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run()
stream.stop()
