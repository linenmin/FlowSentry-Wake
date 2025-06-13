# Copyright Axelera AI, 2025
from __future__ import annotations

import collections
import dataclasses
import functools
import os
import queue
import time
from typing import TYPE_CHECKING, Sequence, Tuple
import uuid

import PIL
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import numpy as np

from axelera import types

from . import display, logging_utils, meta

if TYPE_CHECKING:
    from . import inf_tracers

LOG = logging_utils.getLogger(__name__)

SegmentationMask = tuple[int, int, int, int, int, int, int, int, np.ndarray]


def _read_new_data(wnds, queues):
    frames = {}
    others = {}
    for wnd, q in zip(wnds, queues):
        try:
            while True:
                msg = q.get(block=False)
                if msg is display.SHUTDOWN or msg is display.THREAD_COMPLETED:
                    return msg
                wndname = (
                    f"{wnd} {msg.stream_id}"
                    if isinstance(msg, display._StreamMessage) and msg.stream_id > 0
                    else wnd
                )
                if isinstance(msg, display._Frame):
                    frames[wndname] = msg  # keep only the latest frame data
                else:
                    others.setdefault(wndname, []).append(msg)  # don't throw away other messages
        except queue.Empty:
            pass
    msgs = []
    for wndname, wndmsgs in others.items():
        msgs.extend([(wndname, m) for m in wndmsgs])
    msgs.extend(frames.items())
    return msgs


def _make_splash(window_size):
    for ico_sz, path in reversed(display.ICONS.items()):
        if ico_sz < min(window_size):
            ico = cv2.imread(path)
            break
    top = int((window_size[1] - ico.shape[0]) / 2)
    bottom = window_size[1] - top - ico.shape[0]
    left = int((window_size[0] - ico.shape[1]) / 2)
    right = window_size[0] - left - ico.shape[1]
    return cv2.copyMakeBorder(ico, top, bottom, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))


def rgb_to_grayscale_rgb(img: np.ndarray, grayness: float) -> np.ndarray:
    if grayness > 0.0:
        grey = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        orig, img = img, np.stack((grey, grey, grey), axis=-1)
        if grayness < 1.0:
            img *= grayness
            img += orig * (1.0 - grayness)
        img = img.astype('uint8')
    return img


@dataclasses.dataclass
class CVOptions(display.Options):
    pass  # No additional options for opencv


class CVApp(display.App):
    SupportedOptions = CVOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream_wnds = set()
        self._meta_cache = display.MetaCache()
        self._layers: dict[uuid.UUID, display._Layer] = collections.defaultdict()

    def _create_new_window(self, q, wndname, size):
        del q  # unused
        if size == display.FULL_SCREEN:
            cv2.namedWindow(wndname, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(wndname, cv2.WINDOW_NORMAL)
            cv2.imshow(wndname, _make_splash(size))
        cv2.setWindowProperty(wndname, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        self._stream_wnds.add(wndname)
        return wndname

    def _destroy_all_windows(self):
        cv2.destroyAllWindows()

    def _get_layers(self, stream_id):
        '''
        Delete any layers if necessary, and return the layers to be rendered by the
        requested window.

        Layers with a stream_id of -1 are global and will be rendered on all windows.
        Otherwise the layer will only be rendered on the window with the same stream_id.
        '''
        now = time.time()
        expired = [
            k
            for k, v in self._layers.items()
            if v.fadeout_start and now - v.fadeout_start >= v.fadeout_duration
        ]
        for k in expired:
            self._layers.pop(k)
        return [x for x in self._layers.values() if x.stream_id in [stream_id, -1]]

    def _run(self, interval=1 / 30):
        last_frame = time.time()
        options: dict[str, CVOptions] = collections.defaultdict(CVOptions)
        pending_titles: dict[str, CVOptions] = {}
        while 1:
            self._create_new_windows()
            new_msgs = _read_new_data(self._wnds, self._queues)
            if new_msgs is display.SHUTDOWN:
                return
            if new_msgs is display.THREAD_COMPLETED:
                continue  # ignore, just wait for user to close

            for wndname, msg in new_msgs:
                opts = options[wndname]
                if isinstance(msg, display._SetOptions):
                    oldtitle = opts.title
                    opts.update(**msg.options)
                    if oldtitle != opts.title:
                        pending_titles[wndname] = options[wndname].title
                    continue
                elif isinstance(msg, display._Delete):
                    if msg.id in self._layers and not self._layers[msg.id].fadeout_start:
                        self._layers[msg.id].fadeout_start = time.time()
                        self._layers[msg.id].fadeout_duration = msg.fadeout
                    else:
                        LOG.trace(f'layer {msg.id} already deleted or fading out')
                    continue
                elif isinstance(msg, display._Layer):
                    self._layers[msg.id] = msg
                    continue
                elif not isinstance(msg, display._Frame):
                    LOG.debug(f"Unknown message: {msg}")
                    continue

                if wndname not in self._wnds:
                    self._create_new_window(None, wndname, msg.image.size)
                layers = self._get_layers(msg.stream_id)
                draw = CVDraw(msg.image, layers, opts)

                _, meta_map = self._meta_cache.get(msg.stream_id, msg.meta)
                for m in meta_map.values():
                    m.visit(lambda m: m.draw(draw))
                draw.draw()
                if (
                    msg.image.color_format == types.ColorFormat.RGB
                    or msg.image.color_format == types.ColorFormat.RGBA
                ):
                    bgr = msg.image.asarray(types.ColorFormat.BGRA)
                else:
                    bgr = msg.image.asarray(types.ColorFormat.RGBA)
                if pending := pending_titles.pop(wndname, None):
                    cv2.setWindowTitle(wndname, pending)
                cv2.imshow(wndname, bgr)

            if any(cv2.getWindowProperty(t, cv2.WND_PROP_VISIBLE) <= 0.0 for t in self._wnds):
                return

            now = time.time()
            remaining = max(1, interval - (now - last_frame))
            last_frame = now
            if cv2.waitKey(remaining) in (ord("q"), ord("Q"), 27, 32):
                return
            if self.has_thread_completed:
                return


_FONT_FAMILIES = {
    display.FontFamily.sans_serif: "sans-serif",
    display.FontFamily.serif: "serif",
    # 'cursive', 'fantasy', or 'monospace'
}


def _coords(centre, length):
    return centre[0] - length, centre[1] - length, centre[0] + length, centre[1] + length


@functools.lru_cache
def _get_speedometer(diameter):
    here = os.path.dirname(__file__)
    x = PIL.Image.open(f'{here}/speedo-alpha-transparent.png')
    return x.resize((diameter, diameter))


@functools.lru_cache(maxsize=128)
def _load_pil_image_from_file(filename):
    return PIL.Image.open(filename)


def _normalize_anchor_point(
    x: int,
    y: int,
    width: int,
    height: int,
    anchor_x: str,
    anchor_y: str,
) -> Tuple[int, int]:
    '''
    Default pillow anchor is the top left corner. If we are using a different anchor point,
    calculate the x and y co-ordinate of the top-left corner from the given point, anchor and
    renderable size, and use that.
    '''
    if anchor_x == 'center':
        x -= width / 2
    elif anchor_x == 'right':
        x -= width
    if anchor_y == 'center':
        y -= height / 2
    elif anchor_y == 'bottom':
        y -= height
    return x, y


class _DrawList(list):
    def __getattr__(self, name):
        def _draw(*args):
            self.append((name,) + args)

        return _draw


class CVDraw(display.Draw):
    def __init__(
        self, image: types.Image, layers: list[display._Layer], options: CVOptions = CVOptions()
    ):
        self._canvas_size = (image.width, image.height)
        self._img = image
        rgb = image.asarray('RGB')
        self._pil = PIL.Image.fromarray(rgb_to_grayscale_rgb(rgb, options.grayscale))
        self._draw = PIL.ImageDraw.Draw(self._pil, "RGBA")
        self._dlist = _DrawList()
        self._font_cache = {}
        self._speedometer_index = 0
        self._options = options

        now = time.time()
        for l in layers:
            if l.fadeout_start is None or l.fadeout_start > now:
                fadeout_percent = 1.0
            else:
                fadeout_percent = max(
                    0.0, 1.0 - ((now - l.fadeout_start) / float(l.fadeout_duration))
                )

            pt = l.position.as_px(self._canvas_size)
            if isinstance(l, display._Text):
                font = display.Font(size=l.font_size)
                text_size = self.textsize(l.text, font)
                pt = _normalize_anchor_point(
                    *pt,
                    text_size[0],
                    text_size[1],
                    l.anchor_x,
                    l.anchor_y,
                )
                color = l.color[:3] + (int(l.color[3] * fadeout_percent),)
                bgcolor = l.bgcolor[:3] + (int(l.bgcolor[3] * fadeout_percent),)
                txt = PIL.Image.new('RGBA', text_size, bgcolor)
                txt_draw = PIL.ImageDraw.Draw(txt, "RGBA")
                txt_draw.text((0, 0), l.text, color, self._load_font(font), "lt")
                self._dlist.paste(txt, pt, txt)
            elif isinstance(l, display._Image):
                img = _load_pil_image_from_file(l.path)
                if l.scale is None:
                    scale = 1.0, 1.0
                elif isinstance(l.scale, float):
                    scale = l.scale, l.scale
                img = img.resize((int(scale[0] * img.width), int(scale[1] * img.height)))
                if fadeout_percent < 1.0:
                    img = img.convert("RGBA")
                    alpha = np.array(img.split()[-1])
                    alpha = (alpha * fadeout_percent).astype(np.uint8)
                    img.putalpha(PIL.Image.fromarray(alpha))
                pt = _normalize_anchor_point(
                    *pt,
                    img.width,
                    img.height,
                    l.anchor_x,
                    l.anchor_y,
                )

                self._dlist.paste(img, pt, img)
            else:
                LOG.debug(f"Unknown layer type {l.__class__.__name__} ignoring...")

    def _pt_transform(self, pt: tuple[int | float, int | float]) -> tuple[int, int]:
        if type(pt[0]) is float:
            return int(pt[0] * self._canvas_size[0]), int(pt[1] * self._canvas_size[1])
        return pt

    @property
    def options(self) -> CVOptions:
        return self._options

    @property
    def canvas_size(self) -> display.Point:
        return self._canvas_size

    @property
    def _speedometer_dlist_calls(self):
        return 4 * self._speedometer_index

    def draw_speedometer(self, metric: inf_tracers.TraceMetric):
        text = display.calculate_speedometer_text(metric)
        needle_pos = display.calculate_speedometer_needle_pos(metric)
        m = display.SpeedometerMetrics(self._canvas_size, self._speedometer_index)
        font = display.Font(size=m.text_size)

        speedometer = _get_speedometer(m.diameter)
        C = m.center
        self._dlist.paste(speedometer, m.top_left, speedometer)
        pos = (C[0], C[1] + m.text_offset)
        self._dlist.text(pos, text, m.text_color, self._load_font(font), "mb")
        font = dataclasses.replace(font, size=round(0.8 * font.size))
        pos = (C[0], C[1] + m.radius * 75 // 100)
        self._dlist.text(pos, metric.title, m.text_color, self._load_font(font), "mb")
        needle_coords = _coords(C, m.needle_radius)
        self._dlist.pieslice(needle_coords, needle_pos - 2, needle_pos + 2, m.needle_color)

        self._speedometer_index += 1

    def draw_statistics(self, stats):
        pass

    def polylines(
        self,
        lines: Sequence[Sequence[display.Point]],
        closed: bool = False,
        color: display.Color = (255, 255, 255, 255),
        width: int = 1,
    ) -> None:
        import itertools

        # flatten the points into `[[x1, y1, x2, y2, ...], ...]` because PIL
        # insists on tuple for the points if given.
        lines = [list(itertools.chain.from_iterable(line)) for line in lines]
        for line in lines:
            if closed:
                line = line + line[:2]  # make a copy with the first point at the end
            self._dlist.line(line, color, width)

    def _load_font(self, font: display.Font):
        args = dataclasses.astuple(font)
        try:
            return self._font_cache[args]
        except KeyError:
            path = os.path.join(os.path.dirname(__file__), "axelera-sans.ttf")
            f = self._font_cache[args] = PIL.ImageFont.truetype(path, size=font.size)
            return f

    def rectangle(self, p1, p2, fill=None, outline=None, width=1):
        self._dlist.rectangle((p1, p2), fill, outline, int(width))

    def textsize(self, text, font: display.Font = display.Font()):
        font = self._load_font(font)
        x1, y1, x2, y2 = font.getbbox(text)
        return (x2 - x1, y2 - y1)

    def text(
        self,
        p,
        text: str,
        txt_color: display.Color,
        back_color: display.OptionalColor = None,
        font: display.Font = display.Font(),
    ):
        if back_color is not None:
            w, h = self.textsize(text, font)
            self.rectangle(p, (p[0] + w, p[1] + h), back_color)
        self._dlist.text(p, text, txt_color, self._load_font(font), "lt")

    def keypoint(
        self, p: display.Point, color: display.Color = (255, 255, 255, 255), size=2
    ) -> None:
        r = size / 2
        p1, p2 = (p[0] - r, p[1] - r), (p[0] + r, p[1] + r)
        self._dlist.ellipse((p1, p2), color)

    def draw(self):
        def call_draw(d):
            if d[0] == 'paste':
                self._pil.paste(*d[1:])
            else:
                getattr(self._draw, d[0])(*d[1:])

        speedometers = self._dlist[: self._speedometer_dlist_calls]
        for d in self._dlist[self._speedometer_dlist_calls :]:
            call_draw(d)
        for d in speedometers:
            call_draw(d)
        self._img.update(pil=self._pil, color_format=self._img.color_format)

    def heatmap(self, data: np.ndarray, color_map: np.ndarray):
        indices = np.clip((data * len(color_map) - 1).astype(int), 0, len(color_map) - 1)
        rgba_mask = color_map[indices]
        mask_pil = PIL.Image.fromarray(rgba_mask)
        self._dlist.paste(mask_pil, (0, 0), mask_pil)

    def segmentation_mask(self, mask_data: SegmentationMask, color: Tuple[int]) -> None:
        x_adjust = y_adjust = 0
        mask, mbox = mask_data[-1], mask_data[4:8]
        img_size = (mbox[2] - mbox[0], mbox[3] - mbox[1])
        if 0 in img_size or 0 in mask.shape:
            return
        resized_image = cv2.resize(mask, img_size, interpolation=cv2.INTER_LINEAR)

        mid_point = np.iinfo(np.uint8).max // 2
        bool_array = resized_image > mid_point
        colored_mask = np.zeros((*bool_array.shape, 4), dtype=np.uint8)
        colored_mask[bool_array] = color

        mask_pil = PIL.Image.fromarray(colored_mask)
        offset = (mbox[0], mbox[1])
        self._dlist.paste(mask_pil, offset, mask_pil)

    def class_map_mask(self, class_map: np.ndarray, color_map: np.ndarray) -> None:
        colored_mask = color_map[class_map]
        colored_mask = cv2.resize(colored_mask, self._canvas_size)
        mask_pil = PIL.Image.fromarray(colored_mask)
        self._dlist.paste(mask_pil, (0, 0), mask_pil)

    def draw_image(self, image: np.ndarray):
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise ValueError("draw_image: image dtype must be np.uint8, np.float32, or np.float64")

        def float_to_uint8_image(float_img):
            d_min = np.min(float_img)
            d_max = np.max(float_img)

            if np.isclose(d_min, d_max):
                uint8_img = np.zeros_like(image, dtype=np.uint8)
            else:
                uint8_img = np.clip((image - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(
                    np.uint8
                )

            return uint8_img

        if image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(axis=0)  # Remove batch dimension
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(axis=0)  # Remove channel dimension if single channel

        if image.dtype == np.float32 or image.dtype == np.float64:
            image = float_to_uint8_image(image)

        # Convert CHW to HWC format
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        image = cv2.resize(image, self._canvas_size)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        pil_image = PIL.Image.fromarray(image)
        self._dlist.paste(pil_image, (0, 0), pil_image)
