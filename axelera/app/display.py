# Copyright Axelera AI, 2025
from __future__ import annotations

import abc
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import enum
import logging
import os
import queue
import threading
import time
import traceback
import typing
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
import uuid

import numpy as np
import psutil

from axelera import types

from . import config, logging_utils, utils

if TYPE_CHECKING:
    from . import inf_tracers
    from .meta import AxMeta, AxTaskMeta


LOG = logging_utils.getLogger(__name__)

FULL_SCREEN = (-1, -1)
ICONS = {sz: f'{os.path.dirname(__file__)}/axelera-{sz}x{sz}.png' for sz in [32, 128, 192]}


class Coords:
    @classmethod
    def rel(cls, x: float, y: float) -> Coords:
        '''Create a Coords object from relative coordinates.'''
        if type(x) is not float or type(y) is not float:
            raise TypeError("x and y must be floats")
        return cls(f'{x * 100}%', f'{y * 100}%')

    @classmethod
    def px(cls, x: int, y: int) -> Coords:
        '''Create a Coords object from image pixel coordinates.'''
        if type(x) is not int or type(y) is not int:
            raise TypeError("x and y must be ints")
        return cls(f'{x}px', f'{y}px')

    def _parse_coord(self, coord: str) -> float | int:
        try:
            if coord.endswith('%'):
                return float(coord[:-1]) / 100
            if coord.endswith('px'):
                return int(float(coord[:-2]))
        except ValueError:
            pass
        raise ValueError(f"Invalid coordinate: {coord}")

    def __init__(self, x: str, y: str):
        if not isinstance(x, str) or not isinstance(y, str):
            raise TypeError("x and y must be strings")
        self.x = self._parse_coord(x)
        self.y = self._parse_coord(y)

    def __eq__(self, value):
        if isinstance(value, Coords):
            return (
                self.x == value.x
                and self.y == value.y
                and type(self.x) == type(value.x)
                and type(self.y) == type(value.y)
            )
        if isinstance(value, (tuple, list)) and len(value) == 2:
            x, y = value
            if isinstance(x, str):
                try:
                    x = self._parse_coord(x)
                except ValueError:
                    return False
            if isinstance(y, str):
                try:
                    y = self._parse_coord(y)
                except ValueError:
                    return False
            return (
                self.x == x and self.y == y and type(self.x) == type(x) and type(self.y) == type(y)
            )
        return False

    @property
    def format(self) -> str:
        '''Return the format of the coordinates.'''
        if isinstance(self.x, float) and isinstance(self.y, float):
            return 'relative'
        if isinstance(self.x, int) and isinstance(self.y, int):
            return 'image_pixels'
        return 'mixed'

    def as_px(self, image_size: tuple[int, int] = None) -> tuple[int, int]:
        '''Convert the coordinates to image pixel coordinates.'''
        if image_size is None and self.format != 'image_pixels':
            raise ValueError(
                "image_size must be provided as Coords are not in image_pixels format"
            )
        x = self.x
        y = self.y
        if isinstance(x, float):
            x = int(x * image_size[0])
        if isinstance(y, float):
            y = int(y * image_size[1])
        return x, y

    def as_rel(self, image_size: tuple[int, int] = None) -> tuple[float, float]:
        '''Convert the coordinates to relative coordinates.'''
        if image_size is None and self.format != 'relative':
            raise ValueError("image_size must be provided as Coords are not in relative format")
        x = self.x
        y = self.y
        if isinstance(x, int):
            x = float(x) / image_size[0]
        if isinstance(y, int):
            y = float(y) / image_size[1]
        return x, y


class _Message:
    pass


@dataclass
class _StreamMessage(_Message):
    stream_id: int


@dataclass
class _SetOptions(_StreamMessage):
    options: dict[str, Any]


@dataclass
class _Frame(_StreamMessage):
    image: types.Image
    meta: AxMeta | None


@dataclass
class _Shutdown(_Message):
    pass


@dataclass
class _ThreadCompleted(_Message):
    pass


@dataclass
class _Delete(_Message):
    id: uuid.UUID
    fadeout: int


@dataclass
class _Layer(_StreamMessage):
    id: uuid.UUID
    position: Coords
    anchor_x: str
    anchor_y: str
    fadeout_duration: int
    fadeout_start: int | None


@dataclass
class _Text(_Layer):
    text: str
    color: Color
    bgcolor: Color
    font_size: int


@dataclass
class _Image(_Layer):
    path: str
    scale: float | None


class LayerHandle:
    def __init__(self, message: _Layer, window: Window):
        self._fields = asdict(message, dict_factory=OrderedDict)
        self._id = self._fields.pop('id')
        self._stream_id = self._fields.pop('stream_id')
        self._Message = type(message)
        self._window = window
        self._visible = True

    @property
    def id(self) -> uuid.UUID:
        '''The id of the layer.'''
        return self._id

    @property
    def stream_id(self) -> int:
        '''
        The id of the stream the layer belongs to.

        stream_id `-1` means the layer is not associated with a stream, and belongs
        to the window(s).
        '''
        return self._stream_id

    @property
    def visible(self) -> bool:
        '''True if the layer is visible.'''
        return self._visible

    def _send_message(self):
        self._window.layer(
            self._Message,
            *self._fields.values(),
            stream_id=self._stream_id,
            existing=self,
        )

    def set(self, **kwargs):
        changed = False
        for name, value in kwargs.items():
            if name in self._fields:
                self._fields[name] = value
                changed = True
            else:
                LOG.warning(
                    f"Cannot set attribute '{name}' for LayerHandle."
                    f" id={self.id}, message_type={self._Message.__name__}"
                )
        if self.visible and changed:
            self._send_message()

    def __setitem__(self, key, value):
        self.set(**{key: value})

    def __getitem__(self, name):
        if name in self._fields:
            return self._fields[name]
        raise KeyError(
            f"Cannot get attribute '{name}' from LayerHandle."
            f" id={self.id}, message_type={self._Message.__name__}"
        )

    def hide(self, fadeout: Optional[int] = 0):
        '''Hide the layer from the window.'''
        self._window.delete(self._id, fadeout)
        self._visible = False

    def show(self):  # TODO: add fadein
        '''Show the layer in the window.'''
        self._send_message()
        self._visible = True


SHUTDOWN = _Shutdown()
THREAD_COMPLETED = _ThreadCompleted()


def _typecheck(v, expected_type):
    for t in [int, float, str]:
        if expected_type == tuple[t]:
            return isinstance(v, tuple) and all(isinstance(x, t) for x in v)
    return isinstance(v, expected_type)


@dataclass
class Options:
    title: str = ''
    '''The title of the stream window, set to '' to hide the title bar.'''

    grayscale: float | int | bool = 0.0
    '''Render the inferenced image in grayscale. This can be effective when viewing segmentation.
    The value is a float between 0.0 and 1.0 where 0.0 is the original image and 1.0 is completely
    grayscale.
    '''

    bbox_label_format: str = "{label} {scorep:.0f}%"
    '''Control how labels on bounding boxes should be shown. Available keywords are:

    label: str Label if known, or "cls:%d" if no label for that class id
    score: float The score of the detection as a float 0-1
    scorep: float The score of the detection as a percentage 0-100

    An empty string will result in no label.

    If there are formatting errors in the format string, an error is shown and the default is used
    instead.

    For example, assuming label is "apple" and score is 0.75:

        "{label} {score:.2f}" -> "apple 0.75"
        "{label} {scorep:.0f}%" -> "apple 75%"
        "{scorep}" -> "75"
        "{label}" -> "apple"
    '''

    bbox_class_colors: dict = field(default_factory=dict)
    '''A dictionary of class id to color.

    The key should be a class id (int) or a label (str) and the value should be a color tuple
    (r, g, b, a) where each value is an integer between 0 and 255.

    class ids and labels can be mixed. A matching label is given priority over a matching class id.
    '''

    def update(self, **options: dict[str, Any]) -> None:
        '''Update the options with the given dictionary.

        Warns for unsupported options, and logs if the value is not the expected type, in both cases

        '''
        unsupported = []
        types = typing.get_type_hints(type(self))
        for k, v in options.items():
            try:
                expected_type = types[k]
            except KeyError:
                unsupported.append(k)
            else:
                if _typecheck(v, expected_type):
                    setattr(self, k, v)
                else:
                    exp = getattr(expected_type, '__name__', str(expected_type))
                    got = type(v).__name__
                    _for = f"{type(self).__name__}.{k}"
                    LOG.error(f"Expected {exp} for {_for}, but got {got}")

        if unsupported:
            s = 's' if len(unsupported) > 1 else ''
            LOG.info(f"Unsupported option{s} : {type(self).__name__}.{', '.join(unsupported)}")


default_anchor_x = 'left'
default_anchor_y = 'top'
default_fadeout_duration = -1
default_fadeout_start = None


class Window:
    '''Created by `App.create_window` to display inference results.'''

    def __init__(self, q: Optional[queue.Queue], is_closed: threading.Event):
        self._queue = q
        self._is_closed = is_closed
        self._warned_full = False
        self._self_proc = psutil.Process(os.getpid())
        self._last_print = time.time() - 2

    def options(self, stream_id: int, **options: dict[str, Any]) -> None:
        '''Set options for the given stream.

        Valid options depends on the renderer being used. All renderers support title:str.
        '''
        if self._queue is not None:
            self._queue.put(_SetOptions(stream_id, options))

    def delete(self, id: uuid.UUID, fadeout: int = 0) -> None:
        '''Delete the given layer from the stream.'''
        if self._queue is not None:
            self._queue.put(_Delete(id, fadeout))

    def layer(
        self,
        Layer: type[_Layer],
        position: tuple[str, str] | Coords,
        anchor_x: str,
        anchor_y: str,
        fadeout_duration: int,
        fadeout_start: int,
        *args,
        existing: Optional[LayerHandle] = None,
        stream_id: int = -1,
    ) -> LayerHandle | None:
        if self._queue is not None:
            _id = existing.id if existing else uuid.uuid4()
            position = position if isinstance(position, Coords) else Coords(*position)
            layer = Layer(
                stream_id,
                _id,
                position,
                anchor_x,
                anchor_y,
                fadeout_duration,
                fadeout_start,
                *args,
            )
            self._queue.put(layer)
            return existing or LayerHandle(layer, self)
        return None

    def text(
        self,
        position: tuple[str, str] | Coords,
        text: str,
        anchor_x: float = default_anchor_x,
        anchor_y: float = default_anchor_y,
        fadeout_duration: int = default_fadeout_duration,
        fadeout_start: Optional[int] = default_fadeout_start,
        color: Color = (244, 190, 24, 255),  # Orange
        bgcolor: Color = (0, 0, 0, 192),  # Smoked glass
        font_size: int = 32,
        existing: Optional[LayerHandle] = None,
        stream_id: int = -1,
    ) -> LayerHandle | None:
        return self.layer(
            _Text,
            position,
            anchor_x,
            anchor_y,
            fadeout_duration,
            fadeout_start,
            text,
            color,
            bgcolor,
            font_size,
            existing=existing,
            stream_id=stream_id,
        )

    def image(
        self,
        position: tuple[str, str] | Coords,
        path: str,
        anchor_x: float = default_anchor_x,
        anchor_y: float = default_anchor_y,
        fadeout_duration: int = default_fadeout_duration,
        fadeout_start: Optional[int] = default_fadeout_start,
        scale: Optional[float] = None,
        existing: Optional[LayerHandle] = None,
        stream_id: int = -1,
    ) -> LayerHandle | None:
        return self.layer(
            _Image,
            position,
            anchor_x,
            anchor_y,
            fadeout_duration,
            fadeout_start,
            path,
            scale,
            existing=existing,
            stream_id=stream_id,
        )

    def show(self, image: types.Image, meta: AxMeta | None = None, stream_id: int = 0) -> None:
        '''Display an image in the window.'''
        if not isinstance(image, types.Image):
            raise TypeError(f"Expected axelera.types.Image, got {image}")
        if self._queue is not None:
            try:
                self._queue.put_nowait(_Frame(stream_id, image, meta))
            except queue.Full:
                level = LOG.warning if not self._warned_full else LOG.debug
                level("Display queue is full, dropping frame")
                self._warned_full = True
        if LOG.isEnabledFor(logging.DEBUG) and time.time() - self._last_print > 2:
            minfo = self._self_proc.memory_info()
            system = psutil.virtual_memory().used
            qsize = self._queue.qsize() if self._queue is not None else 0
            LOG.debug(
                f"System memory: {system / 1024 ** 2:.2f} MB\t"
                f"axelera: {minfo.rss / 1024 ** 2:.2f} MB, vms = {minfo.vms / 1024 ** 2:.2f} MB\t"
                f"display queue size: {qsize}"
            )
            self._last_print = time.time()

    @property
    def is_closed(self) -> bool:
        '''True if the window has been closed.'''
        return self._is_closed.is_set()

    def close(self):
        '''Close the window.'''
        if self._queue is not None:
            self._queue.put(SHUTDOWN)
        else:
            self._is_closed.set()

    def wait_for_close(self):
        LOG.info("stream has a single frame, close the window or press q to exit")
        if self._queue is not None and self._queue.empty():
            self._queue.put(THREAD_COMPLETED)
        while not self.is_closed:
            time.sleep(0.1)


def _find_display_class(display: str | bool, opengl: config.HardwareEnable):
    from . import display_console, display_cv

    display_env = os.environ.get('DISPLAY')
    # take care not to import display_gl before checking for the backend availablity
    display = 'auto' if display is True else display
    display = 'none' if display is False else display

    if display in ('auto', 'opengl'):
        if display == 'auto' and opengl == config.HardwareEnable.detect:
            opengl = (
                config.HardwareEnable.enable
                if utils.is_opengl_available(config.env.opengl_backend)
                else config.HardwareEnable.disable
            )
        if display != 'auto' or opengl == config.HardwareEnable.enable:
            try:
                from . import display_gl

                return display_gl.GLApp
            except Exception as e:
                if display_env:
                    msg = f"DISPLAY environment variable={display_env}"
                else:
                    msg = "Please try exporting the environment variable DISPLAY=:0.0"
                msg = f"Failed to initialize OpenGL: {e!r}\n{msg}"
                if display == 'opengl':
                    # if user explicilty requested opengl, we should not fallback to anything
                    raise RuntimeError(msg)
                LOG.warning(msg)

        return display_cv.CVApp if display_env else display_console.ConsoleApp
    elif display == 'opencv':
        return display_cv.CVApp
    elif display == 'console':
        return display_console.ConsoleApp
    elif display != 'none':
        expect = "'auto', 'opengl', 'opencv', 'console', 'none' or False"
        raise ValueError(f"Invalid display option: {display}, expect one of {expect}")
    return NullApp


class App:
    '''The App instance manages the windows and event queues.

    Because most UI frameworks require that windows be created in the main
    thread, any workload must be created in a sub thread using `start_thread`
    and then the event handling must be processed using `run`.

    For example:

        with display.App(visible=args.display) as app:
            app.start_thread(main, (args, stream, app), name='InferenceThread')
            app.run(interval=1 / 10)
    '''

    Window = Window
    SupportedOptions = Options

    def __init__(self, *args, **kwargs):
        self._wnds = []
        self._queues = []
        self._create_queue = queue.Queue()
        self._is_closed = threading.Event()
        self._thr = None

    def __new__(
        cls,
        visible: str | bool = False,
        opengl: config.HardwareEnable = config.HardwareEnable.detect,
        buffering=True,
    ):
        if cls is App:
            cls = _find_display_class(visible, opengl)

        x = object.__new__(cls)
        x.__init__(buffering=buffering)
        return x

    def create_window(self, title: str, size: tuple[int, int]) -> Window:
        '''Create a new Window, with given title and size.

        This method can be called from any thread.  The returned window may not
        be visible immediately.

        size is a tuple of (width, height) in pixels.  Use FULL_SCREEN for full
        screen.

        Note the title given is the title of the window, not the title of the stream(s).
        '''
        # note that windows must be created in UI thread, so push to create Q
        self._queues.append(q := queue.Queue(maxsize=100))
        self._create_queue.put_nowait((q, title, size))
        cls = type(self)
        return cls.Window(q, self._is_closed)

    def start_thread(self, target, args=(), kwargs={}, name=None):
        '''Start a worker thread.

        The thread starts immediately and is joined at the end of the `with`
        block.  Arguments are similar to the standard `thread.Thread`
        constructor.
        '''

        def _target():
            try:
                target(*args, **kwargs)
            except Exception as e:
                LOG.error('Exception in inference thread: %s', name, exc_info=True)
                if LOG.isEnabledFor(logging.DEBUG):
                    LOG.error(traceback.format_exc())
                else:
                    LOG.error(traceback.format_exception_only(type(e), e))
            finally:
                self._destroy_all_windows()

        thr = threading.Thread(target=_target, name=name)
        thr.start()
        self._thr = thr

    @property
    def has_thread_completed(self):
        '''True if the thread has completed.

        Specifically, this returns True if start_thread has been called and the
        worker thread has completed.

        Typically this is used from backends in the event handling code to
        shut down the app.
        '''
        return self._thr and not self._thr.is_alive()

    def run(self, interval=1 / 30):
        '''Start handling UI events.

        This function will not return until the thread has completed.
        '''
        try:
            self._run(interval)
        finally:
            self._is_closed.set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def _create_new_windows(self):
        while True:
            try:
                args = self._create_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self._wnds.append(self._create_new_window(*args))


class FontFamily(enum.Enum):
    sans_serif = enum.auto()
    serif = enum.auto()


@dataclass
class Font:
    family: FontFamily = FontFamily.sans_serif
    size: int = 12
    bold: bool = False
    italic: bool = False

    @property
    def weight(self):
        return 'bold' if self.bold else 'normal'


Point = tuple[int, int]
Color = tuple[int, int, int, int]
OptionalColor = Color | None


def midpoint(a: Point, b: Point) -> Point:
    '''Return the midpoint between two points.'''
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


class Draw(abc.ABC):
    @property
    @abc.abstractmethod
    def options(self) -> Options:
        '''Return the options for the renderer.'''

    @property
    @abc.abstractmethod
    def canvas_size(self) -> Point:
        '''Return the (width, height) of the canvas in pixels.'''

    @abc.abstractmethod
    def polylines(
        self,
        lines: Sequence[Sequence[Point]],
        closed: bool = False,
        color: Color = (255, 255, 255, 255),
        width: int = 1,
    ) -> None:
        '''Draw a series of lines, with the given color.

        Args:
            lines: a sequence of polylines to draw. Each polyline is a
              sequence of Points.
            closed: If True, then the first and last points of each polyline
              are connected.
            color: The color to use for the rendering of the lines.
            width: Width of the lines in pixels.
        '''

    @abc.abstractmethod
    def rectangle(
        self,
        p1: Point,
        p2: Point,
        fill: OptionalColor = None,
        outline: OptionalColor = None,
        width: int = 1,
    ) -> None:
        '''Draw a rectangle from p1 to p2, inclusive, with the given fill and outline.

        Args:
            p1: First point of the rectangle (typically top left)
            p2: Opposite point of the rectangle (typically bottom right)
            fill:  The color to use for the fill of the rectangle, or None for unfilled.
            outline:  The color to use for the border of the rectangle, or None for no border.
            width: The width of the border of rectangle (if outline is not None)
        '''

    @abc.abstractmethod
    def textsize(self, text: str, font: Font = Font()) -> tuple[int, int]:
        '''Return the size of the text in pixels, given the font.'''

    @abc.abstractmethod
    def text(
        self,
        p: Point,
        text: str,
        txt_color: Color,
        back_color: OptionalColor = None,
        font: Font = Font(),
    ) -> None:
        '''Draw the text at the given point, with the given colors and font.'''

    @abc.abstractmethod
    def keypoint(self, p: Point, color: Color = (255, 255, 255, 255), size=1) -> None:
        """Draw a point/dot/marker at given position of the given color.

        Args:
            p: Center coordinate of the point.
            color: The color to use.
            size: Size of point to display in pixels.
        """

    @abc.abstractmethod
    def draw(self) -> None:
        '''Render all of the items to the output, called by the UI system.

        This may be called multiple times by the display window, for example
        when an invalidation occurs or a resize event occurs.
        '''

    @abc.abstractmethod
    def heatmap(self, data: np.ndarray, color_map: np.ndarray) -> None:
        '''Overlay a heatmap mask on the image. `data` is expected to be
        an float32 np.ndarray of the same size as the image. With values for
        each pixel between 0-1. Float values will be mapped to the nearest color
        in the color map, with lower values choosing from the start, and higher
        values choosing from the end.
        '''

    @abc.abstractmethod
    def segmentation_mask(
        self, mask: np.ndarray, master_box: np.array, color: tuple[int, ...]
    ) -> None:
        '''Overlay a mask on the image.
        If resize_to_input is True, the mask is expected to be resized to the input size.
        '''

    @abc.abstractmethod
    def class_map_mask(self, class_map: np.ndarray, color_map: np.ndarray) -> None:
        '''Overlay a class map mask on the image.
        `class_map` is expected to be a numpy array where each pixel value is the class ID
        of what is detected in that pixel. The mask will be resized to the input size.
        '''

    @abc.abstractmethod
    def draw_image(self, image: np.ndarray) -> None:
        '''Draw an image on the canvas.'''

    def labelled_box(self, p1: Point, p2: Point, label: str, color: OptionalColor = None) -> None:
        """Draw a labelled bounding box in the best way for the renderer.

        The default implementation is to show an unfilled box with the label on
        top of the box if it fits, or else inside the box.  But the
        implementation may choose to render it completely differently or even
        make it an interactive UI element.
        """
        txt_color = 244, 190, 24, 255  # axelera orange
        label_back_color = 0, 0, 0, 255
        line_width = max(1, self.canvas_size[1] // 640)
        font = Font(size=max(self.canvas_size[1] // 50, 12))
        if color is not None:
            self.rectangle(p1, p2, outline=color, width=line_width)
        if label:
            _, text_height = self.textsize('Iy', font)
            text_box_height = text_height + line_width + 1
            label_outside = p1[1] - text_box_height >= 0
            top = p1[1] - text_box_height if label_outside else p1[1]
            self.text((p1[0], top + 1), label, txt_color, label_back_color, font)

    def trajectory(self, bboxes, color):
        """Draw the trajectory of an object based on its bounding boxes.

        Args:
            bboxes (2D np.array or list of lists): List of bounding boxes (x1, y1, x2, y2).
            color (tuple): Color of the trajectory line.
        """
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)

        if bboxes.shape[0] < 2:
            return

        # Calculate midpoints of the bottom edge of bounding boxes
        mid_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        bottom_y = bboxes[:, 3]

        footprints = np.stack((mid_x, bottom_y), axis=-1).astype(int).reshape(-1, 2)
        footprints = [footprints.tolist()]  # Convert to a list of lists format for polylines

        self.polylines(footprints, color=color, width=2)


class NullApp(App):
    def create_window(self, title, size) -> Window:
        del title  # unused
        del size  # unused
        return Window(None, self._is_closed)

    def start_thread(self, target, args=(), kwargs={}, name=None):
        del name  # unused
        target(*args, **kwargs)

    def _run(self, interval=1 / 30):
        del interval  # unused

    def _destroy_all_windows(self):
        pass


def calculate_speedometer_text(metric: inf_tracers.TraceMetric):
    text = f'{round(metric.value, 1):-5}' if metric.max < 100.0 else f'{round(metric.value):-4}'
    text += metric.unit
    return text


def calculate_speedometer_needle_pos(metric: inf_tracers.TraceMetric):
    limits = 90 - 45, 270 + 45
    dial_range = limits[1] - limits[0]
    angle = limits[0] + (metric.value / max(1, metric.max)) * dial_range
    # input is angle in degrees clockwise from 6 o'clock
    # mapped to ImageDraw angles starting from 3 o'clock
    return (angle + 90) % 360


class SpeedometerMetrics:
    needle_color = (255, 0, 0, 255)
    text_color = (255, 255, 255, 255)

    def __init__(self, canvas_size, index):
        _size = min(canvas_size)
        self.radius = _size // 10
        self.diameter = self.radius * 2
        xoffset = index * round(self.diameter * 1.1)
        self.top_left = (_size // 20 + xoffset, _size // 20)
        self.needle_radius = self.radius - _size // 48
        self.text_offset = self.radius * 2 // 5
        self.text_size = _size // 35

    @property
    def center(self):
        return (self.top_left[0] + self.radius, self.top_left[1] + self.radius)


class MetaCache:
    '''When inference skip frames is enabled, meta will be None on those frames
    that are skipped.

    This cache stores the last non-None meta for each stream.
    '''

    def __init__(self):
        self._last = {}

    def get(
        self, stream_id, meta_map: Mapping[str, AxTaskMeta] | None
    ) -> tuple[bool, Mapping[str, AxTaskMeta]]:
        '''Return the meta for given stream as (cached, meta_map).

        If meta is valid then the cache is updated. Otherwise the cached value is returned.

        Correctly handles stream_id which includes `__fps__` and other special metas.
        '''
        cached = False
        if stream_id == 0 and meta_map:
            # stream_id 0 always contains fps/latency etc. so we need to separate that
            meta_metas = {k: v for k, v in meta_map.items() if k.startswith('__')}
            actual_metas = {k: v for k, v in meta_map.items() if not k.startswith('__')}
            if actual_metas:
                meta_map = self._last[stream_id] = actual_metas
            else:
                meta_map = self._last.get(stream_id, {})
                cached = True
            meta_map = {**meta_metas, **meta_map}

        elif meta_map:
            meta_map = self._last[stream_id] = meta_map
        else:
            meta_map = self._last.get(stream_id, {})
            cached = True
        return cached, meta_map
