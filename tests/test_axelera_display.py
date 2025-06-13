# Copyright Axelera AI, 2025
import builtins
import contextlib
import os
import re
import sys
from unittest.mock import patch

import numpy as np
import pytest

from axelera import types
from axelera.app import config, display, display_console, display_cv, inf_tracers, utils


def test_null_app_with_data():
    with display.App(visible=False) as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        def t():
            wnd.show(i, meta)
            wnd.close()

        app.start_thread(t)
        app.run()
        assert wnd.is_closed == True


def test_wrong_type_image():
    with display.App(visible=False) as app:
        wnd = app.create_window('test', (640, 480))
        i = np.zeros((480, 640, 3), np.uint8)
        meta = object()

        def t():
            with pytest.raises(TypeError, match='Expected axelera.types.Image'):
                wnd.show(i, meta)
            wnd.close()

        app.start_thread(t)
        app.run()
        assert wnd.is_closed == True


class MockApp(display.App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received = []

    def _create_new_window(self, q, title, size):
        return title

    def _destroy_all_windows(self):
        pass

    def _run(self, interval=1 / 30):
        while 1:
            self._create_new_windows()
            new = display_cv._read_new_data(self._wnds, self._queues)
            if new is display.SHUTDOWN:
                return
            self.received.extend(new)
            if self.has_thread_completed:
                return


def test_mock_window_creation():
    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        app.start_thread(wnd.close)
        app.run()
        assert wnd.is_closed == True
    assert app.received == []


def test_mock_window_creation_with_data():
    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        def t():
            wnd.show(i, meta)

        app.start_thread(t)
        app.run()
        assert wnd.is_closed == True
    assert app.received == [('test', display._Frame(0, i, meta))]


def test_speedometer_metrics():
    m = display.SpeedometerMetrics((1000, 2000), 0)
    assert m.top_left == (50, 50)
    assert m.radius == 100
    assert m.needle_radius == 80
    assert m.center == (150, 150)
    assert m.diameter == 200
    assert m.text_offset == 40
    assert m.text_size == 28


@pytest.mark.parametrize(
    'metric, text, needle_pos',
    [
        (inf_tracers.TraceMetric('k', 'title', 0.0, 99.0), '  0.0', 90 + 45),
        (inf_tracers.TraceMetric('k', 'title', 10.0, 100.0), '  10', 162.0),
        (inf_tracers.TraceMetric('k', 'title', 99.0, 110.0), '  99', 18.0),
        (inf_tracers.TraceMetric('k', 'title', 100.1, 110.0), ' 100', 20.7),
        (inf_tracers.TraceMetric('k', 'title', 10.0, 100.0, '%'), '  10%', 162.0),
        (inf_tracers.TraceMetric('k', 'title', 25.0, 50.0, '%'), ' 25.0%', 270.0),
        (inf_tracers.TraceMetric('k', 'title', 99.0, 100.0, '%'), '  99%', 42.3),
    ],
)
def test_text_and_needle(metric, text, needle_pos):
    assert text == display.calculate_speedometer_text(metric)
    assert needle_pos == round(display.calculate_speedometer_needle_pos(metric), 1)


def test_meta_cache():
    mc = display.MetaCache()
    meta0_0 = {'a': 10, 'b': 20, '__fps__': 30}
    meta1_0 = {'a': 11, 'b': 21}
    meta0_1 = {'__fps__': 31}
    meta0_2 = {'a': 110, 'b': 120, '__fps__': 32}
    meta1_2 = {'a': 111, 'b': 121}
    meta0_3 = {'__fps__': 33}
    assert (False, meta0_0) == mc.get(0, meta0_0)
    assert (False, meta1_0) == mc.get(1, meta1_0)
    assert (True, {'a': 10, 'b': 20, '__fps__': 31}) == mc.get(0, meta0_1)
    assert (True, {'a': 11, 'b': 21}) == mc.get(1, None)
    assert (False, {'a': 110, 'b': 120, '__fps__': 32}) == mc.get(0, meta0_2)
    assert (False, {'a': 111, 'b': 121}) == mc.get(1, meta1_2)
    assert (True, {'a': 110, 'b': 120, '__fps__': 33}) == mc.get(0, meta0_3)
    assert (True, {'a': 111, 'b': 121}) == mc.get(1, None)


class A:
    pass


@pytest.mark.parametrize(
    'prop,value,exp,got',
    [
        ('title', 0, 'str', 'int'),
        ('grayscale', '0', 'float | int | bool', 'str'),
        ('grayscale', A(), 'float | int | bool', 'A'),
        ('bbox_label_format', 0, 'str', 'int'),
    ],
)
def test_options_invalid_type(caplog, prop, value, exp, got):
    opts = display.Options()
    opts.update(**{prop: value})
    assert f'Expected {exp} for Options.{prop}, but got {got}' in caplog.text


@pytest.mark.parametrize(
    'props,err',
    [
        (['titel'], 'Options.titel'),
        (['badger', 'mushroom'], 'Options.badger, mushroom'),
    ],
)
def test_options_invalid_prop(caplog, props, err):
    with caplog.at_level('INFO'):
        opts = display.Options()
        opts.update(**dict.fromkeys(props, 0))
        s = 's' if ',' in err else ''
        assert f'Unsupported option{s} : {err}' in caplog.text


@pytest.mark.parametrize(
    'prop,value,got',
    [
        ('title', 'test', 'test'),
        ('grayscale', 0.5, 0.5),
        ('grayscale', 0, 0.0),
        ('grayscale', True, 1.0),
        ('bbox_label_format', 'test', 'test'),
        ('bbox_label_format', '', ''),
    ],
)
def test_options_valid(prop, value, got):
    opts = display.Options()
    opts.update(**{prop: value})
    assert getattr(opts, prop) == got


disable = config.HardwareEnable.disable
detect = config.HardwareEnable.detect
enable = config.HardwareEnable.enable


@pytest.mark.parametrize(
    'display_opt, exp_class',
    [
        (False, display.NullApp),
        ('none', display.NullApp),
        ('opencv', display_cv.CVApp),
        ('console', display_console.ConsoleApp),
    ],
)
@pytest.mark.parametrize('detect_state', [disable, enable, detect])
def test_find_display_class_non_opengl(display_opt, exp_class, detect_state):
    #  opengl enable state should not affect opencv/console/none, nor should they try to detect it
    with patch.object(utils, 'is_opengl_available', return_value=False) as is_opengl_available:
        with patch.dict(os.environ, {'DISPLAY': ''}):
            assert display._find_display_class(display_opt, detect_state) is exp_class
        with patch.dict(os.environ, {'DISPLAY': 'something'}):
            assert display._find_display_class(display_opt, detect_state) is exp_class
    is_opengl_available.assert_not_called()


def test_find_display_class_bad_option():
    with pytest.raises(ValueError, match='Invalid display option'):
        display._find_display_class('bad_option', detect)


def test_find_display_class_auto_opengl_disabled():
    with patch.object(utils, 'is_opengl_available', return_value=False) as is_opengl_available:
        with patch.dict(os.environ, {'DISPLAY': ''}):
            assert display._find_display_class('auto', disable) is display_console.ConsoleApp
        with patch.dict(os.environ, {'DISPLAY': 'something'}):
            assert display._find_display_class('auto', disable) is display_cv.CVApp
        with patch.dict(os.environ, {'DISPLAY': ''}):
            assert display._find_display_class(True, disable) is display_console.ConsoleApp
    is_opengl_available.assert_not_called()


@contextlib.contextmanager
def _mock_display_gl_import(succeeds=True):
    # importing display_gl is a little risky, so we mock it
    # (we should probably make it less risky to import display_gl!!!)

    orig_import = builtins.__import__

    class package:
        class display_gl:
            class GLApp:
                pass

    class WasImported:
        was_imported = False

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if 'display_gl' in fromlist:
            if succeeds:
                return package
            raise ImportError('display_gl import failed')
        return orig_import(name, globals, locals, fromlist, level)

    with patch.dict(sys.modules, {'axelera.app.display_gl': None}):
        del sys.modules['axelera.app.display_gl']
        with patch.object(builtins, '__import__', new_import):
            yield WasImported


def test_find_display_class_auto_opengl_enabled():
    # we should not try to detect opengl if explicitly requested or --enable-opengl specifed
    with patch.object(utils, 'is_opengl_available') as m:
        with _mock_display_gl_import():
            assert display._find_display_class('auto', enable).__name__ == 'GLApp'
            assert display._find_display_class('opengl', enable).__name__ == 'GLApp'
            assert display._find_display_class('opengl', detect).__name__ == 'GLApp'

    m.assert_not_called()


def test_find_display_class_gl_import_succeeds():
    with patch.object(utils, 'is_opengl_available', return_value=True) as m:
        with _mock_display_gl_import():
            assert display._find_display_class('auto', detect).__name__ == 'GLApp'

    m.assert_called_once()


def test_find_display_class_opengl_import_fails():
    with patch.dict(os.environ, {'DISPLAY': ''}):
        with patch.object(utils, 'is_opengl_available', return_value=True) as m:
            with _mock_display_gl_import(succeeds=False):
                with pytest.raises(RuntimeError, match=r'Failed to init.*\n.*try exporting'):
                    display._find_display_class('opengl', detect)
    with patch.dict(os.environ, {'DISPLAY': 'something'}):
        with patch.object(utils, 'is_opengl_available', return_value=True) as m:
            with _mock_display_gl_import(succeeds=False):
                with pytest.raises(RuntimeError, match=r'Failed to init.*\n.*something'):
                    display._find_display_class('opengl', detect)


def test_find_display_class_auto_import_fails(caplog):
    with patch.dict(os.environ, {'DISPLAY': ''}):
        with patch.object(utils, 'is_opengl_available', return_value=True) as m:
            with _mock_display_gl_import(succeeds=False):
                display._find_display_class('auto', detect)
    assert any('Failed to init' in x for x in caplog.messages)


def test_display_text():
    expected_text = 'text'
    expected_position = ('50%', '50%')

    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        wnd.text(expected_position, expected_text)

        def t():
            wnd.show(i, meta)

        app.start_thread(t)
        app.run()
        assert wnd.is_closed == True
    assert app.received[0][0] == 'test'
    assert isinstance(app.received[0][1], display._Text)
    assert app.received[0][1].text == expected_text
    assert app.received[0][1].position == expected_position
    assert app.received[1] == ('test', display._Frame(0, i, meta))


def test_display_image():
    expected_path = '/path/to/image.png'
    expected_position = ('50%', '50%')

    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        wnd.image(expected_position, expected_path)

        def t():
            wnd.show(i, meta)

        app.start_thread(t)
        app.run()
        assert wnd.is_closed == True
    assert app.received[0][0] == 'test'
    assert isinstance(app.received[0][1], display._Image)
    assert app.received[0][1].path == expected_path
    assert app.received[0][1].position == expected_position
    assert app.received[1] == ('test', display._Frame(0, i, meta))


def test_layer_handle_init():
    expected_text = 'text'
    expected_position = ('50%', '50%')

    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        with patch('axelera.app.display.uuid') as mock:
            mock.uuid4.return_value = '1234'
            handle = wnd.text(expected_position, expected_text)

        def t():
            wnd.show(i, meta)

        app.start_thread(t)
        app.run()
    assert handle['text'] == expected_text
    assert handle['position'] == expected_position
    assert handle._window == wnd
    assert handle._Message == display._Text
    assert handle.id == '1234'
    assert handle.visible == True


def test_layer_handle_set_fields():
    txt0 = 'text'
    pos0 = ('50%', '50%')
    txt1 = 'new_text'
    pos1 = ('10%', '10%')

    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        with patch('axelera.app.display.uuid') as mock:
            mock.uuid4.return_value = '1234'
            handle = wnd.text(pos0, txt0)

        handle.set(text=txt1, position=pos1)

        def t():
            wnd.show(i, meta)

        app.start_thread(t)
        app.run()

    assert app.received[0][0] == 'test'
    assert isinstance(app.received[0][1], display._Text)
    assert app.received[0][1].id == '1234'
    assert app.received[0][1].text == txt0
    assert app.received[0][1].position == pos0

    assert app.received[1][0] == 'test'
    assert isinstance(app.received[1][1], display._Text)
    assert app.received[1][1].id == '1234'
    assert app.received[1][1].text == txt1
    assert app.received[1][1].position == pos1

    assert handle['text'] == txt1
    assert handle['position'] == pos1
    assert handle._window == wnd
    assert handle._Message == display._Text
    assert handle.id == '1234'
    assert handle.visible == True


def test_layer_handle_set_fields_dict_style():
    txt0 = 'text'
    pos0 = ('50%', '50%')
    txt1 = 'new_text'
    pos1 = ('10%', '10%')

    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        with patch('axelera.app.display.uuid') as mock:
            mock.uuid4.return_value = '1234'
            handle = wnd.text(pos0, txt0)

        handle['text'] = txt1
        handle['position'] = pos1

        def t():
            wnd.show(i, meta)

        app.start_thread(t)
        app.run()

    assert app.received[0][0] == 'test'
    assert isinstance(app.received[0][1], display._Text)
    assert app.received[0][1].id == '1234'
    assert app.received[0][1].text == txt0
    assert app.received[0][1].position == pos0

    assert app.received[1][0] == 'test'
    assert isinstance(app.received[1][1], display._Text)
    assert app.received[1][1].id == '1234'
    assert app.received[1][1].text == txt1
    assert app.received[1][1].position == pos0

    assert app.received[2][0] == 'test'
    assert isinstance(app.received[2][1], display._Text)
    assert app.received[2][1].id == '1234'
    assert app.received[2][1].text == txt1
    assert app.received[2][1].position == pos1

    assert handle['text'] == txt1
    assert handle['position'] == pos1
    assert handle._window == wnd
    assert handle._Message == display._Text
    assert handle.id == '1234'
    assert handle.visible == True


def test_layer_handle_hide():
    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        with patch('axelera.app.display.uuid') as mock:
            mock.uuid4.return_value = '1234'
            handle = wnd.text(('50%', '50%'), 'text')

        handle.hide(fadeout=5)

        def t():
            wnd.show(i, meta)

        app.start_thread(t)
        app.run()

    assert handle.visible == False

    assert app.received[1][0] == 'test'
    assert isinstance(app.received[1][1], display._Delete)
    assert app.received[1][1].id == '1234'
    assert app.received[1][1].fadeout == 5


def test_layer_handle_set_when_hidden():
    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        with patch('axelera.app.display.uuid') as mock:
            mock.uuid4.return_value = '1234'
            handle = wnd.text(('50%', '50%'), 'text')

        handle.hide()
        handle.set(text='not sent')
        handle.set(position=('10%', '10%'))
        handle.set(text='will be sent')
        handle.show()

        def t():
            wnd.show(i, meta)

        app.start_thread(t)
        app.run()

    assert handle.visible == True

    # Verify that messages are only sent when the handle is shown.
    # Hence even though three updates are called when hidden, we only
    # get one more text message after the delete. Which is when show
    # is called, but current state modified during the hide is respected.
    assert isinstance(app.received[0][1], display._Text)
    assert isinstance(app.received[1][1], display._Delete)
    assert isinstance(app.received[2][1], display._Text)
    assert isinstance(app.received[3][1], display._Frame)
    assert app.received[2][1].text == 'will be sent'
    assert app.received[2][1].position == ('10%', '10%')

    assert len(app.received) == 4


def test_layer_handle_warn_on_set_bad_key(caplog):
    caplog.set_level('WARNING')
    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        with patch('axelera.app.display.uuid') as mock:
            mock.uuid4.return_value = '1234'
            handle = wnd.text(('50%', '50%'), 'text')

        handle.set(bad_key='not sent')

        def t():
            wnd.show(i, meta)

        app.start_thread(t)
        app.run()

    assert re.match(
        r"WARNING .* Cannot set attribute 'bad_key' for LayerHandle. id=1234, message_type=_Text",
        caplog.text,
    )

    # Check the bad update didn't result in a message being sent.
    assert len(app.received) == 2
    assert isinstance(app.received[0][1], display._Text)
    assert isinstance(app.received[1][1], display._Frame)


def test_layer_handle_error_on_get_bad_key():
    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))

        with patch('axelera.app.display.uuid') as mock:
            mock.uuid4.return_value = '1234'
            handle = wnd.text(('50%', '50%'), 'text')

        with pytest.raises(
            KeyError,
            match="Cannot get attribute 'bad_key' from LayerHandle. id=1234, message_type=_Text",
        ) as e:
            _ = handle['bad_key']


@pytest.mark.parametrize(
    'pt, expected_pt, expected_format',
    [
        (('10px', '100px'), (10, 100), 'image_pixels'),
        (('0px', '0px'), (0, 0), 'image_pixels'),
        (('100.0px', '100.0px'), (100, 100), 'image_pixels'),
        (('25%', '50%'), (0.25, 0.5), 'relative'),
        (('0%', '0%'), (0.0, 0.0), 'relative'),
        (('100%', '100%'), (1.0, 1.0), 'relative'),
        (('37.5%', '43.75%'), (0.375, 0.4375), 'relative'),
        (('100px', '50%'), (100, 0.5), 'mixed'),
        (('0px', '0%'), (0, 0.0), 'mixed'),
        (('100px', '50%'), (100, 0.5), 'mixed'),
    ],
)
def test_coords_type_cast(pt, expected_pt, expected_format):
    got = display.Coords(*pt)
    assert got == expected_pt
    assert got.format == expected_format


@pytest.mark.parametrize(
    'pt',
    [
        (0.1, 0.2),
        (0.0, 0.0),
        (1.0, 1.0),
        (1.5, 2.5),
    ],
)
def test_coords_rel(pt):
    got = display.Coords.rel(*pt)
    assert got == pt
    assert got.format == 'relative'


@pytest.mark.parametrize(
    'pt',
    [
        (100, 200),
        (0, 0),
        (1000, 1000),
        (500, 250),
    ],
)
def test_coords_px(pt):
    got = display.Coords.px(*pt)
    assert got == pt
    assert got.format == 'image_pixels'


@pytest.mark.parametrize(
    'x, y, format',
    [
        ('10px', '10px', 'image_pixels'),
        ('10%', '10%', 'relative'),
        ('10px', '10%', 'mixed'),
    ],
)
def test_coords_format(x, y, format):
    assert display.Coords(x, y).format == format


@pytest.mark.parametrize(
    'pt, image_size, expected',
    [
        (('25px', '50px'), (200, 100), (0.125, 0.5)),
        (('0px', '0px'), (200, 100), (0.0, 0.0)),
        (('200px', '100px'), (200, 100), (1.0, 1.0)),
        (('25%', '50%'), None, (0.25, 0.5)),
        (('0%', '0%'), None, (0.0, 0.0)),
        (('100%', '100%'), None, (1.0, 1.0)),
        (('37.5%', '43.75%'), None, (0.375, 0.4375)),
        (('37.5%', '43.75%'), (200, 100), (0.375, 0.4375)),
        (('100px', '50%'), (200, 100), (0.5, 0.5)),
        (('0px', '0%'), (200, 100), (0.0, 0.0)),
    ],
)
def test_coords_as_rel(pt, image_size, expected):
    assert display.Coords(*pt).as_rel(image_size) == expected


@pytest.mark.parametrize(
    'pt, image_size, expected',
    [
        (('25px', '50px'), None, (25, 50)),
        (('25px', '50px'), (200, 100), (25, 50)),
        (('0px', '0px'), None, (0, 0)),
        (('200px', '100px'), None, (200, 100)),
        (('25%', '50%'), (200, 100), (50, 50)),
        (('0%', '0%'), (200, 100), (0, 0)),
        (('100%', '100%'), (200, 100), (200, 100)),
        (('37.5%', '43.75%'), (200, 100), (75, 43)),
        (('100px', '50%'), (200, 100), (100, 50)),
        (('0px', '0%'), (200, 100), (0, 0)),
    ],
)
def test_coords_as_px(pt, image_size, expected):
    assert display.Coords(*pt).as_px(image_size) == expected


@pytest.mark.parametrize(
    'x,y',
    [
        ('10', '10'),
        ('', ''),
        ('px10', 'px10'),
        ('px', 'px'),
        ('%', '%'),
        ('nonsensepx', 'nonsensepx'),
        ('nonsense%', 'nonsense%'),
        ('10px10px', '10px10px'),
        ('10%10%', '10%10%'),
        ('10px%', '10px%'),
        ('10%px', '10%px'),
        ('100.0', '100.0'),
    ],
)
def test_coords_init_bad_format(x, y):
    with pytest.raises(ValueError, match='Invalid coordinate'):
        display.Coords(x, y)


@pytest.mark.parametrize(
    'x,y',
    [
        (10, 10),
        (0.1, 0.1),
        ('10px', 10),
        (0.1, '10%'),
        (None, None),
    ],
)
def test_coords_init_bad_type(x, y):
    with pytest.raises(TypeError, match='x and y must be strings'):
        display.Coords(x, y)


@pytest.mark.parametrize(
    'x,y',
    [
        ('10px', '10px'),
        ('10%', '10%'),
        ('10px', 10),
        (0.1, '10%'),
        (None, None),
    ],
)
def test_coords_rel_bad_type(x, y):
    with pytest.raises(TypeError, match='x and y must be floats'):
        display.Coords.rel(x, y)


@pytest.mark.parametrize(
    'x,y',
    [
        ('10px', '10px'),
        ('10%', '10%'),
        ('10px', 10),
        (0.1, '10%'),
        (None, None),
    ],
)
def test_coords_px_bad_type(x, y):
    with pytest.raises(TypeError, match='x and y must be ints'):
        display.Coords.px(x, y)


def test_coords_as_px_no_image_size():
    with pytest.raises(ValueError, match='image_size must be provided'):
        display.Coords('10%', '10%').as_px()


def test_coords_as_rel_no_image_size():
    with pytest.raises(ValueError, match='image_size must be provided'):
        display.Coords('10px', '10px').as_rel()
