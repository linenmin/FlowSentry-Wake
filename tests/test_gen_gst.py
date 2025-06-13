# Copyright Axelera AI, 2024
import contextlib
import dataclasses
import io
import itertools
from pathlib import Path
import platform
import re
import tempfile
from unittest.mock import Mock, mock_open, patch

import cv2
import pytest

onnxruntime = pytest.importorskip(
    'onnxruntime'
)  # TODO we should not need onnxruntime in the runtime tests
import yaml
from yaml_clean import yaml_clean

from axelera import types
from axelera.app import config, network, operators, pipe, transforms
from axelera.app.pipe import gst_helper

# isort: off
from gi.repository import Gst

# isort: on


def _ncore(manifest: types.Manifest, ncores: int) -> types.Manifest:
    inp = (ncores,) + manifest.input_shapes[0][1:]
    out = (ncores,) + manifest.output_shapes[0][1:]
    return dataclasses.replace(manifest, input_shapes=[inp], output_shapes=[out])


SQ_IN_YAML = 'ax_models/model_cards/torchvision/classification/squeezenet1.0-imagenet-onnx.yaml'
SQ_MANIFEST = types.Manifest(
    'sq',
    ((0.01863, -14), (0.01863, -14), (0.01863, -14)),
    ((0.9, 0),),
    input_shapes=[(1, 224, 224, 3)],
    output_shapes=[(1, 1000)],
    n_padded_ch_inputs=[(1, 2, 3, 4)],
    model_lib_file='model.json',
)
SQ_MANIFEST4 = _ncore(SQ_MANIFEST, 4)

RN34_IN_YAML = 'ax_models/model_cards/torchvision/classification/resnet34-imagenet-onnx.yaml'
RN50_IN_YAML = 'ax_models/model_cards/torchvision/classification/resnet50-imagenet-onnx.yaml'
RN_MANIFEST = types.Manifest(
    'sq',
    ((0.01863, -14), (0.01863, -14), (0.01863, -14)),
    ((0.9, 0),),  # dequant?
    input_shapes=[(1, 224, 224, 3)],
    output_shapes=[(1, 1000)],
    n_padded_ch_inputs=[(1, 2, 3, 4)],
    model_lib_file='lib_export/model.json',
)
RN_MANIFEST4 = _ncore(RN_MANIFEST, 4)

YOLOV5S_V5_IN_YAML = 'ax_models/model_cards/yolo/object_detection/yolov5s-relu-coco-onnx.yaml'
YOLOV5S_V5_MANIFEST = types.Manifest(
    'yolo',
    [(0.003919653594493866, -128)],
    [[0.08142165094614029, 70], [0.09499982744455338, 82], [0.09290479868650436, 66]],
    input_shapes=[(1, 320, 320, 64)],
    output_shapes=[[1, 20, 20, 256], [1, 40, 40, 256], [1, 80, 80, 256]],
    n_padded_ch_inputs=[(0, 0, 0, 0, 0, 0, 0, 52)],
    n_padded_ch_outputs=[
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
    ],
    model_lib_file='lib_export/model.json',
    postprocess_graph='lib_export/post_process.onnx',
    preprocess_graph='tests/assets/focus_preprocess_graph.onnx',
)

YOLOV5S_V7_IN_YAML = 'ax_models/model_cards/yolo/object_detection/yolov5s-v7-coco-onnx.yaml'
YOLOV5S_V7_MANIFEST = types.Manifest(
    'yolo',
    [(0.003919653594493866, -128)],
    [[0.08142165094614029, 70], [0.09499982744455338, 82], [0.09290479868650436, 66]],
    input_shapes=[(1, 640, 640, 64)],
    output_shapes=[[1, 20, 20, 256], [1, 40, 40, 256], [1, 80, 80, 256]],
    n_padded_ch_inputs=[(0, 0, 0, 0, 0, 0, 0, 61)],
    n_padded_ch_outputs=[
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
    ],
    model_lib_file='lib_export/model.json',
    postprocess_graph='lib_export/post_process.onnx',
)

YOLOV5S_V7_IN_YAML = 'ax_models/model_cards/yolo/object_detection/yolov5m-v7-coco-onnx.yaml'
YOLO_TRACKER_RN_IN_YAML = 'ax_models/reference/cascade/with_tracker/yolov5m-tracker-resnet50.yaml'
YOLOV5M_V7_MANIFEST = types.Manifest(
    'yolo',
    [(0.003919653594493866, -128)],
    [[0.0038571979384869337, -128], [0.0038748111110180616, -128], [0.0038069516886025667, -128]],
    input_shapes=[(1, 644, 656, 4)],
    output_shapes=[[1, 20, 20, 256], [1, 40, 40, 256], [1, 80, 80, 256]],
    n_padded_ch_inputs=[(0, 0, 2, 2, 2, 14, 0, 1)],
    n_padded_ch_outputs=[
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
    ],
    model_lib_file='lib_export/lib.so',
    postprocess_graph='',
)


YOLOV8POSE_YOLOV8N_IN_YAML = 'ax_models/reference/cascade/yolov8spose-yolov8n.yaml'
YOLOV8POSE_MANIFEST = types.Manifest(
    'yolov8spose-coco-onnx',
    [(0.003919653594493866, -128)],
    [
        [0.055154770612716675, -65],
        [0.05989416316151619, -65],
        [0.06476129591464996, -56],
        [0.10392487794160843, 109],
        [0.17826798558235168, 109],
        [0.16040770709514618, 107],
        [0.04365239664912224, 12],
        [0.057816002517938614, 7],
        [0.066075898706913, 15],
    ],
    input_shapes=[(1, 642, 656, 4)],
    output_shapes=[
        [1, 80, 80, 64],
        [1, 40, 40, 64],
        [1, 20, 20, 64],
        [1, 80, 80, 64],
        [1, 40, 40, 64],
        [1, 20, 20, 64],
        [1, 80, 80, 64],
        [1, 40, 40, 64],
        [1, 20, 20, 64],
    ],
    n_padded_ch_inputs=[(0, 0, 1, 1, 1, 15, 0, 1)],
    n_padded_ch_outputs=[
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 63),
        (0, 0, 0, 0, 0, 0, 0, 63),
        (0, 0, 0, 0, 0, 0, 0, 63),
        (0, 0, 0, 0, 0, 0, 0, 13),
        (0, 0, 0, 0, 0, 0, 0, 13),
        (0, 0, 0, 0, 0, 0, 0, 13),
    ],
    model_lib_file='lib_export/yolov8pose/model.json',
    postprocess_graph='lib_export/yolov8pose/post_process.onnx',
)
YOLOV8N_MANIFEST = types.Manifest(
    'yolov8n-coco-onnx',
    [(0.003919653594493866, -128)],
    [
        [0.08838965743780136, -60],
        [0.07353860884904861, -57],
        [0.07168316841125488, -44],
        [0.10592737793922424, 127],
        [0.15443256497383118, 117],
        [0.18016019463539124, 104],
    ],
    input_shapes=[(1, 642, 656, 4)],
    output_shapes=[
        [1, 80, 80, 64],
        [1, 40, 40, 64],
        [1, 20, 20, 64],
        [1, 80, 80, 128],
        [1, 40, 40, 128],
        [1, 20, 20, 128],
    ],
    n_padded_ch_inputs=[(0, 0, 1, 1, 1, 15, 0, 1)],
    n_padded_ch_outputs=[
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 48),
        (0, 0, 0, 0, 0, 0, 0, 48),
        (0, 0, 0, 0, 0, 0, 0, 48),
    ],
    model_lib_file='lib_export/yolov8n/model.json',
    postprocess_graph='lib_export/post_process.onnx',
)

SOMETEMPFILE = '/path/to/sometempfile.txt'


def mock_temp(*args, **kwargs):
    c = io.StringIO()
    c.name = SOMETEMPFILE
    return c


def _generate_pipeline(nn, input, output, hardware_caps):
    '''Construct gst E2E pipeline'''
    for task in nn.tasks:
        transforms.run_all_transformers(task.preprocess, hardware_caps=hardware_caps)

    nn.model_infos = network.ModelInfos()
    for task in nn.tasks:
        mi = task.model_info
        nn.model_infos.add_model(mi, manifest_path=Path('./mock_manifest_path'))

    dm = _mock_device_manager()
    assert ['metis-0:1:0'] == [d.name for d in dm.devices]
    task_graph = pipe.graph.DependencyGraph(nn.tasks)
    p = pipe.create_pipe(dm, 'gst', nn, Path('./'), hardware_caps, None, task_graph)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(tempfile, 'NamedTemporaryFile', mock_temp))
        # prevent ./gst_pipeline.yaml being written by tests
        stack.enter_context(patch.object(Path, 'write_text', return_value=None))
        stack.enter_context(patch.object(Path, 'exists', return_value=True))

        p.propagate_model_and_context_info()
        p.gen_end2end_pipe(input, output)

    return p.pipeline


class MockCapture:
    def __init__(self, path):
        pass

    def get(self, attr):
        attrs = {
            cv2.CAP_PROP_FRAME_WIDTH: 1600,
            cv2.CAP_PROP_FRAME_HEIGHT: 1200,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30,
        }
        return float(attrs[attr])

    def isOpened(self):
        return True

    def release(self):
        pass


def _create_pipein(*paths, **kwargs):
    assert len(paths) >= 1
    with patch.object(Path, 'exists', return_value=True):
        with patch.object(Path, 'is_file', return_value=True):
            with patch.object(cv2, 'VideoCapture', MockCapture):
                return pipe.io.MultiplexPipeInput('gst', paths, **kwargs)


def _video_path(out_name: str) -> Path:
    return Path(__file__).parent / 'golden-gst' / out_name


def _actual_path(out_name: str) -> Path:
    return Path(__file__).parent / 'out' / out_name


def _expected_path(out_name: str) -> Path:
    return Path(__file__).parent / 'exp' / out_name


def _prepare_expected(out_name, manifests, tasks, hardware_caps, gold_path=_video_path):
    exp = gold_path(out_name).read_text()
    extra_args = _expansion_params(manifests, tasks, hardware_caps)

    def do_subs(m):
        try:
            return str(extra_args[m.group(1)])
        except KeyError:
            return m.group(2) or m.group(0)

    exp = re.sub(r"{\{([^:\}]+)\s*(?::\s*([^}\n]*))?\}\}", do_subs, exp, flags=re.MULTILINE)
    return yaml_clean(exp)


def _write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or content != path.read_text():
        path.write_text(content)


def _compare_yaml(expected, actual, out_name, gold_path=_video_path):
    exp_path, actual_path = _expected_path(out_name), _actual_path(out_name)
    if expected != actual:
        _write_file(exp_path, expected)
        _write_file(actual_path, actual)
        a = actual_path.relative_to(Path.cwd())
        e = exp_path.relative_to(Path.cwd())
        g = gold_path(out_name).relative_to(Path.cwd())
        assert (
            expected == actual
        ), f'{a} {e} : generated yaml does not match gold, gold was generated from {g} '


def _mock_device_manager():
    m = Mock()
    m.name = 'metis-0:1:0'
    return Mock(devices=[m])


def _load_highlevel(requested_cores: int, path: str, *manifests):
    if len(manifests) == 1 and isinstance(manifests[0], (list, tuple)):
        manifests = manifests[0]
    nn = network.parse_network_from_path(path)
    network.restrict_cores(nn, 'gst', requested_cores, config.Metis.pcie)
    device_man = _mock_device_manager()
    for manifest, task in itertools.zip_longest(manifests, nn.tasks):
        task.model_info.manifest = manifest

        if 'YOLO' in task.model_info.extra_kwargs:
            task.model_info.extra_kwargs['YOLO']['anchors'] = [
                [1.25, 1.625, 2.0, 3.75, 4.125, 2.875],
                [1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375],
                [3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875],
            ]

        config_content = {
            "remove_quantization_of_inputs_outputs_from_graph": True,
            "apply_pword_padding": True,
            "remove_padding_and_layout_transform_of_inputs_outputs": True,
            "host_arch": "x86_64",
            "target": "axelera",
            "aipu_cores": 1,
            "io_location": "L2",
            "wgt_location": "L2",
            "quantize_only": False,
        }

        if task.is_dl_task:
            with patch('builtins.open', mock_open()):
                with patch('json.load', return_value=config_content):
                    task.inference = operators.Inference(
                        device_man=device_man,
                        compiled_model_dir=Path('build/manifest.json').parent,
                        model_name=task.model_info.name,
                        model=manifest,
                        input_tensor_layout=task.model_info.input_tensor_layout,
                        inference_op_config=task.inference_config,
                    )
                    # Only patch for the focus preprocess_graph test asset
                    if (
                        getattr(manifest, 'preprocess_graph', None)
                        == 'tests/assets/focus_preprocess_graph.onnx'
                    ):
                        task.inference.compiled_model_dir = Path('build')
                        manifest.model_lib_file = 'lib_export/model.json'
                    if 'YOLO' not in task.model_info.extra_kwargs:
                        task.inference._icdf_params = object()  # not None!
    return nn


def _pads(manifest):
    pads = list(manifest.n_padded_ch_inputs[0]) if manifest.n_padded_ch_inputs else [0] * 4
    if len(pads) == 4:
        t, l, b, r = pads
        pads = [0, 0, t, b, l, r, 0, 0]
    pads[7] = pads[7] - 1 if pads[7] in (1, 61) else pads[7]
    return ','.join(str(x) for x in pads)


def _expansion_params(manifests, tasks, hardware_caps):
    base = {f'input_video{n}': f'/path/to/src{n}.mp4' for n in range(8)}

    def add(name, values, replace_dot=False):
        _replace = lambda x: x.replace('.', '_') if replace_dot else x
        base.update({f'{name}{n}': _replace(v) for n, v in enumerate(values)})
        base[name] = values[0]

    add('model_lib', [f'build/{m.model_lib_file}' if m else '' for m in manifests])
    add('model_name', [t.model_info.name for t in tasks], replace_dot=True)
    add('task_name', [t.name for t in tasks], replace_dot=True)
    add('label_file', ['/path/to/sometempfile.txt' for _ in tasks])
    add('tracker_params_json', ['/path/to/sometempfile.txt' for _ in tasks])
    add('pads', [_pads(m) if m else '0,0,0,0,0,0,0,0' for m in manifests])
    add('quant_scale', [m.quantize_params[0][0] if m else 0 for m in manifests])
    add('quant_zeropoint', [m.quantize_params[0][1] if m else 0 for m in manifests])
    add('dequant_scale', [m.dequantize_params[0][0] if m else 0 for m in manifests])
    add('dequant_zeropoint', [m.dequantize_params[0][1] if m else 0 for m in manifests])
    post_proc = 'lib_cpu_post_processing.so'
    add(
        'post_model_lib',
        [Path(m.model_lib_file).parent / post_proc if m else '' for m in manifests],
    )
    add('input_w', [m.input_shapes[0][2] if m else 0 for m in manifests])
    add('input_h', [m.input_shapes[0][1] if m else 0 for m in manifests])
    return dict(
        base,
        force_sw_decoders=not hardware_caps.vaapi,
        prefix='',
        confidence_threshold=0.3,
        nms_threshold=0.5,
        class_agnostic=1,
        max_boxes=30000,
        nms_top_k=200 if 'ssd' in tasks[0].model_info.name.lower() else 300,
        sigmoid_in_postprocess=0,
    )


ALL = dataclasses.replace(config.HardwareCaps.ALL, aipu_cores=1)
ALL4 = dataclasses.replace(config.HardwareCaps.ALL, aipu_cores=4)
AIPU = dataclasses.replace(config.HardwareCaps.AIPU, aipu_cores=1)
AIPU4 = dataclasses.replace(config.HardwareCaps.AIPU, aipu_cores=4)
NONE = dataclasses.replace(config.HardwareCaps.NONE, aipu_cores=1)
OPENCL = dataclasses.replace(config.HardwareCaps.OPENCL, aipu_cores=1)
OPENCL4 = dataclasses.replace(config.HardwareCaps.OPENCL, aipu_cores=4)

gen_gst_marker = pytest.mark.parametrize(
    'caps, src, manifest, golden_template, num_inputs, proc, limit_fps',
    [
        (
            AIPU,
            YOLOV8POSE_YOLOV8N_IN_YAML,
            [YOLOV8POSE_MANIFEST, YOLOV8N_MANIFEST],
            'yolov8pose-yolov8n.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            OPENCL4,
            SQ_IN_YAML,
            SQ_MANIFEST,
            'opencl/classifier-imagenet.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            OPENCL,
            YOLO_TRACKER_RN_IN_YAML,
            [YOLOV5M_V7_MANIFEST, None, RN_MANIFEST],
            'opencl/yolov5m-tracker-resnet50.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            AIPU4,
            SQ_IN_YAML,
            SQ_MANIFEST4,
            'classifier-imagenet-4core-3streams.yaml',
            3,
            'x86_64',
            0,
        ),
        (
            AIPU4,
            RN50_IN_YAML,
            RN_MANIFEST4,
            'classifier-imagenet.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            AIPU4,
            RN50_IN_YAML,
            RN_MANIFEST4,
            'classifier-imagenet-limit-fps.yaml',
            1,
            'x86_64',
            15,
        ),
        (
            AIPU4,
            RN50_IN_YAML,
            RN_MANIFEST4,
            'classifier-imagenet-arm.yaml',
            1,
            'arm',
            0,
        ),
        (
            AIPU,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'yolov5s-axelera-coco-1stream.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            AIPU,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'yolov5s-axelera-coco-1stream.yaml',
            1,
            'arm',
            0,
        ),
        (
            AIPU,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'yolov5s-axelera-coco-4streams.yaml',
            4,
            'x86_64',
            0,
        ),
        (
            AIPU,
            'ax_models/reference/image_preprocess/yolov5s-v7-perspective-onnx.yaml',
            YOLOV5S_V5_MANIFEST,
            'yolov5s-v7-perspective-4streams.yaml',
            4,
            'x86_64',
            0,
        ),
        (
            OPENCL,
            'ax_models/reference/image_preprocess/yolov5s-v7-perspective-onnx.yaml',
            YOLOV5S_V5_MANIFEST,
            'opencl/yolov5s-v7-perspective-4streams.yaml',
            4,
            'x86_64',
            0,
        ),
    ],
)


@gen_gst_marker
def test_lowlevel_output(caps, src, manifest, golden_template, num_inputs, proc, limit_fps):
    nn = _load_highlevel(
        caps.aipu_cores, src, *([manifest] if not isinstance(manifest, list) else manifest)
    )
    inputs = [f'/path/to/src{i}.mp4' for i in range(num_inputs)]
    pipein = _create_pipein(
        *inputs,
        hardware_caps=caps,
        allow_hardware_codec=False,
        color_format=types.ColorFormat.RGB,
        specified_frame_rate=limit_fps,
    )
    with patch.object(platform, 'processor', return_value=proc):
        pipeout = pipe.PipeOutput()
        pipeline = _generate_pipeline(nn, pipein, pipeout, hardware_caps=caps)
    actual = yaml.dump([{'pipeline': pipeline}], sort_keys=False)
    manifests = manifest if isinstance(manifest, list) else [manifest]
    exp = _prepare_expected(golden_template, manifests, nn.tasks, hardware_caps=caps)
    _compare_yaml(exp, actual, golden_template)


@gen_gst_marker
def test_gst_pipeline_builder(caps, src, manifest, golden_template, num_inputs, proc, limit_fps):
    # For the sake of better code coverage and to ensure that what we created was maybe
    # approximately sensible lowlevel yaml, also build the pipeline, hweever this does
    # not work in tox env, so allow it to skip if it fails to create a gst element
    del golden_template
    nn = _load_highlevel(
        caps.aipu_cores, src, *([manifest] if not isinstance(manifest, list) else manifest)
    )
    inputs = [f'/path/to/src{i}.mp4' for i in range(num_inputs)]
    pipein = _create_pipein(
        *inputs,
        hardware_caps=caps,
        allow_hardware_codec=False,
        color_format=types.ColorFormat.RGB,
        specified_frame_rate=limit_fps,
    )
    with patch.object(platform, 'processor', return_value=proc):
        pipeout = pipe.PipeOutput()
        pipeline = _generate_pipeline(nn, pipein, pipeout, hardware_caps=caps)
    try:
        gst_helper.build_pipeline(pipeline)
    except Exception as e:
        if 'Failed to create element of type' in str(e):
            pytest.skip('axstreamer plugins not installed')
        raise
