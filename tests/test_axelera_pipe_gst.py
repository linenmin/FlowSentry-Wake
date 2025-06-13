# Copyright Axelera AI, 2023
# Construct GStreamer application pipeline
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import call, patch

import pytest

from axelera.app import gst_builder, operators, pipe, pipeline
from axelera.app.pipe import gst, gst_helper, io

# isort: off
from gi.repository import Gst

# isort: on


def _sorted(elements):
    return sorted(elements, key=lambda e: e.get_name())


@pytest.mark.parametrize('num_sinks', [0, 1, 2, 4])
def test_iteration_appsinks(num_sinks):
    if not Gst.is_initialized():
        Gst.init(None)

    pipeline = Gst.Pipeline()
    videosrc = Gst.ElementFactory.make('videotestsrc', 'source')
    pipeline.add(videosrc)
    appsinks = []
    for n in range(num_sinks):
        element = Gst.ElementFactory.make('appsink', f'sink{n}')
        pipeline.add(element)
        appsinks.append(element)

    assert [videosrc] == gst_helper.list_all_by_element_factory_name(pipeline, 'videotestsrc')
    assert _sorted(appsinks) == _sorted(
        gst_helper.list_all_by_element_factory_name(pipeline, 'appsink')
    )


def _expected_rtsp_input_builder(source_id_offset=0):
    exp = gst_builder.Builder()
    exp.rtspsrc(
        {'user-id': '', 'user-pw': ''},
        location='rtsp://localhost:8554/test',
        latency=500,
        connections={'stream_%u': f'rtspcapsfilter{source_id_offset}.sink'},
    )
    exp.capsfilter(
        {'caps': 'application/x-rtp,media=video'}, name=f'rtspcapsfilter{source_id_offset}'
    )
    exp.decodebin(
        {'expose-all-streams': False, 'force-sw-decoders': False},
        caps='video/x-raw(ANY)',
        connections={'src_%u': f'decodebin-link{source_id_offset}.sink'},
    )
    # exp.queue(name='queue_in0')
    exp.axinplace(
        lib='libinplace_addstreamid.so',
        mode='meta',
        options=f'stream_id:{source_id_offset}',
        name=f'decodebin-link{source_id_offset}',
    )
    return exp


def test_build_input_with_normal_input():
    pipein = io.SinglePipeInput('gst', 'rtsp://localhost:8554/test')
    builder = gst_builder.Builder()
    tasks = [
        pipeline.AxTask(
            'task0',
            operators.Input(),
        )
    ]
    pipe.gst._build_input_pipeline(builder, tasks, pipein, 0)
    exp = _expected_rtsp_input_builder()
    exp.videoconvert()
    exp.capsfilter({'caps': 'video/x-raw,format=RGBA'})
    exp.queue(connections={'src': 'inference-task0.sink_%u'})
    assert list(builder) == list(exp)


def test_build_input_with_normal_input_with_sourceid():
    pipein = io.SinglePipeInput('gst', 'rtsp://localhost:8554/test')
    builder = gst_builder.Builder()
    tasks = [
        pipeline.AxTask(
            'task0',
            operators.Input(),
        )
    ]
    pipe.gst._build_input_pipeline(builder, tasks, pipein, 4)
    exp = _expected_rtsp_input_builder(4)
    exp.videoconvert()
    exp.capsfilter({'caps': 'video/x-raw,format=RGBA'})
    exp.queue(connections={'src': 'inference-task0.sink_%u'})
    assert list(builder) == list(exp)


def test_build_input_with_image_processing():
    pipein = io.SinglePipeInput('gst', 'rtsp://localhost:8554/test')
    builder = gst_builder.Builder()
    matrix = '0.6,0.3,-67.2,-0.4,0.5,493.6,0.0,0.0,1.0'
    tasks = [
        pipeline.AxTask(
            'task0',
            operators.InputWithImageProcessing(
                image_processing=[
                    operators.custom_preprocessing.Perspective(matrix),
                ],
            ),
        )
    ]
    pipe.gst._build_input_pipeline(builder, tasks, pipein, 0)
    exp = _expected_rtsp_input_builder()
    exp.videoconvert()
    exp.capsfilter({'caps': 'video/x-raw,format=RGBA'})
    exp.videoconvert()
    exp.capsfilter({'caps': 'video/x-raw,format=RGBA'})
    exp.perspective(
        matrix='1.1904761904761905,-0.7142857142857143,432.5714285714286,'
        '0.9523809523809526,1.4285714285714286,-641.1428571428572,0.0,0.0,1.0'
    )
    exp.queue(connections={'src': 'inference-task0.sink_%u'})
    assert list(builder) == list(exp)


def test_save_axnet_files():
    elements = [
        {
            'instance': 'axinferencenet',
            'name': 'task0',
            'model': '/cwd/build/modelA.json',
            'p0_options': 'something:a;classlabels_file:bob;;mode:meta',
        },
        {
            'instance': 'axinferencenet',
            'name': 'task1',
            'model': '/abs/build/modelB.json',
            'p0_options': 'something:a;classlabels_file:bob',
        },
        {
            'instance': 'axinferencenet',
            'name': 'task2',
            'model': '/cwd/build/modelC.json',
            'p0_options': 'classlabels_file:bob;mode:meta',
        },
        {
            'instance': 'axinferencenet',
            'name': 'task3',
            'model': '/cwd/build/modelD.json',
            'p0_options': 'classlabels_file:bob',
        },
    ]
    task_names = ['model0', 'model1', 'model2', 'model3']
    with patch.object(os, 'getcwd', return_value='/cwd'):
        with patch.object(Path, 'write_text') as m:
            gst._save_axnet_files(elements, task_names, Path('/abs'))
        m.assert_has_calls(
            [
                call('model=build/modelA.json\np0_options=something:a;mode:meta'),
                call('model=/abs/build/modelB.json\np0_options=something:a'),
                call('model=build/modelC.json\np0_options=mode:meta'),
                call('model=build/modelD.json\np0_options='),
            ]
        )
