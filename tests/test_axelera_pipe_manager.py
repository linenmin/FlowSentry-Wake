# Copyright Axelera AI, 2024

import os
import unittest.mock as mock
from unittest.mock import patch

import pytest

from axelera.app import config
from axelera.app.pipe import manager


def test_create_manager_from_model_name():
    with patch.dict(os.environ, {'AXELERA_FRAMEWORK': "."}):
        network_name = "mc-yolov5s-v7-coco"
        expected_path = "ax_models/model_cards/yolo/object_detection/yolov5s-v7-coco.yaml"
        output = manager._get_real_path_if_path_is_model_name(network_name)
        assert output == expected_path


def test_task_render_proxy():
    """Test the TaskProxy class which allows task-specific render settings."""
    # Create a mock task
    mock_task = mock.MagicMock()
    mock_task.name = 'detections'

    # Create a mock pipeline with the task
    mock_pipeline = mock.MagicMock()
    mock_pipeline.nn.tasks = [mock_task]

    # Create a mock PipeManager with a real RenderConfig
    pipe_manager = mock.MagicMock()
    pipe_manager._pipeline = mock_pipeline
    pipe_manager.pipeout = mock.MagicMock()
    # Pre-register the 'detections' task in RenderConfig
    pipe_manager.pipeout.render_config = config.RenderConfig(detections=config.TaskRenderConfig())

    # Create a TaskProxy for a specific task
    proxy = manager.TaskProxy(pipe_manager, 'detections')

    # Test that the set_render method works correctly and returns self
    result = proxy.set_render(show_annotations=False, show_labels=False)

    # Verify the task settings were updated in the RenderConfig
    task_settings = pipe_manager.pipeout.render_config.get('detections')
    assert task_settings is not None
    assert task_settings.show_annotations is False
    assert task_settings.show_labels is False

    # Verify that set_render returns self for method chaining
    assert result is proxy


def test_pipe_manager_get_attribute_for_task():
    """Test that PipeManager's __getattr__ returns a TaskProxy for tasks."""
    # Create a mock for the pipeline with tasks
    mock_pipeline = mock.MagicMock()
    mock_task = mock.MagicMock()
    mock_task.name = 'detections'
    mock_pipeline.nn.tasks = [mock_task]

    # Create a simplified PipeManager for testing
    class TestPipeManager(manager.PipeManager):
        def __init__(self):
            # Skip the actual initialization
            self._pipeline = mock_pipeline
            self.pipeout = mock.MagicMock()
            # Use a real RenderConfig
            self.pipeout.render_config = config.RenderConfig()

    # Create our test manager
    pipe_mgr = TestPipeManager()

    # Test getting a task by name
    task_proxy = pipe_mgr.detections

    # Verify it's a TaskProxy
    assert isinstance(task_proxy, manager.TaskProxy)
    assert task_proxy.task_name == 'detections'


def test_pipe_manager_set_render():
    """Test PipeManager's set_render method affects all tasks."""
    # Create a mock for the pipeline with tasks
    mock_pipeline = mock.MagicMock()
    mock_task1 = mock.MagicMock()
    mock_task1.name = 'task1'
    mock_task2 = mock.MagicMock()
    mock_task2.name = 'task2'
    mock_pipeline.nn.tasks = [mock_task1, mock_task2]

    # Create a simplified PipeManager for testing
    class TestPipeManager(manager.PipeManager):
        def __init__(self):
            # Skip the actual initialization
            self._pipeline = mock_pipeline
            self.pipeout = mock.MagicMock()
            # Use a real RenderConfig and pre-register tasks
            self.pipeout.render_config = config.RenderConfig(
                task1=config.TaskRenderConfig(),
                task2=config.TaskRenderConfig(),
            )

    # Create our test manager
    pipe_mgr = TestPipeManager()

    # Call set_render to update all tasks (use only keyword arguments)
    pipe_mgr.set_render(show_annotations=False, show_labels=False)

    # Test that the render_config was updated for all tasks
    render_config = pipe_mgr.pipeout.render_config

    # Check settings for both tasks
    for task_name in ['task1', 'task2']:
        task_settings = render_config.get(task_name)
        assert task_settings is not None
        assert task_settings.show_annotations is False
        assert task_settings.show_labels is False


def test_set_render_config_extra_keys():
    class DummyTask:
        def __init__(self, name):
            self.name = name
            self.task_render_config = config.TaskRenderConfig()

    nn = type('NN', (), {})()
    nn.tasks = [DummyTask('a'), DummyTask('b')]
    render_config = config.RenderConfig(
        a=config.TaskRenderConfig(), b=config.TaskRenderConfig(), c=config.TaskRenderConfig()
    )
    with pytest.raises(ValueError, match=r'render_config.*not in nn.tasks'):
        manager._set_render_config(nn, render_config)


def test_set_render_config_missing_keys():
    class DummyTask:
        def __init__(self, name):
            self.name = name
            self.task_render_config = config.TaskRenderConfig()

    nn = type('NN', (), {})()
    nn.tasks = [DummyTask('a'), DummyTask('b')]
    render_config = config.RenderConfig(a=config.TaskRenderConfig())
    result = manager._set_render_config(nn, render_config)
    assert 'b' in result._config
    assert isinstance(result._config['b'], config.TaskRenderConfig)


def test_set_render_config_none():
    class DummyTask:
        def __init__(self, name):
            self.name = name

    nn = type('NN', (), {})()
    nn.tasks = [DummyTask('a'), DummyTask('b')]
    # Use keyword-based initialization to pre-register tasks
    render_config = config.RenderConfig(a=config.TaskRenderConfig(), b=config.TaskRenderConfig())
    result = manager._set_render_config(nn, render_config)
    assert set(result._config.keys()) == {'a', 'b'}
