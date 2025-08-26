# Copyright Axelera AI, 2025
import json
import types
from unittest.mock import MagicMock, patch

from llm.model_instance import AxInstance
import numpy as np
import pytest


class DummyEmbeddingProcessor:
    def __init__(self, embedding_file):
        self.embedding_file = embedding_file

    def get_embedding_shape(self):
        return (42, 8)

    def process_batch(self, input_ids):
        return input_ids


def make_dummy_yaml():
    return {
        "name": "test_model",
        "models": {
            "test": {
                "precompiled_path": "build/test/model",
                "precompiled_md5": "abc123",
                "precompiled_url": "https://example.com/model.zip",
                "extra_kwargs": {
                    "llm": {
                        "embeddings_url": "https://example.com/embedding.npz",
                        "embeddings_md5": "def456",
                    }
                },
            }
        },
    }


def test_ax_instance_init_and_run(tmp_path, monkeypatch):
    # Create mock device
    dummy_device = MagicMock()
    dummy_device.name = "metis-0:1:0"
    dummy_device.max_memory = 8 * (1024**3)  # 8GB
    dummy_device.subdevice_count = 4

    # Create mock Context and connection
    dummy_ctx = MagicMock()
    dummy_model = MagicMock()
    dummy_instance = MagicMock()
    dummy_connection = MagicMock()

    # Set up the Context mock to return devices and handle device connection
    dummy_ctx.list_devices.return_value = [dummy_device]
    dummy_ctx.load_model.return_value = dummy_model
    dummy_ctx.device_connect.return_value = dummy_connection
    dummy_connection.load_model_instance.return_value = dummy_instance
    dummy_instance.run.return_value = None

    # Patch the imports and dependencies
    monkeypatch.setattr("llm.model_instance.EmbeddingProcessor", DummyEmbeddingProcessor)

    # Mock the Context class to return our dummy context
    def mock_context():
        return dummy_ctx

    monkeypatch.setattr("axelera.runtime.Context", mock_context)

    monkeypatch.setattr(
        "llm.model_instance.utils",
        types.SimpleNamespace(
            download_and_extract_asset=lambda url, path, md5: None,
            download=lambda url, path, md5=None: None,
            generate_md5=lambda path: 'abc123',
            md5_validates=lambda path, md5: True,
        ),
    )
    monkeypatch.setattr(
        "llm.model_instance.logging_utils",
        types.SimpleNamespace(getLogger=lambda name: MagicMock()),
    )
    monkeypatch.setattr("llm.model_instance.json", __import__('json'))
    monkeypatch.setattr("llm.model_instance.Path", __import__('pathlib').Path)

    yaml = make_dummy_yaml()
    build_root = tmp_path

    # Create model and embedding files
    model_dir = tmp_path / "test" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.json").touch()

    embedding_path = tmp_path / "embedding.npz"
    embedding_path.touch()

    # Test instance creation and running
    instance = AxInstance(yaml, build_root, ddr_requirement_gb=4)

    # Verify Context was called correctly
    dummy_ctx.list_devices.assert_called_once()
    dummy_ctx.device_connect.assert_called_once_with(device=dummy_device, num_sub_devices=4)
    dummy_ctx.load_model.assert_called_once()
    dummy_connection.load_model_instance.assert_called_once_with(dummy_model)

    # Test running inference
    input_data = (np.array([1, 2, 3]), [np.array([[1, 2, 3]], dtype=np.int8)])
    result = instance.run(input_data)

    dummy_instance.run.assert_called_once()
    assert result is not None


def test_ax_instance_device_selection(tmp_path, monkeypatch):
    """Test device selection functionality."""
    # Create mock devices
    device1 = MagicMock()
    device1.name = "metis-0:1:0"
    device1.max_memory = 8 * (1024**3)
    device1.subdevice_count = 4

    device2 = MagicMock()
    device2.name = "metis-0:3:0"
    device2.max_memory = 8 * (1024**3)
    device2.subdevice_count = 4

    # Create mock Context
    dummy_ctx = MagicMock()
    dummy_model = MagicMock()
    dummy_instance = MagicMock()
    dummy_connection = MagicMock()

    dummy_ctx.list_devices.return_value = [device1, device2]
    dummy_ctx.load_model.return_value = dummy_model
    dummy_ctx.device_connect.return_value = dummy_connection
    dummy_connection.load_model_instance.return_value = dummy_instance

    # Patch dependencies
    monkeypatch.setattr("llm.model_instance.EmbeddingProcessor", DummyEmbeddingProcessor)
    monkeypatch.setattr("axelera.runtime.Context", lambda: dummy_ctx)
    monkeypatch.setattr(
        "llm.model_instance.utils",
        types.SimpleNamespace(
            download_and_extract_asset=lambda url, path, md5: None,
            download=lambda url, path, md5=None: None,
            generate_md5=lambda path: 'abc123',
            md5_validates=lambda path, md5: True,
        ),
    )
    monkeypatch.setattr(
        "llm.model_instance.logging_utils",
        types.SimpleNamespace(getLogger=lambda name: MagicMock()),
    )
    monkeypatch.setattr("llm.model_instance.json", __import__('json'))
    monkeypatch.setattr("llm.model_instance.Path", __import__('pathlib').Path)

    yaml = make_dummy_yaml()
    build_root = tmp_path

    # Create dummy files
    model_dir = tmp_path / "test" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.json").touch()

    embedding_path = tmp_path / "embedding.npz"
    embedding_path.touch()

    # Test device selection by index
    instance = AxInstance(yaml, build_root, ddr_requirement_gb=4, device_selector="1")

    # Should connect to the second device (index 1)
    dummy_ctx.device_connect.assert_called_with(device=device2, num_sub_devices=4)


def test_ax_instance_device_selection_env_var(tmp_path, monkeypatch):
    """Test device selection via environment variable."""
    # Create mock devices
    device1 = MagicMock()
    device1.name = "metis-0:1:0"
    device1.max_memory = 8 * (1024**3)
    device1.subdevice_count = 4

    device2 = MagicMock()
    device2.name = "metis-0:3:0"
    device2.max_memory = 8 * (1024**3)
    device2.subdevice_count = 4

    # Create mock Context
    dummy_ctx = MagicMock()
    dummy_model = MagicMock()
    dummy_instance = MagicMock()
    dummy_connection = MagicMock()

    dummy_ctx.list_devices.return_value = [device1, device2]
    dummy_ctx.load_model.return_value = dummy_model
    dummy_ctx.device_connect.return_value = dummy_connection
    dummy_connection.load_model_instance.return_value = dummy_instance

    # Set environment variable
    monkeypatch.setenv("AXELERA_DEVICE_SELECTOR", "0")

    # Patch dependencies
    monkeypatch.setattr("llm.model_instance.EmbeddingProcessor", DummyEmbeddingProcessor)
    monkeypatch.setattr("axelera.runtime.Context", lambda: dummy_ctx)
    monkeypatch.setattr(
        "llm.model_instance.utils",
        types.SimpleNamespace(
            download_and_extract_asset=lambda url, path, md5: None,
            download=lambda url, path, md5=None: None,
            generate_md5=lambda path: 'abc123',
            md5_validates=lambda path, md5: True,
        ),
    )
    monkeypatch.setattr(
        "llm.model_instance.logging_utils",
        types.SimpleNamespace(getLogger=lambda name: MagicMock()),
    )
    monkeypatch.setattr("llm.model_instance.json", __import__('json'))
    monkeypatch.setattr("llm.model_instance.Path", __import__('pathlib').Path)

    yaml = make_dummy_yaml()
    build_root = tmp_path

    # Create dummy files
    model_dir = tmp_path / "test" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.json").touch()

    embedding_path = tmp_path / "embedding.npz"
    embedding_path.touch()

    # Test device selection via environment variable
    instance = AxInstance(yaml, build_root, ddr_requirement_gb=4)

    # Should connect to the first device (index 0)
    dummy_ctx.device_connect.assert_called_with(device=device1, num_sub_devices=4)


def test_download_precompiled_network_md5_match(tmp_path, monkeypatch):
    """Should not download if file exists and MD5 matches."""
    dummy_yaml = make_dummy_yaml()
    model_name = "test"
    model_dir = tmp_path / model_name / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.json"
    model_path.write_text("dummy model content")
    log_path = tmp_path / "downloaded_models.json"
    log_data = {
        dummy_yaml['models'][model_name]['precompiled_md5']: {
            "model_path": str(model_path),
            "model_md5": "abc123",
        }
    }
    log_path.write_text(json.dumps(log_data))

    called = {"download": False}
    monkeypatch.setattr("llm.model_instance.utils.generate_md5", lambda path: "abc123")
    monkeypatch.setattr(
        "llm.model_instance.utils.download_and_extract_asset",
        lambda url, path, md5: called.update({"download": True}),
    )
    monkeypatch.setattr("llm.model_instance.json", __import__('json'))
    monkeypatch.setattr("llm.model_instance.Path", __import__('pathlib').Path)

    instance = AxInstance.__new__(AxInstance)
    result_path = instance._download_precompiled_network(dummy_yaml, tmp_path)
    assert result_path == model_path
    assert not called["download"]


def test_download_precompiled_network_file_missing(tmp_path, monkeypatch):
    """Should download if file does not exist."""
    dummy_yaml = make_dummy_yaml()
    model_name = "test"
    model_dir = tmp_path / model_name / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.json"
    log_path = tmp_path / "downloaded_models.json"
    log_data = {}
    log_path.write_text(json.dumps(log_data))

    called = {"download": False}
    monkeypatch.setattr("llm.model_instance.utils.generate_md5", lambda path: "abc123")
    monkeypatch.setattr(
        "llm.model_instance.utils.download_and_extract_asset",
        lambda url, path, md5: called.update({"download": True}),
    )
    monkeypatch.setattr("llm.model_instance.json", __import__('json'))
    monkeypatch.setattr("llm.model_instance.Path", __import__('pathlib').Path)

    instance = AxInstance.__new__(AxInstance)
    result_path = instance._download_precompiled_network(dummy_yaml, tmp_path)
    assert result_path == model_path
    assert called["download"]


def test_download_precompiled_network_md5_mismatch(tmp_path, monkeypatch):
    """Should re-download if file exists but MD5 does not match."""
    dummy_yaml = make_dummy_yaml()
    model_name = "test"
    model_dir = tmp_path / model_name / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.json"
    model_path.write_text("dummy model content")
    log_path = tmp_path / "downloaded_models.json"
    log_data = {
        dummy_yaml['models'][model_name]['precompiled_md5']: {
            "model_path": str(model_path),
            "model_md5": "not_the_same_md5",
        }
    }
    log_path.write_text(json.dumps(log_data))

    called = {"download": False}
    monkeypatch.setattr("llm.model_instance.utils.generate_md5", lambda path: "different_md5")
    monkeypatch.setattr(
        "llm.model_instance.utils.download_and_extract_asset",
        lambda url, path, md5: called.update({"download": True}),
    )
    monkeypatch.setattr("llm.model_instance.json", __import__('json'))
    monkeypatch.setattr("llm.model_instance.Path", __import__('pathlib').Path)

    instance = AxInstance.__new__(AxInstance)
    result_path = instance._download_precompiled_network(dummy_yaml, tmp_path)
    assert result_path == model_path
    assert called["download"]
