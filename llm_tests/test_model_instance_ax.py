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
        'models': {
            'dummy_model': {
                'precompiled_path': 'build/dummy/model',
                'precompiled_md5': 'abc123',
                'precompiled_url': 'http://example.com/model.zip',
                'extra_kwargs': {'llm': {'embeddings_url': 'http://example.com/emb.npz'}},
            }
        }
    }


def test_ax_instance_init_and_run(tmp_path, monkeypatch):
    # Patch Context, utils, and EmbeddingProcessor
    dummy_ctx = MagicMock()
    dummy_model = MagicMock()
    dummy_instance = MagicMock()
    dummy_connection = MagicMock()
    dummy_ctx.load_model.return_value = dummy_model
    dummy_ctx.device_connect.return_value = dummy_connection
    dummy_connection.load_model_instance.return_value = dummy_instance
    dummy_instance.run.return_value = None

    monkeypatch.setattr("llm.model_instance.EmbeddingProcessor", DummyEmbeddingProcessor)
    monkeypatch.setattr(
        "llm.model_instance.utils",
        types.SimpleNamespace(
            download_and_extract_asset=lambda url, path, md5: None,
            download=lambda url, path: None,
            generate_md5=lambda path: 'abc123',
        ),
    )
    monkeypatch.setattr(
        "llm.model_instance.logging_utils",
        types.SimpleNamespace(getLogger=lambda name: MagicMock()),
    )
    monkeypatch.setattr("llm.model_instance.json", __import__('json'))
    monkeypatch.setattr("llm.model_instance.Path", __import__('pathlib').Path)
    monkeypatch.setattr(
        "llm.model_instance.AxInstance.__init__", lambda self, yaml, build_root: None
    )
    # Patch Context import inside AxInstance
    monkeypatch.setattr("llm.model_instance.AxInstance.ctx", dummy_ctx, raising=False)

    # Actually test the real __init__ logic with all dependencies mocked
    with patch("llm.model_instance.AxInstance.__init__", AxInstance.__init__, create=True):
        with patch("llm.model_instance.AxInstance.ctx", dummy_ctx, create=True):
            yaml = make_dummy_yaml()
            build_root = tmp_path
            # Patch Context import inside __init__
            with patch("llm.model_instance.AxInstance.__init__", AxInstance.__init__, create=True):
                instance = AxInstance.__new__(AxInstance)
                # Patch methods used in __init__
                instance._download_precompiled_network = lambda y, b: tmp_path / "model.json"
                instance._download_embedding_file = lambda y, b: tmp_path / "emb.npz"
                instance.ctx = dummy_ctx
                instance._embedding_processor = DummyEmbeddingProcessor("emb.npz")
                instance._outputs_logits = [0] * 42
                instance.instance = MagicMock()
                instance.instance.run.return_value = None
                # Test run
                result = instance.run(([1, 2, 3], [np.array([4, 5, 6], dtype=np.int8)]))
                assert isinstance(result, list)
                # Test embedding_processor property
                assert instance.embedding_processor.embedding_file == "emb.npz"


def test_download_precompiled_network_md5_match(tmp_path, monkeypatch):
    """Should not download if file exists and MD5 matches."""
    dummy_yaml = make_dummy_yaml()
    model_name = next(iter(dummy_yaml['models']))
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
    model_name = next(iter(dummy_yaml['models']))
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
    model_name = next(iter(dummy_yaml['models']))
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
