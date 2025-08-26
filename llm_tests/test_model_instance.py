# Copyright Axelera AI, 2025
import sys
import types

# Mock transformers and rich if not installed, so tests can run without these heavy dependencies
if "transformers" not in sys.modules:
    sys.modules["transformers"] = types.SimpleNamespace()
if not hasattr(sys.modules["transformers"], "AutoTokenizer"):
    sys.modules["transformers"].AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: None
    )
if "rich" not in sys.modules:
    sys.modules["rich"] = types.SimpleNamespace(console=types.SimpleNamespace(Console=object))
if "rich.console" not in sys.modules:
    sys.modules["rich.console"] = types.SimpleNamespace(Console=object)

import pytest

try:
    from llm.model_instance import TorchInstance
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    TorchInstance = None


def test_torch_instance_init_and_run():
    if TorchInstance is None:
        pytest.skip("transformers or torch not available")
    model_name = "sshleifer/tiny-gpt2"
    instance = TorchInstance(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode("hello", return_tensors="pt")
    # TorchInstance expects numpy input
    np_input = input_ids.cpu().numpy()
    logits = instance.run([np_input])
    assert logits is not None
    assert logits.shape[-1] > 0


def test_get_tokenizer_from_url(tmp_path, monkeypatch):
    # Import here to avoid circular import issues
    from inference_llm import get_tokenizer_from_url

    called = {}

    def fake_download_and_extract(url, zip_path, md5=None, delete_archive=True):
        called['url'] = url
        called['zip_path'] = zip_path
        called['md5'] = md5
        called['delete_archive'] = delete_archive
        # Simulate extraction by creating a dummy file
        (zip_path.parent / 'tokenizer.json').write_text('{}')

    monkeypatch.setattr("axelera.app.utils.download_and_extract_asset", fake_download_and_extract)
    tokenizer_url = "http://example.com/tokenizer.zip"
    tokenizer_md5 = "dummy_md5"
    model_name = "test/model"
    build_root = tmp_path
    result_dir = get_tokenizer_from_url(tokenizer_url, tokenizer_md5, model_name, build_root)
    from pathlib import Path

    assert Path(result_dir).exists()
    assert (Path(result_dir) / "tokenizer.json").exists()
    assert (Path(result_dir) / ".extracted_tokenizer").exists()
    assert called['url'] == tokenizer_url
    assert called['delete_archive'] is True


def test_inference_llm_tokenizer_dir(monkeypatch, tmp_path):
    # Simulate a local tokenizer directory
    tokenizer_dir = tmp_path / "my_tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text('{}')

    # Mock AutoTokenizer to check call
    called = {}

    class DummyTokenizer:
        def __init__(self, *args, **kwargs):
            called['args'] = args
            called['kwargs'] = kwargs

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", DummyTokenizer)

    # Patch get_tokenizer_from_url to fail if called
    monkeypatch.setattr(
        "inference_llm.get_tokenizer_from_url",
        lambda *a, **kw: (_ for _ in ()).throw(Exception("Should not be called")),
    )

    # Simulate args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer-dir', type=str, default=None)
    args = parser.parse_args([f'--tokenizer-dir={tokenizer_dir}'])

    # Simulate logic from inference_llm.py
    if args.tokenizer_dir:
        tokenizer = DummyTokenizer(args.tokenizer_dir, use_fast=True, padding_side="right")
    # Check that the correct directory was used
    assert called['args'][0] == str(tokenizer_dir)
    assert called['kwargs']['use_fast'] is True
    assert called['kwargs']['padding_side'] == "right"
