# Copyright Axelera AI, 2025
import os
import tempfile

from llm.embedding_processor import EmbeddingProcessor
import numpy as np
import pytest


def make_dummy_embedding_file(tmp_path, vocab_size=10, embedding_dim=4):
    arr = np.random.rand(vocab_size, embedding_dim).astype(np.float16)
    file_path = tmp_path / "embeddings.npz"
    np.savez(file_path, embeddings=arr, vocab_size=vocab_size, embedding_dim=embedding_dim)
    return str(file_path)


def test_embedding_processor_init_and_shape(tmp_path):
    file_path = make_dummy_embedding_file(tmp_path)
    processor = EmbeddingProcessor(file_path)
    shape = processor.get_embedding_shape()
    assert shape == (10, 4)


def test_embedding_processor_process_batch(tmp_path):
    file_path = make_dummy_embedding_file(tmp_path)
    processor = EmbeddingProcessor(file_path)
    input_ids = np.array([[1, 2, 3]])
    embeddings, mask = processor.process_batch(input_ids)
    assert embeddings.shape == (1, 3, 4)
    assert mask.shape == (1, 3)
    # Check that embeddings match the file
    arr = np.load(file_path)["embeddings"]
    np.testing.assert_allclose(embeddings[0, 0], arr[1])
    np.testing.assert_allclose(embeddings[0, 1], arr[2])
    np.testing.assert_allclose(embeddings[0, 2], arr[3])
