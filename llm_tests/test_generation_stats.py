# Copyright Axelera AI, 2025
from llm.generation_stats import GenerationStats
import pytest


class DummyTime:
    def __init__(self):
        self._now = 1000.0

    def time(self):
        self._now += 1.0
        return self._now


def test_generation_stats_full(monkeypatch):
    stats = GenerationStats()
    dummy_time = DummyTime()
    monkeypatch.setattr("time.time", dummy_time.time)

    stats.start_tokenization()
    stats.end_tokenization()
    stats.end_prefill()
    stats.start_generation()
    for _ in range(3):
        stats.record_token()
    stats.end_generation()
    s = stats.get_stats()
    assert s["tokenize_time"] == 1.0
    assert s["prefill_time"] == 1.0
    assert s["ttft"] == 1.0
    assert s["total_time"] == 4.0
    assert s["tokens_sec"] == 0.75
    assert s["num_tokens"] == 3
    summary = stats.summary(log=False)
    assert "Tokenization" in summary
    assert "Tokens: 3" in summary


def test_generation_stats_reset(monkeypatch):
    stats = GenerationStats()
    dummy_time = DummyTime()
    monkeypatch.setattr("time.time", dummy_time.time)
    stats.start_tokenization()
    stats.end_tokenization()
    stats.reset()
    s = stats.get_stats()
    assert s["tokenize_time"] is None
    assert s["num_tokens"] == 0
