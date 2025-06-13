# Copyright Axelera AI, 2025
from llm.conversation import ChatEncoder
import numpy as np
import pytest


class DummyTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        self.calls.append((text, add_special_tokens, return_tensors))
        # Return a simple encoding: each char as int
        if return_tensors == "np":
            return np.array([[ord(c) for c in text]])
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Just join all message contents
        return " ".join(m["content"] for m in messages)

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs back to a string"""
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        return "".join(chr(tid) for tid in token_ids)


class DummyEmbeddingProcessor:
    def process_batch(self, input_ids):
        return [np.ones((1, input_ids.shape[1], 4))]  # dummy embedding


def test_chat_encoder_basic():
    tokenizer = DummyTokenizer()
    embedding_processor = DummyEmbeddingProcessor()
    encoder = ChatEncoder(tokenizer, max_tokens=16, embedding_processor=embedding_processor)
    input_ids, embeddings = encoder.encode("hello", [])
    assert input_ids.shape[0] == 1
    assert embeddings.shape[0] == 1
    assert encoder.last_messages[0]["role"] == "system"
    assert encoder.last_messages[-1]["role"] == "user"
    encoder.add_to_history("hi", "there")
    assert encoder.cached_history == [("hi", "there")]
    encoder.reset()
    assert encoder.cached_history == []


def test_chat_encoder_history():
    tokenizer = DummyTokenizer()
    encoder = ChatEncoder(tokenizer, max_tokens=16)
    encoder.add_to_history("a", "b")
    encoder.add_to_history("c", "d")
    assert len(encoder.cached_history) == 2
    encoder.remove_from_history(1)
    assert len(encoder.cached_history) == 1
    encoder.remove_from_history(1)
    assert len(encoder.cached_history) == 0


def test_chat_encoder_min_response_space():
    tokenizer = DummyTokenizer()
    # Explicit override
    encoder = ChatEncoder(tokenizer, max_tokens=16, min_response_space=2)
    assert encoder.min_response_space == 2
    # Default logic: min_response_space = min(128, max(32, max_tokens // 8))
    encoder_16 = ChatEncoder(tokenizer, max_tokens=16)
    assert encoder_16.min_response_space == 32  # min is 32
    encoder_128 = ChatEncoder(tokenizer, max_tokens=128)
    assert encoder_128.min_response_space == 32  # 128//8=16, but min is 32
    encoder_1024 = ChatEncoder(tokenizer, max_tokens=1024)
    assert encoder_1024.min_response_space == 128  # 1024//8=128
    encoder_2048 = ChatEncoder(tokenizer, max_tokens=2048)
    assert encoder_2048.min_response_space == 128  # capped at 128


def test_chat_encoder_reset_with_preserve_system_prompt():
    """Test that reset with preserve_system_prompt=True maintains the system prompt."""
    tokenizer = DummyTokenizer()
    system_prompt = "You are a helpful assistant."
    encoder = ChatEncoder(tokenizer, max_tokens=16, system_prompt=system_prompt)

    # Add some history and check it's recorded
    encoder.add_to_history("user question", "assistant response")
    assert len(encoder.cached_history) == 1

    # Reset with preserve_system_prompt=True
    encoder.reset(preserve_system_prompt=True)

    # Verify history is cleared
    assert encoder.cached_history == []

    # Verify system prompt is preserved
    assert encoder.system_prompt == system_prompt

    # Verify internal state is as expected
    assert encoder.current_cutoff is None
    assert encoder.cached_history_token_counts == []
    assert encoder.cached_history_token_total == 0


def test_chat_encoder_update_system_prompt():
    """Test that update_system_prompt properly updates the system prompt."""
    tokenizer = DummyTokenizer()
    initial_prompt = "Initial prompt"
    encoder = ChatEncoder(tokenizer, max_tokens=16, system_prompt=initial_prompt)

    # Add some history
    encoder.add_to_history("user question", "assistant response")
    assert len(encoder.cached_history) == 1

    # Update system prompt
    new_prompt = "New system prompt"
    encoder.update_system_prompt(new_prompt)

    # Verify system prompt is updated
    assert encoder.system_prompt == new_prompt

    # Verify history is cleared by default
    assert encoder.cached_history == []

    # Verify template is updated
    assert new_prompt in encoder.system_template


def test_chat_encoder_stateless_mode():
    """Test that we can use the ChatEncoder in a stateless mode (like --no-history)."""
    tokenizer = DummyTokenizer()
    system_prompt = "You are a helpful assistant."
    encoder = ChatEncoder(tokenizer, max_tokens=16, system_prompt=system_prompt)

    # First conversation turn
    input_ids_1, _ = encoder.encode("first question", [])
    encoder.add_to_history("first question", "first response")

    # Clear history to simulate --no-history flag
    encoder.reset(preserve_system_prompt=True)

    # Second conversation turn - should be processed independently
    input_ids_2, _ = encoder.encode("second question", [])

    # Both inputs should include the system prompt and only the current question
    # But NOT the history from the previous turn
    assert input_ids_1.shape[1] != input_ids_2.shape[1]  # Different input lengths

    # Verify history remains empty after second turn
    assert len(encoder.cached_history) == 0

    # Third turn with history to compare
    encoder.add_to_history("second question", "second response")
    input_ids_3, _ = encoder.encode("third question", [("second question", "second response")])

    # Input with history should be different than without
    assert input_ids_2.shape[1] != input_ids_3.shape[1]


def test_chat_encoder_system_prompt_in_encode_output():
    """Test that the system prompt is properly included in encode output."""
    tokenizer = DummyTokenizer()
    # Make the initial and new prompts different lengths to ensure different encodings
    initial_prompt = "You are an AI assistant for Axelera."
    encoder = ChatEncoder(tokenizer, max_tokens=16, system_prompt=initial_prompt)

    # First encode with initial prompt
    input_ids_1, _ = encoder.encode("Tell me about yourself", [])

    # Update to a completely different system prompt with a different length
    new_prompt = "You are a helpful cooking assistant that specializes in recipes."
    encoder.update_system_prompt(new_prompt)

    # Encode with new system prompt
    input_ids_2, _ = encoder.encode("Tell me about yourself", [])

    # Check that the encoded outputs are different sizes due to different system prompts
    assert len(tokenizer.decode(input_ids_1[0])) != len(tokenizer.decode(input_ids_2[0]))

    # Verify last_messages properly contains the new system prompt
    assert encoder.last_messages[0]["role"] == "system"
    assert encoder.last_messages[0]["content"] == new_prompt

    # Check that system prompt is in the encoded output by checking decoded string
    encoded_text = tokenizer.decode(input_ids_2[0])
    assert new_prompt in encoded_text
    assert initial_prompt not in encoded_text
