# Copyright Axelera AI, 2025
"""
Test the run_chat_loop function's history keeping functionality
"""

from unittest.mock import MagicMock, patch

import pytest


def test_run_chat_loop_clear_history_when_keep_history_is_false():
    """Test that run_chat_loop respects the keep_history=False flag."""
    from inference_llm import run_chat_loop

    # Mock dependencies
    mock_display_fn = MagicMock()
    mock_chat_encoder = MagicMock()
    # Mock encode to return two values as expected
    mock_chat_encoder.encode.return_value = (MagicMock(), MagicMock())
    mock_model_instance = MagicMock()
    mock_tokenizer = MagicMock()
    mock_log = MagicMock()

    # Set up a counter to control flow - user input once, then EOFError
    call_count = [0]

    def input_side_effect(*args, **kwargs):
        if args and args[0] is None:
            call_count[0] += 1
            if call_count[0] == 1:
                return "test message"
            raise EOFError()
        return "Response text"

    mock_display_fn.side_effect = input_side_effect

    # Mock stream_response
    with patch("inference_llm.stream_response") as mock_stream:
        # Create a proper iterator that yields a tuple
        mock_stream.return_value = iter(
            [("Response text", {"ttft": 0.5, "tokens_per_sec": 10, "tokens": 5})]
        )

        # Set history and other params
        history = []

        # Run the function with no_history=True
        run_chat_loop(
            mock_display_fn,
            show_stats=False,
            history=history,
            chat_encoder=mock_chat_encoder,
            model_instance=mock_model_instance,
            tokenizer=mock_tokenizer,
            max_tokens=1024,
            temperature=0.7,
            eos_token_id=None,
            end_token_id=None,
            LOG=mock_log,
            keep_history=False,
        )

        # Assert reset was called with preserve_system_prompt=True
        mock_chat_encoder.reset.assert_called_once_with(preserve_system_prompt=True)

        # Assert that history contains the message before being cleared
        # (history gets appended before keep_history=False clears it)
        assert mock_chat_encoder.encode.called
        assert mock_chat_encoder.encode.call_args[0][0] == "test message"


def test_run_chat_loop_with_history_when_keep_history_is_true():
    """Test that run_chat_loop keeps history when keep_history=True."""
    from inference_llm import run_chat_loop

    # Mock dependencies
    mock_display_fn = MagicMock()
    mock_chat_encoder = MagicMock()
    # Mock encode to return two values as expected
    mock_chat_encoder.encode.return_value = (MagicMock(), MagicMock())
    mock_model_instance = MagicMock()
    mock_tokenizer = MagicMock()
    mock_log = MagicMock()

    # Set up a counter to control flow - user input once, then EOFError
    call_count = [0]

    def input_side_effect(*args, **kwargs):
        if args and args[0] is None:
            call_count[0] += 1
            if call_count[0] == 1:
                return "test message"
            raise EOFError()
        return "Response text"

    mock_display_fn.side_effect = input_side_effect

    # Mock stream_response
    with patch("inference_llm.stream_response") as mock_stream:
        mock_stream.return_value = iter(
            [("Response text", {"ttft": 0.5, "tokens_per_sec": 10, "tokens": 5})]
        )

        # Set history and other params
        history = []

        # Run the function with no_history=False
        run_chat_loop(
            mock_display_fn,
            show_stats=False,
            history=history,
            chat_encoder=mock_chat_encoder,
            model_instance=mock_model_instance,
            tokenizer=mock_tokenizer,
            max_tokens=1024,
            temperature=0.7,
            eos_token_id=None,
            end_token_id=None,
            LOG=mock_log,
            keep_history=True,
        )

        # Assert reset was NOT called
        mock_chat_encoder.reset.assert_not_called()

        # Assert that history list contains the test message and response
        assert len(history) == 1
        assert history[0] == ("test message", "Response text")


def test_run_chat_loop_clear_history_command():
    """Test that run_chat_loop handles the !CLEAR_HISTORY! command."""
    from inference_llm import run_chat_loop

    # Mock dependencies
    mock_display_fn = MagicMock()
    mock_chat_encoder = MagicMock()
    mock_model_instance = MagicMock()
    mock_tokenizer = MagicMock()
    mock_log = MagicMock()

    # Set up a counter to control flow
    call_count = [0]

    def input_side_effect(*args, **kwargs):
        if args and args[0] is None:
            call_count[0] += 1
            if call_count[0] == 1:
                return "!CLEAR_HISTORY!"
            raise EOFError()
        return "Response text"

    mock_display_fn.side_effect = input_side_effect

    # Set history and other params
    history = [("previous message", "previous response")]

    # Run the function
    run_chat_loop(
        mock_display_fn,
        show_stats=False,
        history=history,
        chat_encoder=mock_chat_encoder,
        model_instance=mock_model_instance,
        tokenizer=mock_tokenizer,
        max_tokens=1024,
        temperature=0.7,
        eos_token_id=None,
        end_token_id=None,
        LOG=mock_log,
        keep_history=True,  # This should be ignored for clear history command
    )

    # Assert reset was called
    mock_chat_encoder.reset.assert_called_once()

    # Assert that history list was cleared
    assert history == []
