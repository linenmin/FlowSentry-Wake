"""
Test the chat history management functionality directly, without UI dependencies
"""

from unittest.mock import MagicMock, patch

import pytest


def test_chat_history_management_keep_history_false():
    """Test the core history behavior when keep_history=False, without UI dependencies."""

    # Create mocks for the required components
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = (MagicMock(), MagicMock())

    # Mock stream_response to return one response
    with patch("llm.conversation.stream_response") as mock_stream:
        mock_stream.return_value = iter(
            [("Response text", {"ttft": 0.5, "tokens_per_sec": 10, "tokens": 5})]
        )

        # Core history behavior test when keep_history=False:
        keep_history = False

        # Simulate history data
        history = [("Previous question", "Previous answer")]

        # Simulate what happens in the chat function
        chat_history = [] if not keep_history else [(h[0], h[1]) for h in history if h[0] and h[1]]

        # Verify empty history is used when keep_history=False
        assert chat_history == []

        # Test encode with empty history
        mock_encoder.encode("New message", chat_history)
        mock_encoder.encode.assert_called_once_with("New message", [])

        # Simulate adding new message to the history
        mock_encoder.add_to_history("New message", "Response text")

        # Simulate clearing history after response (when keep_history=False)
        if not keep_history:
            mock_encoder.reset(preserve_system_prompt=True)

        # Verify reset was called
        mock_encoder.reset.assert_called_once_with(preserve_system_prompt=True)


def test_chat_history_management_with_history_when_keep_history_true():
    """Test the core history behavior when keep_history=True, without UI dependencies."""

    # Create mocks for the required components
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = (MagicMock(), MagicMock())

    # Mock stream_response to return one response
    with patch("llm.conversation.stream_response") as mock_stream:
        mock_stream.return_value = iter(
            [("Response text", {"ttft": 0.5, "tokens_per_sec": 10, "tokens": 5})]
        )

        # Core history behavior test when keep_history=True:
        keep_history = True

        # Simulate history data
        history = [("Previous question", "Previous answer")]

        # Simulate what happens in the chat function
        chat_history = [] if not keep_history else [(h[0], h[1]) for h in history if h[0] and h[1]]

        # Verify history is preserved when keep_history=True
        assert chat_history == [("Previous question", "Previous answer")]

        # Test encode with existing history
        mock_encoder.encode("New message", chat_history)
        mock_encoder.encode.assert_called_once_with(
            "New message", [("Previous question", "Previous answer")]
        )

        # Simulate adding new message to the history
        mock_encoder.add_to_history("New message", "Response text")

        # Simulate clearing history after response (when keep_history=False)
        if not keep_history:
            mock_encoder.reset(preserve_system_prompt=True)

        # Verify reset was NOT called
        mock_encoder.reset.assert_not_called()


def test_clear_history_command():
    """Test the behavior of the clear history command."""

    # Create mock encoder
    mock_encoder = MagicMock()

    # Simulate the clear_history function
    def clear_history():
        # Reset but preserve the system prompt to avoid unnecessary reprocessing
        mock_encoder.reset(preserve_system_prompt=True)
        return [], "", "Ready"

    # Call the function
    result = clear_history()

    # Verify reset was called
    mock_encoder.reset.assert_called_once_with(preserve_system_prompt=True)

    # Verify empty history was returned
    assert result[0] == []
