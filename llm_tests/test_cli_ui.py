# Copyright Axelera AI, 2025
import pytest

pytest.importorskip("rich.live")
from llm.cli_ui import display_plain, display_rich_factory


class DummyConsole:
    def print(self, *args, **kwargs):
        # Just print to stdout for test visibility
        print(*args, **kwargs)

    @property
    def size(self):
        class Size:
            height = 24

        return Size()


def test_display_rich_factory_smoke():
    console = DummyConsole()
    display_fn = display_rich_factory(console)
    # Test user input prompt
    # Simulate is_user=True (should return user input)
    # We'll patch input to return 'hello'
    import builtins

    orig_input = builtins.input
    builtins.input = lambda *a, **k: 'hello'
    try:
        result = display_fn(None, None, True, [])
        assert result == 'hello'
    finally:
        builtins.input = orig_input
    # Test goodbye
    assert display_fn('Goodbye!', None, False, []) == ''


def test_display_plain_smoke():
    # Test user input prompt
    import builtins

    orig_input = builtins.input
    builtins.input = lambda *a, **k: 'hello'
    try:
        result = display_plain(None, None, True)
        assert result == 'hello'
    finally:
        builtins.input = orig_input
    # Test goodbye
    assert display_plain('Goodbye!', None, False) == ''
