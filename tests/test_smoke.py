"""Smoke tests — confirm the package imports and basic plumbing works."""

from automotive_regulations_rag import __version__, hello


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"


def test_hello_returns_alive_message() -> None:
    assert hello() == "automotive-regulations-rag scaffold is alive"
