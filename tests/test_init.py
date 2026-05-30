"""Tests for artificial_dataset package initialisation."""

import importlib
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import artificial_dataset


def test_version_is_string() -> None:
    """__version__ is a non-empty string."""
    assert isinstance(artificial_dataset.__version__, str)
    assert artificial_dataset.__version__ != ""


def test_version_fallback() -> None:
    """__version__ falls back to 'unknown' when package metadata is unavailable."""
    try:
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            importlib.reload(artificial_dataset)
            assert artificial_dataset.__version__ == "unknown"
    finally:
        importlib.reload(artificial_dataset)


def test_public_api() -> None:
    """All symbols listed in __all__ are importable from the package."""
    for name in artificial_dataset.__all__:
        assert hasattr(artificial_dataset, name), f"Missing public symbol: {name}"
