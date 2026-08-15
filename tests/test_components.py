"""Tests for ```_components.py``` file."""

import pytest
import torch

from artificial_dataset._components import (
    compose_weighted,
    logarithmic,
    periodic_seasonal,
)


def test_logarithmic_invalid_shift_raises() -> None:
    """A non-positive shift raises ValueError to keep log(x + shift) defined."""
    x = torch.arange(5, dtype=torch.float32)
    with pytest.raises(ValueError, match="shift must be > 0"):
        logarithmic(x, shift=0.0)


def test_periodic_seasonal_invalid_period_raises() -> None:
    """A non-positive period raises ValueError."""
    x = torch.arange(10, dtype=torch.float32)
    with pytest.raises(ValueError, match="period must be > 0"):
        periodic_seasonal(x, period=0.0)


def test_periodic_seasonal_unknown_waveform_raises() -> None:
    """An unrecognised waveform name raises ValueError."""
    x = torch.arange(10, dtype=torch.float32)
    with pytest.raises(ValueError, match="Unknown waveform"):
        periodic_seasonal(x, waveform="hexagon")


def test_compose_weighted_empty_components_raises() -> None:
    """An empty components list raises ValueError."""
    x = torch.arange(5, dtype=torch.float32)
    with pytest.raises(ValueError, match="at least one entry"):
        compose_weighted(x, [])
