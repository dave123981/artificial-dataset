"""Tests for ```_components.py``` file."""

import pytest
import torch

from artificial_dataset._components import (
    compose,
    compose_weighted,
    linear,
    logarithmic,
    periodic_seasonal,
    polynomial,
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


@pytest.mark.parametrize("waveform", ["sine", "square", "triangle", "sawtooth"])
def test_periodic_seasonal_waveforms_stay_in_amplitude_bounds(waveform: str) -> None:
    """Every supported waveform stays within [-amplitude, amplitude] + offset."""
    x = torch.linspace(0, 40, steps=200)
    y = periodic_seasonal(x, period=10.0, amplitude=2.0, offset=0.0, waveform=waveform)
    assert float(y.max()) <= 2.0 + 1e-5
    assert float(y.min()) >= -2.0 - 1e-5


def test_periodic_seasonal_square_is_bipolar() -> None:
    """The square waveform only takes values in {-amplitude, 0, amplitude}."""
    x = torch.linspace(0, 20, steps=41)
    y = periodic_seasonal(x, period=10.0, amplitude=1.0, waveform="square")
    assert set(torch.unique(y).tolist()).issubset({-1.0, 0.0, 1.0})


def test_linear_slope_and_intercept() -> None:
    """linear() applies slope and intercept correctly."""
    x = torch.tensor([0.0, 1.0, 2.0])
    y = linear(x, slope=2.0, intercept=1.0)
    assert torch.equal(y, torch.tensor([1.0, 3.0, 5.0]))


def test_polynomial_quadratic() -> None:
    """coefficients=[c0, c1, c2] evaluates c0 + c1*x + c2*x^2."""
    x = torch.tensor([0.0, 1.0, 2.0])
    y = polynomial(x, coefficients=[1.0, 2.0, 3.0])
    assert torch.equal(y, torch.tensor([1.0, 6.0, 17.0]))


def test_compose_unknown_keys_are_ignored() -> None:
    """compose() silently ignores keys that are not recognised components."""
    x = torch.arange(5, dtype=torch.float32)
    y = compose(x, {"linear": {"slope": 1.0, "intercept": 0.0}, "not_a_component": {}})
    assert torch.equal(y, x)


def test_compose_weighted_default_weight_is_one() -> None:
    """Omitting the weight key defaults that component's weight to 1.0."""
    x = torch.arange(5, dtype=torch.float32)
    y = compose_weighted(x, [{"constant": {"value": 3.0}}])
    assert torch.equal(y, torch.full((5,), 3.0))
