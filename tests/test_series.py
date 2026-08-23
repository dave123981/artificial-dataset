"""Unit tests for the 1D dataset generation and SyneticSeries dataclass."""

from typing import Any

import pytest
import torch

from artificial_dataset.series import (
    SyntheticSeries,
    make_composite_series,
    make_series,
)


def test_synthetic_series_dataclass_initialization() -> None:
    """Verify SyntheticSeries dataclass instantiation and default field structures."""
    series_length = 50
    x = torch.arange(series_length, dtype=torch.float32)
    y = torch.zeros(series_length, dtype=torch.float32)
    is_anomaly = torch.zeros(series_length, dtype=torch.bool)
    anomaly_type = [""] * series_length
    anomalies: list[dict[str, Any]] = []

    series = SyntheticSeries(
        x=x,
        y=y,
        is_anomaly=is_anomaly,
        anomaly_type=anomaly_type,
        anomalies=anomalies,
    )

    assert series.x.shape == (series_length,)
    assert series.y.shape == (series_length,)
    assert series.is_anomaly.dtype == torch.bool
    assert len(series.anomaly_type) == series_length
    assert series.anomalies == []


def test_synthetic_series_pipe_applies_function() -> None:
    """pipe() calls func(self, *args, **kwargs) and returns its result."""
    series = make_series(series_length=10, function_type="constant")
    result = series.pipe(lambda s, factor: len(s) * factor, factor=3)
    assert result == 30


def test_generate_timeline_nonpositive_length_raises() -> None:
    """make_series raises ValueError when series_length is not positive."""
    with pytest.raises(ValueError, match="positive integer"):
        make_series(series_length=0, function_type="constant")


@pytest.mark.parametrize(
    "function_type, params",
    [
        ("constant", {"value": 3.5}),
        ("linear", {"slope": 0.5, "intercept": 1.0}),
        ("sinusoidal", {"amplitude": 2.0, "frequency": 0.05}),
        ("exponential", {"initial_value": 1.0, "growth_rate": 0.02}),
        ("logarithmic", {"scale": 2.0, "shift": 1.0}),
        (
            "periodic_seasonal",
            {"period": 12.0, "amplitude": 1.5, "waveform": "triangle"},
        ),
    ],
)
def test_make_series_single_functions(
    function_type: str, params: dict[str, Any]
) -> None:
    """Verify make_series output structure for base signals.

    Validates that generated output matches expected shapes, types, and anomaly tags.
    """
    length = 100
    series = make_series(
        series_length=length,
        function_type=function_type,
        function_params=params,
        noise_std=0.0,
    )

    assert isinstance(series, SyntheticSeries)
    assert series.x.shape == (length,)
    assert series.y.shape == (length,)
    assert not series.is_anomaly.any()
    assert all(t == "" for t in series.anomaly_type)
    assert len(series.anomalies) == 0


def test_make_series_noise_reproducibility() -> None:
    """Verify that random_state produces identical noisy outputs."""
    length = 100
    series_a = make_series(
        series_length=length,
        function_type="sinusoidal",
        noise_std=0.2,
        random_state=42,
    )
    series_b = make_series(
        series_length=length,
        function_type="sinusoidal",
        noise_std=0.2,
        random_state=42,
    )

    assert torch.equal(series_a.y, series_b.y)


def test_make_series_composite() -> None:
    """Regression test documenting a known components-schema mismatch.

    Passing {"function_type", "function_params"}-style entries (the schema
    used by make_series) into make_composite_series's components list is
    silently accepted: neither key is recognised by compose(), so every
    term contributes zero and the resulting series is flat. This test pins
    down the current (buggy) behaviour; if the schemas are unified, this
    test should be updated or removed alongside that fix.
    """
    wrong_schema_components = [
        {"function_type": "linear", "function_params": {"slope": 0.1}, "weight": 1.0},
    ]
    series = make_composite_series(
        series_length=20, components=wrong_schema_components, noise_std=0.0
    )
    assert torch.equal(series.y, torch.zeros(20))


def test_make_series_linear_trend_key_mismatch_produces_zero_signal() -> None:
    """Regression test documenting a known key-name mismatch.

    compose() only recognises the key "linear", but the docstring of
    make_series advertises "linear_trend". Passing "linear_trend" is
    silently accepted and produces an all-zero signal instead of raising
    an error or computing a trend.
    """
    series = make_series(
        series_length=20,
        function_type="linear_trend",
        function_params={"slope": 0.5, "intercept": 1.0},
        noise_std=0.0,
    )
    assert torch.equal(series.y, torch.zeros(20))


def test_make_series_noise_without_random_state() -> None:
    """noise_std > 0 without a random_state still perturbs the clean signal."""
    length = 100
    clean = make_series(series_length=length, function_type="constant", noise_std=0.0)
    noisy = make_series(series_length=length, function_type="constant", noise_std=0.5)
    assert noisy.y.shape == (length,)
    assert not torch.equal(noisy.y, clean.y)


def test_make_composite_series_noise_reproducibility() -> None:
    """random_state makes noisy composite series reproducible."""
    components = [{"constant": {"value": 1.0}}]
    series_a = make_composite_series(
        series_length=50, components=components, noise_std=0.3, random_state=7
    )
    series_b = make_composite_series(
        series_length=50, components=components, noise_std=0.3, random_state=7
    )
    assert torch.equal(series_a.y, series_b.y)
    assert not torch.equal(series_a.y, torch.ones(50))


def test_make_composite_series_noise_without_random_state() -> None:
    """noise_std > 0 without a random_state still perturbs the clean signal."""
    components = [{"constant": {"value": 1.0}}]
    clean = make_composite_series(
        series_length=50, components=components, noise_std=0.0
    )
    noisy = make_composite_series(
        series_length=50, components=components, noise_std=0.5
    )
    assert not torch.equal(noisy.y, clean.y)
