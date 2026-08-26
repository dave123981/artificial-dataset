"""Unit tests for the 1D dataset generation and SyneticSeries dataclass."""

from typing import Any

import pytest
import torch

from artificial_dataset.injectors import add_level_shift, add_point_anomalies
from artificial_dataset.series import (
    SyntheticSeries,
    SyntheticSeriesSplits,
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


def test_split_along_time_axis() -> None:
    """A split cuts the timeline into contiguous, recombinable segments."""
    series = make_series(
        series_length=256,
        function_type="sinusoidal",
        function_params={"amplitude": 1.0, "frequency": 0.05},
        noise_std=0.0,
    )
    splits = series.split((0.5, 0.25, 0.25))

    assert isinstance(splits, SyntheticSeriesSplits)
    assert len(splits.train) == 128
    assert len(splits.val) == 64
    assert len(splits.test) == 64

    recombined = torch.cat([splits.train.y, splits.val.y, splits.test.y])
    assert torch.equal(recombined, series.y)


def test_split_rebases_point_anomaly_indices() -> None:
    """Point-style 'indices' entries are filtered and re-based per segment."""
    series = make_series(series_length=100, function_type="constant")
    series = add_point_anomalies(series, n_anomalies=3, random_state=5)

    splits = series.split((0.5, 0.3, 0.2))
    for subset in (splits.train, splits.val, splits.test):
        for entry in subset.anomalies:
            if entry["type"] != "point":
                continue
            assert all(0 <= i < len(subset) for i in entry["indices"])
            assert torch.all(subset.is_anomaly[entry["indices"]])

    total_indices = sum(
        len(entry["indices"])
        for subset in (splits.train, splits.val, splits.test)
        for entry in subset.anomalies
        if entry["type"] == "point"
    )
    assert total_indices == series.is_anomaly.sum().item()


def test_split_clips_span_anomaly_crossing_a_boundary() -> None:
    """A start_idx/end_idx span crossing a split boundary is clipped in both."""
    series = make_series(series_length=100, function_type="constant")
    series = add_level_shift(series, start_idx=40, duration=20, random_state=1)

    splits = series.split((0.5, 0.25, 0.25))  # boundaries at 50 and 75

    train_entry = next(e for e in splits.train.anomalies if e["type"] == "level_shift")
    assert train_entry["start_idx"] == 40
    assert train_entry["end_idx"] == 50

    val_entry = next(e for e in splits.val.anomalies if e["type"] == "level_shift")
    assert val_entry["start_idx"] == 0
    assert val_entry["end_idx"] == 10

    assert all(e["type"] != "level_shift" for e in splits.test.anomalies)


def test_split_drops_entries_entirely_outside_window() -> None:
    """An anomaly confined to one segment doesn't leak into the others."""
    series = make_series(series_length=100, function_type="constant")
    series = add_level_shift(series, start_idx=5, duration=10, random_state=1)

    splits = series.split((0.5, 0.25, 0.25))
    assert any(e["type"] == "level_shift" for e in splits.train.anomalies)
    assert not any(e["type"] == "level_shift" for e in splits.val.anomalies)
    assert not any(e["type"] == "level_shift" for e in splits.test.anomalies)


def test_split_method_matches_manual_slicing() -> None:
    """split() partitions y, x, and is_anomaly consistently with manual slicing."""
    series = make_series(series_length=200, function_type="linear")
    splits = series.split((0.5, 0.3, 0.2))

    assert torch.equal(splits.train.y, series.y[:100])
    assert torch.equal(splits.val.x, series.x[100:160])
    assert torch.equal(splits.test.is_anomaly, series.is_anomaly[160:])


def test_invalid_split_fractions_raises() -> None:
    """ValueError is raised when split fractions do not sum to one."""
    series = make_series(series_length=50, function_type="constant")
    with pytest.raises(ValueError, match="sum to 1"):
        series.split((0.5, 0.2, 0.2))


def test_split_unrecognised_anomaly_schema_raises() -> None:
    """An anomaly log entry without 'indices' or 'start_idx'/'end_idx' raises."""
    series = make_series(series_length=50, function_type="constant")
    series.anomalies.append({"type": "mystery"})
    with pytest.raises(ValueError, match="Cannot split anomaly log entry"):
        series.split((0.5, 0.3, 0.2))
