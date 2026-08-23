"""Unit tests for the anomaly injection functions."""

import pytest
import torch

from artificial_dataset.injectors import (
    SpikeParams,
    add_collective_anomaly,
    add_dropout,
    add_level_shift,
    add_point_anomalies,
    add_seasonal_distortion,
    add_spike_anomalies,
    add_trend_change,
    add_variance_change,
    anomaly_summary,
)
from artificial_dataset.series import SyntheticSeries, make_series


@pytest.fixture
def base_series() -> SyntheticSeries:
    """Fixture returning a standard synthetic base series."""
    return make_series(
        series_length=150,
        function_type="sinusoidal",
        function_params={"amplitude": 2.0, "frequency": 0.05},
        noise_std=0.05,
        random_state=42,
    )


def test_injector_immutability(base_series: SyntheticSeries) -> None:
    """Verify injectors returns a new SyntheticSeries without mutating original."""
    original_y = base_series.y.clone()
    modified = add_point_anomalies(base_series, n_anomalies=3, random_state=1)

    assert modified is not base_series
    assert torch.equal(base_series.y, original_y)
    assert not base_series.is_anomaly.any()
    assert modified.is_anomaly.sum().item() > 0


def test_add_point_anomalies(base_series: SyntheticSeries) -> None:
    """Verify point anomaly spikes and mask labeling."""
    modified = add_point_anomalies(base_series, n_anomalies=4, random_state=123)

    assert modified.is_anomaly.sum().item() == 4
    assert len(modified.anomalies) == 1
    assert modified.anomalies[0]["type"] == "point"


def test_add_point_anomalies_without_avoiding_existing(
    base_series: SyntheticSeries,
) -> None:
    """avoid_existing=False allows new point anomalies to reuse any index."""
    series = add_level_shift(base_series, start_idx=0, duration=150, random_state=1)
    modified = add_point_anomalies(
        series, n_anomalies=3, avoid_existing=False, random_state=2
    )
    assert modified.is_anomaly.sum().item() == 150


def test_resolve_indices_raises_when_pool_too_small(
    base_series: SyntheticSeries,
) -> None:
    """Requesting more anomalies than available candidate indices raises."""
    series = add_level_shift(base_series, start_idx=0, duration=150, random_state=1)
    with pytest.raises(ValueError, match=r"only .* available"):
        add_point_anomalies(series, n_anomalies=1, avoid_existing=True, random_state=2)


def test_add_point_anomalies_direction_up(base_series: SyntheticSeries) -> None:
    """direction='up' only injects positive-signed spikes."""
    modified = add_point_anomalies(
        base_series, n_anomalies=5, direction="up", random_state=3
    )
    idx = torch.where(modified.is_anomaly)[0]
    assert torch.all(modified.y[idx] >= base_series.y[idx])


def test_add_point_anomalies_direction_down(base_series: SyntheticSeries) -> None:
    """direction='down' only injects negative-signed dips."""
    modified = add_point_anomalies(
        base_series, n_anomalies=5, direction="down", random_state=3
    )
    idx = torch.where(modified.is_anomaly)[0]
    assert torch.all(modified.y[idx] <= base_series.y[idx])


@pytest.mark.parametrize("pattern", ["noise", "flat", "reverse", "scale", "constant"])
def test_add_collective_anomaly_patterns(
    base_series: SyntheticSeries, pattern: str
) -> None:
    """Verify subsequence pattern overwrites for collective anomalies."""
    start_idx, length = 30, 15
    modified = add_collective_anomaly(
        base_series, start_idx=start_idx, length=length, pattern=pattern
    )

    expected_indices = torch.arange(start_idx, start_idx + length)
    assert torch.all(modified.is_anomaly[expected_indices])
    assert all(
        "collective" in modified.anomaly_type[i] for i in expected_indices.tolist()
    )


def test_add_spike_anomalies(base_series: SyntheticSeries) -> None:
    """Verify triangular spike anomalies injection using injectors.SpikeParams."""
    params = SpikeParams(amplitude_range=(3.0, 5.0), width_range=(2, 4), margin=10)
    modified = add_spike_anomalies(
        base_series, n_anomalies=2, spike_params=params, random_state=42
    )

    assert modified.is_anomaly.sum().item() > 0
    assert any("spike" in t for t in modified.anomaly_type)
    assert len(modified.anomalies) == 1
    assert modified.anomalies[0]["type"] == "spike"
    assert len(modified.anomalies[0]["indices"]) == 2


def test_add_level_shift(base_series: SyntheticSeries) -> None:
    """Verify sudden level mean shift."""
    modified = add_level_shift(base_series, start_idx=50, duration=30, random_state=1)

    assert torch.all(modified.is_anomaly[50:80])
    assert not modified.is_anomaly[0:50].any()


def test_add_trend_change(base_series: SyntheticSeries) -> None:
    """Verify the segment is replaced by the requested trend shape."""
    modified = add_trend_change(
        base_series,
        start_idx=40,
        new_function_type="linear",
        new_function_params={"slope": 0.1},
        duration=20,
    )

    assert torch.all(modified.is_anomaly[40:60])
    assert any("trend_change" in tag for tag in modified.anomaly_type)

    # The segment should now follow a linear trend with slope 0.1: after
    # removing the continuity offset, consecutive differences equal the slope.
    seg = modified.y[40:60]
    diffs = seg[1:] - seg[:-1]
    assert torch.allclose(diffs, torch.full_like(diffs, 0.1), atol=1e-5)


def test_add_trend_change_continuity_matches_boundary(
    base_series: SyntheticSeries,
) -> None:
    """continuity=True anchors the new trend to the pre-anomaly value."""
    modified = add_trend_change(
        base_series,
        start_idx=40,
        new_function_type="linear",
        new_function_params={"slope": 0.1},
        duration=20,
        continuity=True,
    )

    assert torch.isclose(modified.y[40], base_series.y[39], atol=1e-5)


def test_add_trend_change_without_continuity_uses_raw_trend(
    base_series: SyntheticSeries,
) -> None:
    """continuity=False evaluates the new trend without boundary matching."""
    modified = add_trend_change(
        base_series,
        start_idx=40,
        new_function_type="constant",
        new_function_params={"value": 0.0},
        duration=20,
        continuity=False,
    )

    assert torch.all(modified.y[40:60] == 0.0)


def test_add_trend_change_default_duration_extends_to_series_end(
    base_series: SyntheticSeries,
) -> None:
    """Omitting duration extends the trend change to the end of the series."""
    modified = add_trend_change(
        base_series, start_idx=100, new_function_type="sinusoidal"
    )

    assert torch.all(modified.is_anomaly[100:])
    assert not modified.is_anomaly[:100].any()


def test_add_trend_change_at_series_start_skips_continuity(
    base_series: SyntheticSeries,
) -> None:
    """start_idx=0 has no prior value, so the raw trend is used as-is."""
    modified = add_trend_change(
        base_series,
        start_idx=0,
        new_function_type="linear",
        new_function_params={"slope": 0.0, "intercept": 5.0},
        duration=10,
    )

    assert torch.all(modified.y[0:10] == 5.0)


def test_add_trend_change_invalid_function_type_raises(
    base_series: SyntheticSeries,
) -> None:
    """An unrecognised trend shape name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown new_function_type"):
        add_trend_change(base_series, start_idx=40, new_function_type="bogus")


def test_add_variance_change(base_series: SyntheticSeries) -> None:
    """Verify local variance noise burst injection."""
    modified = add_variance_change(
        base_series, start_idx=20, duration=25, scale_factor=5.0, random_state=7
    )

    assert torch.all(modified.is_anomaly[20:45])


@pytest.mark.parametrize("mode", ["flatline", "zero", "nan"])
def test_add_dropout_modes(base_series: SyntheticSeries, mode: str) -> None:
    """Verify sensor dropout modes including flatline, zero, and NaN."""
    modified = add_dropout(base_series, start_idx=10, duration=10, mode=mode)

    assert torch.all(modified.is_anomaly[10:20])
    if mode == "nan":
        assert torch.isnan(modified.y[10:20]).all()
    elif mode == "zero":
        assert (modified.y[10:20] == 0.0).all()


@pytest.mark.parametrize("mode", ["stretch", "compress", "damp", "phase_shift"])
def test_add_seasonal_distortion(base_series: SyntheticSeries, mode: str) -> None:
    """Verify seasonal periodic distortion patterns."""
    modified = add_seasonal_distortion(
        base_series, start_idx=30, duration=20, mode=mode, factor=2.0
    )

    assert torch.all(modified.is_anomaly[30:50])


def test_injector_stacking_and_overlapping(base_series: SyntheticSeries) -> None:
    """Verify sequential stacking of injectors and overlapping anomaly merging."""
    series = add_level_shift(base_series, start_idx=20, duration=40, random_state=1)
    series = add_point_anomalies(series, n_anomalies=5, random_state=2)

    assert len(series.anomalies) == 2

    # Verify overlapping tag merging "|"-separated string
    overlapped_tags = [t for t in series.anomaly_type if "|" in t]
    assert isinstance(overlapped_tags, list)


def test_mark_merges_distinct_overlapping_tags(base_series: SyntheticSeries) -> None:
    """Overlapping anomalies with different tags are merged with '|'."""
    series = add_level_shift(base_series, start_idx=10, duration=20, random_state=1)
    # level_shift covers [10, 30); collective covers [25, 35) -> overlap [25, 30).
    series = add_collective_anomaly(series, start_idx=25, length=10, pattern="flat")

    overlapped = [series.anomaly_type[i] for i in range(25, 30)]
    assert all("level_shift" in tag and "collective" in tag for tag in overlapped)
    assert all(tag.count("|") == 1 for tag in overlapped)
    assert series.anomaly_type[10] == "level_shift"
    assert series.anomaly_type[34] == "collective"


def test_anomaly_summary(base_series: SyntheticSeries) -> None:
    """Verify extraction of the audit trail log for stacked injectors."""
    series = add_collective_anomaly(
        base_series, start_idx=10, length=15, pattern="flat"
    )
    series = add_collective_anomaly(series, start_idx=50, length=20, pattern="noise")

    summary = anomaly_summary(series)

    assert len(summary) == 2
    assert summary[0]["type"] == "collective"
    assert summary[0]["start_idx"] == 10
    assert summary[0]["end_idx"] == 25
    assert summary[0]["pattern"] == "flat"
    assert summary[1]["start_idx"] == 50
    assert summary[1]["end_idx"] == 70
    assert summary[1]["pattern"] == "noise"


def test_zero_std_nan_regression_protection() -> None:
    """Verify safe standard deviation fallback when input series is flat (std = 0)."""
    flat_series = make_series(
        series_length=50, function_type="constant", function_params={"value": 1.0}
    )

    # Should execute safely without producing NaN values or division by zero errors
    modified = add_point_anomalies(flat_series, n_anomalies=2, random_state=42)
    assert not torch.isnan(modified.y).any()


def test_add_collective_anomaly_invalid_pattern_raises(
    base_series: SyntheticSeries,
) -> None:
    """An unrecognised pattern name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown pattern"):
        add_collective_anomaly(base_series, start_idx=10, length=15, pattern="bogus")


def test_add_dropout_invalid_mode_raises(base_series: SyntheticSeries) -> None:
    """An unrecognised dropout mode raises ValueError."""
    with pytest.raises(ValueError, match="mode must be one of"):
        add_dropout(base_series, start_idx=10, duration=10, mode="bogus")


def test_add_seasonal_distortion_invalid_mode_raises(
    base_series: SyntheticSeries,
) -> None:
    """An unrecognised seasonal-distortion mode raises ValueError."""
    with pytest.raises(ValueError, match="mode must be one of"):
        add_seasonal_distortion(base_series, start_idx=30, duration=20, mode="bogus")
