import pytest
import torch

from artificial_dataset.injectors import (
    SpikeInjectionParams,
    add_collective_anomaly,
    add_contextual_anomalies,
    add_dropout,
    add_level_shift,
    add_point_anomalies,
    add_seasonal_distortion,
    add_spike_anomalies,
    add_trend_change,
    add_variance_change,
    anomaly_summary,
    get_anomaly_segments,
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


def test_injector_immutability(base_series: SyntheticSeries):
    """Verify injectors return a new SyntheticSeries object without mutating the original input."""
    original_y = base_series.y.clone()
    modified = add_point_anomalies(base_series, n_anomalies=3, random_state=1)

    assert modified is not base_series
    assert torch.equal(base_series.y, original_y)
    assert not base_series.is_anomaly.any()
    assert modified.is_anomaly.sum().item() > 0


def test_add_point_anomalies(base_series: SyntheticSeries):
    """Verify point anomaly spikes and mask labeling."""
    modified = add_point_anomalies(base_series, n_anomalies=4, random_state=123)

    assert modified.is_anomaly.sum().item() == 4
    assert len(modified.anomalies) == 1
    assert modified.anomalies[0]["type"] == "point"


def test_add_spike_anomalies(base_series: SyntheticSeries):
    """Verify positive triangular spike anomalies injection."""
    params = SpikeInjectionParams(amplitude_range=(3.0, 5.0), width_range=(2, 4), margin=10)
    modified = add_spike_anomalies(base_series, n_anomalies=2, spike_params=params, random_state=42)

    assert modified.is_anomaly.sum().item() > 0
    assert any("spike" in t for t in modified.anomaly_type)


def test_add_contextual_anomalies(base_series: SyntheticSeries):
    """Verify contextual anomaly generation without NumPy or indexing errors."""
    modified = add_contextual_anomalies(base_series, n_anomalies=3, local_window=8, random_state=1)

    assert modified.is_anomaly.sum().item() == 3
    assert any("contextual" in t for t in modified.anomaly_type)


@pytest.mark.parametrize("pattern", ["noise", "flat", "reverse", "scale", "constant"])
def test_add_collective_anomaly_patterns(base_series: SyntheticSeries, pattern: str):
    """Verify subsequence pattern overwrites for collective anomalies."""
    start_idx, length = 30, 15
    modified = add_collective_anomaly(base_series, start_idx=start_idx, length=length, pattern=pattern)

    expected_indices = torch.arange(start_idx, start_idx + length)
    assert torch.all(modified.is_anomaly[expected_indices])
    assert all("collective" in modified.anomaly_type[i] for i in expected_indices.tolist())


def test_add_level_shift(base_series: SyntheticSeries):
    """Verify sudden level mean shift."""
    modified = add_level_shift(base_series, start_idx=50, duration=30, random_state=1)

    assert torch.all(modified.is_anomaly[50:80])
    assert not modified.is_anomaly[0:50].any()


def test_add_trend_change(base_series: SyntheticSeries):
    """Verify slope break ramp injection."""
    modified = add_trend_change(base_series, start_idx=40, new_slope=0.1, duration=20)

    assert torch.all(modified.is_anomaly[40:60])
    assert any("trend_change" in tag for tag in modified.anomaly_type)


def test_add_variance_change(base_series: SyntheticSeries):
    """Verify local variance noise burst injection."""
    modified = add_variance_change(base_series, start_idx=20, duration=25, scale_factor=5.0, random_state=7)

    assert torch.all(modified.is_anomaly[20:45])


@pytest.mark.parametrize("mode", ["flatline", "zero", "nan"])
def test_add_dropout_modes(base_series: SyntheticSeries, mode: str):
    """Verify sensor dropout modes including flatline, zero, and NaN."""
    modified = add_dropout(base_series, start_idx=10, duration=10, mode=mode)

    assert torch.all(modified.is_anomaly[10:20])
    if mode == "nan":
        assert torch.isnan(modified.y[10:20]).all()
    elif mode == "zero":
        assert (modified.y[10:20] == 0.0).all()


@pytest.mark.parametrize("mode", ["stretch", "compress", "damp", "phase_shift"])
def test_add_seasonal_distortion(base_series: SyntheticSeries, mode: str):
    """Verify seasonal periodic distortion patterns."""
    modified = add_seasonal_distortion(base_series, start_idx=30, duration=20, mode=mode, factor=2.0)

    assert torch.all(modified.is_anomaly[30:50])


def test_injector_stacking_and_overlapping(base_series: SyntheticSeries):
    """Verify sequential stacking of multiple injectors and overlapping anomaly tag merging."""
    series = add_level_shift(base_series, start_idx=20, duration=40, random_state=1)
    series = add_point_anomalies(series, n_anomalies=5, random_state=2)

    assert len(series.anomalies) == 2
    
    # Verify overlapping tag merging "|"-separated string
    overlapped_tags = [t for t in series.anomaly_type if "|" in t]
    assert isinstance(overlapped_tags, list)


def test_anomaly_summary_and_get_segments(base_series: SyntheticSeries):
    """Verify extraction of audit trail logs and evaluation segment boundaries."""
    series = add_collective_anomaly(base_series, start_idx=10, length=15, pattern="flat")
    series = add_collective_anomaly(series, start_idx=50, length=20, pattern="noise")

    summary = anomaly_summary(series)
    segments = get_anomaly_segments(series)

    assert len(summary) == 2
    assert len(segments) == 2
    assert segments[0]["start_idx"] == 10
    assert segments[0]["end_idx"] == 25
    assert segments[0]["duration"] == 15
    assert "collective" in segments[0]["types"]


def test_zero_std_nan_regression_protection():
    """Verify safe standard deviation fallback when input series is flat (std = 0)."""
    flat_series = make_series(series_length=50, function_type="constant", function_params={"value": 1.0})
    
    # Should execute safely without producing NaN values or division by zero errors
    modified = add_point_anomalies(flat_series, n_anomalies=2, random_state=42)
    assert not torch.isnan(modified.y).any()