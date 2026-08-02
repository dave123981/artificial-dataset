import pytest
import torch

from artificial_dataset.series import SyntheticSeries, make_series, make_composite_series


def test_synthetic_series_dataclass_initialization():
    """Verify SyntheticSeries dataclass instantiation and default field structures."""
    series_length = 50
    x = torch.arange(series_length, dtype=torch.float32)
    y = torch.zeros(series_length, dtype=torch.float32)
    is_anomaly = torch.zeros(series_length, dtype=torch.bool)
    anomaly_type = [""] * series_length
    anomalies: list[dict] = []

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


@pytest.mark.parametrize(
    "function_type, params",
    [
        ("constant", {"value": 3.5}),
        ("linear_trend", {"slope": 0.5, "intercept": 1.0}),
        ("sinusoidal", {"amplitude": 2.0, "frequency": 0.05}),
        ("exponential", {"initial_value": 1.0, "growth_rate": 0.02}),
        ("logarithmic", {"scale": 2.0, "shift": 1.0}),
        ("periodic_seasonal", {"period": 12.0, "amplitude": 1.5, "waveform": "triangle"}),
    ],
)
def test_make_series_single_functions(function_type: str, params: dict):
    """Verify make_series generates correct output shapes, types, and tags for base signals."""
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


def test_make_series_noise_reproducibility():
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


def test_make_series_composite():
    """Verify composite multi-component weighted series generation."""
    components = [
        {"function_type": "linear_trend", "function_params": {"slope": 0.1}, "weight": 1.0},
        {"function_type": "sinusoidal", "function_params": {"frequency": 0.05}, "weight": 0.5},
    ]

    series = make_composite_series(
        series_length=200,
        components=components,
        noise_std=0.0,
    )

    assert series.y.shape == (200,)
    assert isinstance(series.y, torch.Tensor)
    assert not series.is_anomaly.any()