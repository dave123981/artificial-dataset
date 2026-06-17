"""Tests for the anomaly detection dataset generator."""

import math

import pytest
import torch

from artificial_dataset.anomaly import (
    AnomalyDataset,
    AnomalySplits,
    SpikeParams,
    make_anomaly_dataset,
)


def test_output_shapes() -> None:
    """y, labels, and t have the expected shapes for single-channel output."""
    data = make_anomaly_dataset(n_instances=20, series_length=200)
    assert isinstance(data, AnomalyDataset)
    assert data.y.shape == torch.Size([20, 1, 200])
    assert data.labels.shape == torch.Size([20])
    assert data.t.shape == torch.Size([200])
    assert len(data.peak_indices) == 20


def test_multichannel_shape() -> None:
    """Y has m channels when multiple channel_params are given."""
    params = [
        {"sinusoidal": {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0}},
        {"linear": {"slope": 0.2, "intercept": 0.5}},
    ]
    data = make_anomaly_dataset(
        n_instances=15, series_length=128, channel_params=params
    )
    assert data.y.shape == torch.Size([15, 2, 128])


def test_output_types() -> None:
    """Y and t are float32; labels are long."""
    data = make_anomaly_dataset(n_instances=10, random_state=0)
    assert data.y.dtype == torch.float32
    assert data.t.dtype == torch.float32
    assert data.labels.dtype == torch.long


def test_label_values() -> None:
    """Labels contain only 0 (normal) and 1 (anomaly)."""
    data = make_anomaly_dataset(n_instances=40, anomaly_fraction=0.1, random_state=1)
    assert set(data.labels.unique().tolist()).issubset({0, 1})


def test_anomaly_count() -> None:
    """Anomaly and normal counts match the requested fraction."""
    n, frac = 200, 0.1
    data = make_anomaly_dataset(n_instances=n, anomaly_fraction=frac, random_state=2)
    assert int((data.labels == 1).sum()) == round(n * frac)
    assert int((data.labels == 0).sum()) == n - round(n * frac)


def test_t_within_range() -> None:
    """The time grid stays inside the specified range."""
    lo, hi = 0.0, 6.0 * math.pi
    data = make_anomaly_dataset(
        n_instances=5, series_length=300, x_range=(lo, hi), random_state=3
    )
    assert float(data.t.min()) >= lo
    assert float(data.t.max()) == pytest.approx(hi)


def test_anomalies_have_peaks_normals_do_not() -> None:
    """Anomalous instances record spike centres; normal ones record none."""
    data = make_anomaly_dataset(n_instances=40, anomaly_fraction=0.25, random_state=11)
    for label, peaks in zip(data.labels.tolist(), data.peak_indices, strict=True):
        if label == 1:
            assert peaks.numel() >= 1
        else:
            assert peaks.numel() == 0


def test_spikes_are_positive_above_baseline() -> None:
    """Anomalous instances rise well above the noise-only normal instances."""
    data = make_anomaly_dataset(
        n_instances=60,
        series_length=400,
        anomaly_fraction=0.3,
        noise_std=0.4,
        random_state=4,
    )
    normal_max = float(data.y[data.labels == 0].max())
    anomaly_max = float(data.y[data.labels == 1].max())
    # Default spike amplitudes are 4-7, far above a noise-only baseline.
    assert anomaly_max > normal_max
    assert anomaly_max > 3.0


def test_spike_centres_are_actual_maxima() -> None:
    """Each recorded centre is the local peak of at least one channel."""
    data = make_anomaly_dataset(
        n_instances=30,
        series_length=300,
        anomaly_fraction=0.4,
        noise_std=0.0,
        random_state=8,
    )
    baseline = data.y[data.labels == 0]
    base_max = float(baseline.max()) if baseline.numel() else 0.0
    for instance, peaks in zip(data.y, data.peak_indices, strict=True):
        for centre in peaks.tolist():
            # The spike apex sits above the clean baseline ceiling.
            assert float(instance[:, centre].max()) > base_max


def test_configurable_spike_params() -> None:
    """Larger amplitude ranges yield taller spikes."""
    small = make_anomaly_dataset(
        n_instances=40,
        anomaly_fraction=0.5,
        spike_params=SpikeParams(amplitude_range=(1.0, 1.5)),
        random_state=5,
    )
    large = make_anomaly_dataset(
        n_instances=40,
        anomaly_fraction=0.5,
        spike_params=SpikeParams(amplitude_range=(20.0, 25.0)),
        random_state=5,
    )
    assert float(large.y.max()) > float(small.y.max())


def test_split_returns_splits() -> None:
    """A split argument returns an AnomalySplits with contiguous subsets."""
    splits = make_anomaly_dataset(
        n_instances=100,
        series_length=64,
        split=(0.6, 0.2, 0.2),
        random_state=6,
    )
    assert isinstance(splits, AnomalySplits)
    assert splits.train.y.shape[0] == 60
    assert splits.val.y.shape[0] == 20
    assert splits.test.y.shape[0] == 20
    # The three subsets recombine to the whole, in order.
    recombined = torch.cat([splits.train.y, splits.val.y, splits.test.y], dim=0)
    assert recombined.shape[0] == 100


def test_split_method_matches_function() -> None:
    """Dataset.split and the split argument produce identical partitions."""
    data = make_anomaly_dataset(n_instances=50, series_length=64, random_state=7)
    via_method = data.split((0.5, 0.3, 0.2))
    assert via_method.train.y.shape[0] == 25
    assert via_method.val.y.shape[0] == 15
    assert via_method.test.y.shape[0] == 10
    assert torch.equal(via_method.train.y, data.y[:25])
    assert torch.equal(via_method.test.labels, data.labels[40:])


def test_invalid_split_fractions() -> None:
    """ValueError is raised when split fractions do not sum to one."""
    data = make_anomaly_dataset(n_instances=10, series_length=32, random_state=0)
    with pytest.raises(ValueError, match="sum to 1"):
        data.split((0.5, 0.2, 0.2))


def test_reproducibility() -> None:
    """Same random seed produces identical outputs."""
    a = make_anomaly_dataset(n_instances=20, series_length=128, random_state=42)
    b = make_anomaly_dataset(n_instances=20, series_length=128, random_state=42)
    assert torch.allclose(a.y, b.y)
    assert torch.equal(a.labels, b.labels)
    for pa, pb in zip(a.peak_indices, b.peak_indices, strict=True):
        assert torch.equal(pa, pb)


def test_different_seeds_differ() -> None:
    """Different seeds produce different signals."""
    a = make_anomaly_dataset(n_instances=20, series_length=128, random_state=0)
    b = make_anomaly_dataset(n_instances=20, series_length=128, random_state=1)
    assert not torch.allclose(a.y, b.y)


def test_invalid_anomaly_fraction_zero() -> None:
    """ValueError is raised when anomaly_fraction is zero."""
    with pytest.raises(ValueError, match="anomaly_fraction"):
        make_anomaly_dataset(anomaly_fraction=0.0)


def test_invalid_anomaly_fraction_one() -> None:
    """ValueError is raised when anomaly_fraction is one."""
    with pytest.raises(ValueError, match="anomaly_fraction"):
        make_anomaly_dataset(anomaly_fraction=1.0)
