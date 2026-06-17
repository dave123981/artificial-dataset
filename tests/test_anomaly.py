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
    """y, labels, t, and peak_indices have the expected shapes."""
    data = make_anomaly_dataset(series_length=200)
    assert isinstance(data, AnomalyDataset)
    assert data.y.shape == torch.Size([1, 200])
    assert data.labels.shape == torch.Size([200])
    assert data.t.shape == torch.Size([200])
    assert data.peak_indices.ndim == 1


def test_multichannel_shape() -> None:
    """Y has m rows when multiple channel_params are given."""
    params = [
        {"sinusoidal": {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0}},
        {"linear": {"slope": 0.2, "intercept": 0.5}},
    ]
    data = make_anomaly_dataset(series_length=128, channel_params=params)
    assert data.y.shape == torch.Size([2, 128])


def test_output_types() -> None:
    """Y and t are float32; labels and peak_indices are long."""
    data = make_anomaly_dataset(series_length=256, random_state=0)
    assert data.y.dtype == torch.float32
    assert data.t.dtype == torch.float32
    assert data.labels.dtype == torch.long
    assert data.peak_indices.dtype == torch.long


def test_label_values() -> None:
    """Labels contain only 0 (normal) and 1 (anomalous timestep)."""
    data = make_anomaly_dataset(series_length=400, random_state=1)
    assert set(data.labels.unique().tolist()).issubset({0, 1})


def test_peaks_are_sorted_and_marked() -> None:
    """Spike centres are sorted and fall on anomalous timesteps."""
    data = make_anomaly_dataset(series_length=500, random_state=2)
    peaks = data.peak_indices
    assert peaks.numel() >= 1
    assert torch.all(peaks[1:] >= peaks[:-1])
    assert torch.all(data.labels[peaks] == 1)


def test_t_within_range() -> None:
    """The time grid stays inside the specified range."""
    lo, hi = 0.0, 6.0 * math.pi
    data = make_anomaly_dataset(series_length=300, x_range=(lo, hi), random_state=3)
    assert float(data.t.min()) >= lo
    assert float(data.t.max()) == pytest.approx(hi)


def test_spikes_are_positive_above_baseline() -> None:
    """Spike apexes rise above the smooth, noise-free baseline ceiling."""
    data = make_anomaly_dataset(series_length=400, noise_std=0.0, random_state=4)
    # The default 0.6 * sin baseline never exceeds ~0.6; spikes add 4-7.
    assert float(data.y[:, data.peak_indices].max()) > 1.0
    assert float(data.y.max()) > 3.0


def test_configurable_spike_params() -> None:
    """Larger amplitude ranges yield taller spikes."""
    small = make_anomaly_dataset(
        series_length=500,
        spike_params=SpikeParams(amplitude_range=(1.0, 1.5)),
        random_state=5,
    )
    large = make_anomaly_dataset(
        series_length=500,
        spike_params=SpikeParams(amplitude_range=(20.0, 25.0)),
        random_state=5,
    )
    assert float(large.y.max()) > float(small.y.max())


def test_spike_count_is_configurable() -> None:
    """count_range bounds the number of spike events."""
    data = make_anomaly_dataset(
        series_length=800,
        spike_params=SpikeParams(count_range=(2, 2)),
        random_state=9,
    )
    assert data.peak_indices.numel() == 2


def test_split_along_time_axis() -> None:
    """A split argument cuts the timeline into contiguous segments."""
    splits = make_anomaly_dataset(
        series_length=256,
        channel_params=[
            {"sinusoidal": {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0}},
            {"linear": {"slope": 0.2, "intercept": 0.5}},
        ],
        split=(0.5, 0.25, 0.25),
        random_state=6,
    )
    assert isinstance(splits, AnomalySplits)
    assert splits.train.y.shape == torch.Size([2, 128])
    assert splits.val.y.shape == torch.Size([2, 64])
    assert splits.test.y.shape == torch.Size([2, 64])
    # The segments recombine to the whole timeline, in order.
    recombined = torch.cat([splits.train.y, splits.val.y, splits.test.y], dim=1)
    full = make_anomaly_dataset(
        series_length=256,
        channel_params=[
            {"sinusoidal": {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0}},
            {"linear": {"slope": 0.2, "intercept": 0.5}},
        ],
        random_state=6,
    )
    assert torch.equal(recombined, full.y)


def test_split_peaks_are_rebased() -> None:
    """Each segment's peak indices are local and still marked anomalous."""
    splits = make_anomaly_dataset(
        series_length=300, split=(0.34, 0.33, 0.33), random_state=7
    )
    for subset in (splits.train, splits.val, splits.test):
        seg_len = subset.y.shape[1]
        peaks = subset.peak_indices
        assert torch.all(peaks >= 0)
        assert torch.all(peaks < seg_len)
        assert torch.all(subset.labels[peaks] == 1)
    total = sum(s.peak_indices.numel() for s in (splits.train, splits.val, splits.test))
    full = make_anomaly_dataset(series_length=300, random_state=7)
    # Splitting keeps every spike whose centre lies inside the timeline.
    assert total == full.peak_indices.numel()


def test_split_method_matches_function() -> None:
    """Dataset.split and the split argument produce identical partitions."""
    data = make_anomaly_dataset(series_length=200, random_state=8)
    via_method = data.split((0.5, 0.3, 0.2))
    assert via_method.train.y.shape[1] == 100
    assert via_method.val.y.shape[1] == 60
    assert via_method.test.y.shape[1] == 40
    assert torch.equal(via_method.train.y, data.y[:, :100])
    assert torch.equal(via_method.test.labels, data.labels[160:])


def test_invalid_split_fractions() -> None:
    """ValueError is raised when split fractions do not sum to one."""
    data = make_anomaly_dataset(series_length=64, random_state=0)
    with pytest.raises(ValueError, match="sum to 1"):
        data.split((0.5, 0.2, 0.2))


def test_reproducibility() -> None:
    """Same random seed produces identical outputs."""
    a = make_anomaly_dataset(series_length=256, random_state=42)
    b = make_anomaly_dataset(series_length=256, random_state=42)
    assert torch.allclose(a.y, b.y)
    assert torch.equal(a.labels, b.labels)
    assert torch.equal(a.peak_indices, b.peak_indices)


def test_different_seeds_differ() -> None:
    """Different seeds produce different signals."""
    a = make_anomaly_dataset(series_length=256, random_state=0)
    b = make_anomaly_dataset(series_length=256, random_state=1)
    assert not torch.allclose(a.y, b.y)
