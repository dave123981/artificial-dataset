"""Tests for the anomaly detection dataset generator."""

import math

import pytest
import torch

from artificial_dataset.anomaly import make_anomaly_dataset


def test_output_shapes() -> None:
    """x, y, and labels have the expected shapes for single-channel output."""
    x, y, labels = make_anomaly_dataset(n_samples=200, anomaly_fraction=0.1)
    assert x.shape == torch.Size([200])
    assert y.shape == torch.Size([200, 1])
    assert labels.shape == torch.Size([200])


def test_multichannel_shape() -> None:
    """Y has n_channels columns when multiple channel_params are given."""
    params = [
        {"sinusoidal": {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0}},
        {"linear": {"slope": 0.2, "intercept": 0.5}},
    ]
    x, y, labels = make_anomaly_dataset(n_samples=100, channel_params=params)
    assert x.shape == torch.Size([100])
    assert y.shape == torch.Size([100, 2])
    assert labels.shape == torch.Size([100])


def test_output_types() -> None:
    """X and y are float32; labels are long."""
    x, y, labels = make_anomaly_dataset(n_samples=100, random_state=0)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert labels.dtype == torch.long


def test_label_values() -> None:
    """Labels contain only 0 (normal) and 1 (anomaly)."""
    _, _, labels = make_anomaly_dataset(
        n_samples=100, anomaly_fraction=0.1, random_state=1
    )
    assert set(labels.unique().tolist()).issubset({0, 1})


def test_anomaly_count() -> None:
    """Anomaly and normal counts match the requested fraction."""
    n_samples, frac = 500, 0.1
    _, _, labels = make_anomaly_dataset(
        n_samples=n_samples, anomaly_fraction=frac, random_state=2
    )
    assert int((labels == 1).sum()) == round(n_samples * frac)
    assert int((labels == 0).sum()) == n_samples - round(n_samples * frac)


def test_x_values_within_range() -> None:
    """Sampled x values stay inside the specified range."""
    lo, hi = 0.0, 2.0 * math.pi
    x, _, _ = make_anomaly_dataset(n_samples=300, x_range=(lo, hi), random_state=3)
    assert float(x.min()) >= lo
    assert float(x.max()) <= hi


def test_reproducibility() -> None:
    """Same random seed produces identical outputs."""
    x1, y1, labels1 = make_anomaly_dataset(n_samples=100, random_state=42)
    x2, y2, labels2 = make_anomaly_dataset(n_samples=100, random_state=42)
    assert torch.allclose(x1, x2)
    assert torch.allclose(y1, y2)
    assert torch.equal(labels1, labels2)


def test_different_seeds_differ() -> None:
    """Different seeds produce different feature matrices."""
    _, y1, _ = make_anomaly_dataset(n_samples=100, random_state=0)
    _, y2, _ = make_anomaly_dataset(n_samples=100, random_state=1)
    assert not torch.allclose(y1, y2)


def test_anomaly_scale_increases_spread() -> None:
    """Higher anomaly_scale produces larger spread in anomalous signal values."""
    _, y_high, labels_high = make_anomaly_dataset(
        n_samples=1000, anomaly_fraction=0.3, anomaly_scale=20.0, random_state=5
    )
    anomaly_std = float(y_high[labels_high == 1, 0].std())
    assert anomaly_std > 1.0


def test_invalid_anomaly_fraction_zero() -> None:
    """ValueError is raised when anomaly_fraction is zero."""
    with pytest.raises(ValueError, match="anomaly_fraction"):
        make_anomaly_dataset(anomaly_fraction=0.0)


def test_invalid_anomaly_fraction_one() -> None:
    """ValueError is raised when anomaly_fraction is one."""
    with pytest.raises(ValueError, match="anomaly_fraction"):
        make_anomaly_dataset(anomaly_fraction=1.0)


def test_custom_channel_params() -> None:
    """Custom channel_params are accepted and the output shape is correct."""
    params = [
        {
            "polynomial": {"coefficients": [0.0, 1.0, -0.1]},
            "linear": {"slope": 0.5, "intercept": 1.0},
        }
    ]
    x, y, _ = make_anomaly_dataset(n_samples=200, channel_params=params, random_state=6)
    assert x.shape == torch.Size([200])
    assert y.shape == torch.Size([200, 1])
