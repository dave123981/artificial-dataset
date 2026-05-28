"""Tests for the anomaly detection dataset generator."""

import math

import pytest
import torch

from artificial_dataset.anomaly import make_anomaly_dataset


def test_output_shapes() -> None:
    """Features and labels have the expected shapes."""
    feat, labels = make_anomaly_dataset(n_samples=200, anomaly_fraction=0.1)
    assert feat.shape == torch.Size([200, 2])
    assert labels.shape == torch.Size([200])


def test_output_types() -> None:
    """Feature tensor is float32 and label tensor is long."""
    feat, labels = make_anomaly_dataset(n_samples=100, random_state=0)
    assert isinstance(feat, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert feat.dtype == torch.float32
    assert labels.dtype == torch.long


def test_label_values() -> None:
    """Labels contain only 0 (normal) and 1 (anomaly)."""
    _, labels = make_anomaly_dataset(
        n_samples=100, anomaly_fraction=0.1, random_state=1
    )
    assert set(labels.unique().tolist()).issubset({0, 1})


def test_anomaly_count() -> None:
    """Anomaly and normal counts match the requested fraction."""
    n_samples, frac = 500, 0.1
    _, labels = make_anomaly_dataset(
        n_samples=n_samples, anomaly_fraction=frac, random_state=2
    )
    assert int((labels == 1).sum()) == round(n_samples * frac)
    assert int((labels == 0).sum()) == n_samples - round(n_samples * frac)


def test_x_values_within_range() -> None:
    """Sampled x values stay inside the specified range."""
    lo, hi = 0.0, 2.0 * math.pi
    feat, _ = make_anomaly_dataset(n_samples=300, x_range=(lo, hi), random_state=3)
    assert float(feat[:, 0].min()) >= lo
    assert float(feat[:, 0].max()) <= hi


def test_reproducibility() -> None:
    """Same random seed produces identical outputs."""
    feat1, labels1 = make_anomaly_dataset(n_samples=100, random_state=42)
    feat2, labels2 = make_anomaly_dataset(n_samples=100, random_state=42)
    assert torch.allclose(feat1, feat2)
    assert torch.equal(labels1, labels2)


def test_different_seeds_differ() -> None:
    """Different seeds produce different feature matrices."""
    feat1, _ = make_anomaly_dataset(n_samples=100, random_state=0)
    feat2, _ = make_anomaly_dataset(n_samples=100, random_state=1)
    assert not torch.allclose(feat1, feat2)


def test_anomaly_scale_increases_spread() -> None:
    """Higher anomaly_scale produces larger spread in anomalous signal values."""
    feat_high, labels_high = make_anomaly_dataset(
        n_samples=1000, anomaly_fraction=0.3, anomaly_scale=20.0, random_state=5
    )
    anomaly_std = float(feat_high[labels_high == 1, 1].std())
    assert anomaly_std > 1.0


def test_invalid_anomaly_fraction_zero() -> None:
    """ValueError is raised when anomaly_fraction is zero."""
    with pytest.raises(ValueError, match="anomaly_fraction"):
        make_anomaly_dataset(anomaly_fraction=0.0)


def test_invalid_anomaly_fraction_one() -> None:
    """ValueError is raised when anomaly_fraction is one."""
    with pytest.raises(ValueError, match="anomaly_fraction"):
        make_anomaly_dataset(anomaly_fraction=1.0)


def test_custom_signal_params() -> None:
    """Custom signal_params are accepted and the output shape is correct."""
    params = {
        "polynomial": {"coefficients": [0.0, 1.0, -0.1]},
        "linear": {"slope": 0.5, "intercept": 1.0},
    }
    feat, _ = make_anomaly_dataset(n_samples=200, signal_params=params, random_state=6)
    assert feat.shape == torch.Size([200, 2])
