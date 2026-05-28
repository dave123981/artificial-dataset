"""Tests for the anomaly detection dataset generator."""

import math

import pytest
import torch

from artificial_dataset.anomaly import make_anomaly_dataset


def test_output_shapes() -> None:
    X, y = make_anomaly_dataset(n_samples=200, anomaly_fraction=0.1)
    assert X.shape == torch.Size([200, 2])
    assert y.shape == torch.Size([200])


def test_output_types() -> None:
    X, y = make_anomaly_dataset(n_samples=100, random_state=0)
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.dtype == torch.float32
    assert y.dtype == torch.long


def test_label_values() -> None:
    _, y = make_anomaly_dataset(n_samples=100, anomaly_fraction=0.1, random_state=1)
    assert set(y.unique().tolist()).issubset({0, 1})


def test_anomaly_count() -> None:
    n_samples, frac = 500, 0.1
    _, y = make_anomaly_dataset(n_samples=n_samples, anomaly_fraction=frac, random_state=2)
    assert int((y == 1).sum()) == round(n_samples * frac)
    assert int((y == 0).sum()) == n_samples - round(n_samples * frac)


def test_x_values_within_range() -> None:
    lo, hi = 0.0, 2.0 * math.pi
    X, _ = make_anomaly_dataset(n_samples=300, x_range=(lo, hi), random_state=3)
    assert float(X[:, 0].min()) >= lo
    assert float(X[:, 0].max()) <= hi


def test_reproducibility() -> None:
    X1, y1 = make_anomaly_dataset(n_samples=100, random_state=42)
    X2, y2 = make_anomaly_dataset(n_samples=100, random_state=42)
    assert torch.allclose(X1, X2)
    assert torch.equal(y1, y2)


def test_different_seeds_differ() -> None:
    X1, _ = make_anomaly_dataset(n_samples=100, random_state=0)
    X2, _ = make_anomaly_dataset(n_samples=100, random_state=1)
    assert not torch.allclose(X1, X2)


def test_anomaly_scale_increases_spread() -> None:
    _, y_low = make_anomaly_dataset(
        n_samples=1000, anomaly_fraction=0.3, anomaly_scale=1.0, random_state=5
    )
    X_high, y_high = make_anomaly_dataset(
        n_samples=1000, anomaly_fraction=0.3, anomaly_scale=20.0, random_state=5
    )
    anomaly_std = float(X_high[y_high == 1, 1].std())
    assert anomaly_std > 1.0


def test_invalid_anomaly_fraction_zero() -> None:
    with pytest.raises(ValueError, match="anomaly_fraction"):
        make_anomaly_dataset(anomaly_fraction=0.0)


def test_invalid_anomaly_fraction_one() -> None:
    with pytest.raises(ValueError, match="anomaly_fraction"):
        make_anomaly_dataset(anomaly_fraction=1.0)


def test_custom_signal_params() -> None:
    params = {
        "polynomial": {"coefficients": [0.0, 1.0, -0.1]},
        "linear": {"slope": 0.5, "intercept": 1.0},
    }
    X, y = make_anomaly_dataset(
        n_samples=200, signal_params=params, random_state=6
    )
    assert X.shape == torch.Size([200, 2])
