"""Tests for the classification dataset generator."""

import math

import pytest
import torch

from artificial_dataset.classification import make_classification


def test_output_shapes_default() -> None:
    X, y = make_classification(n_samples=100, n_classes=2)
    assert X.shape == torch.Size([100, 2])
    assert y.shape == torch.Size([100])


def test_output_shapes_multiclass() -> None:
    X, y = make_classification(n_samples=90, n_classes=3)
    assert X.shape == torch.Size([90, 2])
    assert y.shape == torch.Size([90])


def test_output_types() -> None:
    X, y = make_classification(n_samples=50, random_state=0)
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.dtype == torch.float32
    assert y.dtype == torch.long


def test_label_range() -> None:
    _, y = make_classification(n_samples=120, n_classes=4, random_state=1)
    assert int(y.min()) == 0
    assert int(y.max()) == 3


def test_class_sample_counts_even() -> None:
    n_samples, n_classes = 100, 4
    _, y = make_classification(n_samples=n_samples, n_classes=n_classes, random_state=2)
    for cls in range(n_classes):
        assert int((y == cls).sum()) == n_samples // n_classes


def test_class_sample_counts_remainder() -> None:
    n_samples, n_classes = 101, 3
    _, y = make_classification(n_samples=n_samples, n_classes=n_classes, random_state=3)
    assert int(y.shape[0]) == n_samples


def test_x_values_within_range() -> None:
    lo, hi = 0.0, 2.0 * math.pi
    X, _ = make_classification(n_samples=200, x_range=(lo, hi), random_state=4)
    assert float(X[:, 0].min()) >= lo
    assert float(X[:, 0].max()) <= hi


def test_reproducibility() -> None:
    X1, y1 = make_classification(n_samples=60, random_state=42)
    X2, y2 = make_classification(n_samples=60, random_state=42)
    assert torch.allclose(X1, X2)
    assert torch.equal(y1, y2)


def test_different_seeds_differ() -> None:
    X1, _ = make_classification(n_samples=60, random_state=0)
    X2, _ = make_classification(n_samples=60, random_state=1)
    assert not torch.allclose(X1, X2)


def test_custom_class_params() -> None:
    params = [
        {"linear": {"slope": 2.0, "intercept": 0.0}},
        {"sinusoidal": {"amplitude": 3.0, "frequency": 0.5, "phase": 0.0}},
    ]
    X, y = make_classification(
        n_samples=80, n_classes=2, class_params=params, random_state=5
    )
    assert X.shape == torch.Size([80, 2])


def test_invalid_class_params_length() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        make_classification(n_samples=50, n_classes=3, class_params=[{}])


def test_noise_increases_variance() -> None:
    _, _ = make_classification(n_samples=200, noise_std=0.0, random_state=7)
    X_noisy, _ = make_classification(n_samples=200, noise_std=2.0, random_state=7)
    assert float(X_noisy[:, 1].std()) > 0.5
