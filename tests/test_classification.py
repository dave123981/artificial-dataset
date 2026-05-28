"""Tests for the classification dataset generator."""

import math

import pytest
import torch

from artificial_dataset.classification import make_classification


def test_output_shapes_default() -> None:
    """Features and labels have the expected shapes for two classes."""
    feat, labels = make_classification(n_samples=100, n_classes=2)
    assert feat.shape == torch.Size([100, 2])
    assert labels.shape == torch.Size([100])


def test_output_shapes_multiclass() -> None:
    """Features and labels have the expected shapes for three classes."""
    feat, labels = make_classification(n_samples=90, n_classes=3)
    assert feat.shape == torch.Size([90, 2])
    assert labels.shape == torch.Size([90])


def test_output_types() -> None:
    """Feature tensor is float32 and label tensor is long."""
    feat, labels = make_classification(n_samples=50, random_state=0)
    assert isinstance(feat, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert feat.dtype == torch.float32
    assert labels.dtype == torch.long


def test_label_range() -> None:
    """Labels span exactly [0, n_classes - 1]."""
    _, labels = make_classification(n_samples=120, n_classes=4, random_state=1)
    assert int(labels.min()) == 0
    assert int(labels.max()) == 3


def test_class_sample_counts_even() -> None:
    """Samples are distributed evenly when n_samples is divisible by n_classes."""
    n_samples, n_classes = 100, 4
    _, labels = make_classification(
        n_samples=n_samples, n_classes=n_classes, random_state=2
    )
    for cls in range(n_classes):
        assert int((labels == cls).sum()) == n_samples // n_classes


def test_class_sample_counts_remainder() -> None:
    """Total sample count is preserved when n_samples is not divisible by n_classes."""
    n_samples, n_classes = 101, 3
    _, labels = make_classification(
        n_samples=n_samples, n_classes=n_classes, random_state=3
    )
    assert int(labels.shape[0]) == n_samples


def test_x_values_within_range() -> None:
    """Sampled x values stay inside the specified range."""
    lo, hi = 0.0, 2.0 * math.pi
    feat, _ = make_classification(n_samples=200, x_range=(lo, hi), random_state=4)
    assert float(feat[:, 0].min()) >= lo
    assert float(feat[:, 0].max()) <= hi


def test_reproducibility() -> None:
    """Same random seed produces identical outputs."""
    feat1, labels1 = make_classification(n_samples=60, random_state=42)
    feat2, labels2 = make_classification(n_samples=60, random_state=42)
    assert torch.allclose(feat1, feat2)
    assert torch.equal(labels1, labels2)


def test_different_seeds_differ() -> None:
    """Different seeds produce different feature matrices."""
    feat1, _ = make_classification(n_samples=60, random_state=0)
    feat2, _ = make_classification(n_samples=60, random_state=1)
    assert not torch.allclose(feat1, feat2)


def test_custom_class_params() -> None:
    """Custom class_params are accepted and the output shape is correct."""
    params = [
        {"linear": {"slope": 2.0, "intercept": 0.0}},
        {"sinusoidal": {"amplitude": 3.0, "frequency": 0.5, "phase": 0.0}},
    ]
    feat, _ = make_classification(
        n_samples=80, n_classes=2, class_params=params, random_state=5
    )
    assert feat.shape == torch.Size([80, 2])


def test_invalid_class_params_length() -> None:
    """ValueError is raised when class_params length mismatches n_classes."""
    with pytest.raises(ValueError, match="n_classes"):
        make_classification(n_samples=50, n_classes=3, class_params=[{}])


def test_noise_increases_variance() -> None:
    """Higher noise_std produces higher variance in the signal column."""
    _, _ = make_classification(n_samples=200, noise_std=0.0, random_state=7)
    feat_noisy, _ = make_classification(n_samples=200, noise_std=2.0, random_state=7)
    assert float(feat_noisy[:, 1].std()) > 0.5
