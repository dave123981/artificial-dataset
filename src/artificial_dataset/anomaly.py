"""Anomaly detection dataset generator built from signal components."""

import math
from typing import Any

import torch

from artificial_dataset._components import compose, gaussian_noise

_DEFAULT_SIGNAL_PARAMS: dict[str, Any] = {
    "sinusoidal": {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0},
    "linear": {"slope": 0.1, "intercept": 0.0},
}


def make_anomaly_dataset(
    n_samples: int = 1000,
    anomaly_fraction: float = 0.05,
    noise_std: float = 0.05,
    x_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    signal_params: dict[str, Any] | None = None,
    anomaly_scale: float = 6.0,
    random_state: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate an anomaly detection dataset from signal components.

    Normal samples follow a smooth signal built from a combination of
    linear, polynomial, and sinusoidal components with small Gaussian
    noise.  Anomalous samples are drawn from the same *x* distribution
    but deviate from the clean signal by a large noise term.

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples (normal + anomalous).
    anomaly_fraction : float, optional
        Fraction of samples marked as anomalies; must be in ``(0, 1)``.
    noise_std : float, optional
        Standard deviation of the additive Gaussian noise for normal
        samples.
    x_range : tuple[float, float], optional
        Closed interval ``[min, max]`` from which input values are drawn
        uniformly.
    signal_params : dict[str, Any] | None, optional
        Component configuration for the normal signal.  Passed directly to
        :func:`~artificial_dataset._components.compose`.  When *None*,
        a default sinusoidal + linear signal is used.
    anomaly_scale : float, optional
        Multiplier applied to *noise_std* for anomalous samples, so an
        anomaly deviates from the clean signal by roughly
        ``anomaly_scale * noise_std``.
    random_state : int | None, optional
        Seed passed to :func:`torch.manual_seed` for reproducibility.

    Returns
    -------
    X : torch.Tensor, shape (n_samples, 2)
        Feature matrix.  Column 0 contains the sampled *x* values; column 1
        contains the observed signal value.
    y : torch.Tensor, shape (n_samples,)
        Labels: ``0`` for normal samples, ``1`` for anomalies,
        dtype ``torch.long``.

    Raises
    ------
    ValueError
        If *anomaly_fraction* is not strictly inside ``(0, 1)``.

    Examples
    --------
    >>> X, y = make_anomaly_dataset(n_samples=500, anomaly_fraction=0.1, random_state=0)
    >>> X.shape, y.shape
    (torch.Size([500, 2]), torch.Size([500]))
    >>> int((y == 1).sum())
    50
    """
    if not 0.0 < anomaly_fraction < 1.0:
        raise ValueError(
            f"anomaly_fraction must be in (0, 1), got {anomaly_fraction}"
        )

    if random_state is not None:
        torch.manual_seed(random_state)

    if signal_params is None:
        signal_params = _DEFAULT_SIGNAL_PARAMS

    n_anomalies = round(n_samples * anomaly_fraction)
    n_normal = n_samples - n_anomalies
    x_min, x_max = x_range

    x_normal = torch.rand(n_normal) * (x_max - x_min) + x_min
    signal_normal = compose(x_normal, signal_params) + gaussian_noise(
        (n_normal,), mean=0.0, std=noise_std
    )

    x_anomaly = torch.rand(n_anomalies) * (x_max - x_min) + x_min
    signal_anomaly = compose(x_anomaly, signal_params) + gaussian_noise(
        (n_anomalies,), mean=0.0, std=noise_std * anomaly_scale
    )

    X = torch.cat(
        [
            torch.stack([x_normal, signal_normal], dim=1),
            torch.stack([x_anomaly, signal_anomaly], dim=1),
        ],
        dim=0,
    )
    y = torch.cat(
        [
            torch.zeros(n_normal, dtype=torch.long),
            torch.ones(n_anomalies, dtype=torch.long),
        ],
        dim=0,
    )
    return X, y
