"""Anomaly detection dataset generator built from signal components."""

import math
from typing import Any

import torch

from artificial_dataset._components import compose, gaussian_noise

_DEFAULT_CHANNEL_PARAMS: list[dict[str, Any]] = [
    {
        "sinusoidal": {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0},
        "linear": {"slope": 0.1, "intercept": 0.0},
    }
]


def make_anomaly_dataset(
    n_samples: int = 1000,
    anomaly_fraction: float = 0.05,
    noise_std: float = 0.05,
    x_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    channel_params: list[dict[str, Any]] | None = None,
    anomaly_scale: float = 6.0,
    random_state: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a multi-channel anomaly detection dataset from signal components.

    Normal samples follow a smooth signal built from a combination of
    linear, polynomial, and sinusoidal components with small Gaussian
    noise.  Anomalous samples are drawn from the same *x* distribution
    but deviate from the clean signal by a large noise term on every
    channel simultaneously.

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
    channel_params : list[dict[str, Any]] | None, optional
        One signal-parameter dict per channel.  Each dict is passed
        directly to :func:`~artificial_dataset._components.compose`.
        When *None*, a single default sinusoidal + linear channel is
        used.
    anomaly_scale : float, optional
        Multiplier applied to *noise_std* for anomalous samples, so an
        anomaly deviates from the clean signal by roughly
        ``anomaly_scale * noise_std`` on every channel.
    random_state : int | None, optional
        Seed passed to :func:`torch.manual_seed` for reproducibility.

    Returns
    -------
    x : torch.Tensor, shape (n_samples,)
        Sampled input coordinates; intended for plotting only and not
        required during classification.
    y : torch.Tensor, shape (n_samples, n_channels)
        Observed signal values.  Column *c* corresponds to
        ``channel_params[c]``.
    labels : torch.Tensor, shape (n_samples,)
        Class labels: ``0`` for normal samples, ``1`` for anomalies,
        dtype ``torch.long``.

    Raises
    ------
    ValueError
        If *anomaly_fraction* is not strictly inside ``(0, 1)``.

    Examples
    --------
    >>> x, y, labels = make_anomaly_dataset(
    ...     n_samples=500, anomaly_fraction=0.1, random_state=0
    ... )
    >>> x.shape, y.shape, labels.shape
    (torch.Size([500]), torch.Size([500, 1]), torch.Size([500]))
    >>> int((labels == 1).sum())
    50
    """
    if not 0.0 < anomaly_fraction < 1.0:
        raise ValueError(f"anomaly_fraction must be in (0, 1), got {anomaly_fraction}")

    if random_state is not None:
        torch.manual_seed(random_state)

    if channel_params is None:
        channel_params = _DEFAULT_CHANNEL_PARAMS

    n_anomalies = round(n_samples * anomaly_fraction)
    n_normal = n_samples - n_anomalies
    x_min, x_max = x_range

    x_normal = torch.rand(n_normal) * (x_max - x_min) + x_min
    x_anomaly = torch.rand(n_anomalies) * (x_max - x_min) + x_min

    normal_channels = [
        compose(x_normal, params) + gaussian_noise((n_normal,), mean=0.0, std=noise_std)
        for params in channel_params
    ]
    anomaly_channels = [
        compose(x_anomaly, params)
        + gaussian_noise((n_anomalies,), mean=0.0, std=noise_std * anomaly_scale)
        for params in channel_params
    ]

    x = torch.cat([x_normal, x_anomaly], dim=0)
    y = torch.cat(
        [
            torch.stack(normal_channels, dim=1),
            torch.stack(anomaly_channels, dim=1),
        ],
        dim=0,
    )
    labels = torch.cat(
        [
            torch.zeros(n_normal, dtype=torch.long),
            torch.ones(n_anomalies, dtype=torch.long),
        ],
        dim=0,
    )
    return x, y, labels
