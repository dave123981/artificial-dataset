"""Synthetic time series dataset generation.

Defines the SyntheticSeries dataclass and entry-point functions for
generating synthetic 1D time series datasets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from artificial_dataset._components import compose, compose_weighted, gaussian_noise


@dataclass
class SyntheticSeries:
    """
    Data container for a generated synthetic 1D time series.

    Attributes
    ----------
    x : torch.Tensor
        Timeline array of shape (T,).
    y : torch.Tensor
        Series values array of shape (T,).
    is_anomaly : torch.Tensor
        Boolean mask of shape (T,), True at any anomalous timestep.
    anomaly_type : list[str]
        List of strings of length T, containing "" for normal timesteps or
        "|"-joined anomaly tags (e.g., "point|level_shift").
    anomalies : list[dict[str, Any]]
        Audit trail logging every injected anomaly and its metadata.
    meta : dict[str, Any]
        Generation metadata (e.g., function_type, parameters, noise_std).
    """

    x: torch.Tensor
    y: torch.Tensor
    is_anomaly: torch.Tensor
    anomaly_type: list[str]
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of samples in the series."""
        return len(self.x)

    def pipe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Pass self to `func(self, *args, **kwargs)` and return the result."""
        return func(self, *args, **kwargs)

    @property
    def label(self) -> torch.Tensor:
        """
        Binary classification target derived from `is_anomaly`.

        Returns
        -------
        torch.Tensor
            Integer tensor of shape (T,), dtype `torch.long`. 1 at every
            timestep flagged anomalous (`is_anomaly[i] == True`), 0
            otherwise. For continuous anomalies (e.g. level shifts,
            collective anomalies), every point in the affected span is
            marked 1, since injectors already flag each index in the span
            via `is_anomaly`.
        """
        return self.is_anomaly.to(torch.long)


def _generate_timeline(length: int) -> torch.tensor:
    """Return the x axis: 0, 1, 2, ..., length - 1 (a.u., step size 1)."""
    if length <= 0:
        raise ValueError("length must be a positive integer.")
    return torch.arange(length, dtype=torch.float32)


def make_series(
    series_length: int,
    function_type: str,
    function_params: dict[str, Any] | None = None,
    noise_std: float = 0.0,
    random_state: int | None = None,
) -> SyntheticSeries:
    """
    Generate a synthetic 1D time series using a single base function.

    Parameters
    ----------
    series_length : int
        Number of timesteps.
    function_type : str
        One of: constant, linear_trend, sinusoidal, exponential,
        logarithmic, periodic_seasonal.
    function_params : dict, optional
        Function-specific parameters.
    noise_std : float, default 0.0
        Standard deviation of additive Gaussian noise.
    random_state : int, optional
        Seed for reproducible noise.

    Returns
    -------
    SyntheticSeries
    """
    x = _generate_timeline(series_length)
    params = {function_type: function_params or {}}
    y = compose(x, params)

    if noise_std > 0.0:
        if random_state is not None:
            gen = torch.Generator()
            gen.manual_seed(random_state)
            noise = torch.randn(x.shape, generator=gen, dtype=torch.float32) * noise_std
        else:
            noise = gaussian_noise(x.shape, mean=0.0, std=noise_std)
        y = y + noise

    return SyntheticSeries(
        x=x,
        y=y,
        is_anomaly=torch.zeros(series_length, dtype=torch.bool),
        anomaly_type=[""] * series_length,
        anomalies=[],
        meta={
            "function_type": function_type,
            "function_params": function_params or {},
            "noise_std": noise_std,
        },
    )


def make_composite_series(
    series_length: int,
    components: list[dict[str, Any]],
    noise_std: float = 0.0,
    random_state: int | None = None,
) -> SyntheticSeries:
    """
    Generate a synthetic 1D time series by summing multiple weighted base functions.

    Parameters
    ----------
    series_length : int
        Number of timesteps.
    components : list of dict
        Each dict requires 'function_type', and optional 'function_params' & 'weight'.
    noise_std : float, default 0.0
        Standard deviation of additive Gaussian noise.
    random_state : int, optional
        Seed for reproducible noise.

    Returns
    -------
    SyntheticSeries
    """
    x = _generate_timeline(series_length)
    y = compose_weighted(x, components)

    if noise_std > 0.0:
        if random_state is not None:
            gen = torch.Generator()
            gen.manual_seed(random_state)
            noise = torch.randn(x.shape, generator=gen, dtype=torch.float32) * noise_std
        else:
            noise = gaussian_noise(x.shape, mean=0.0, std=noise_std)
        y = y + noise

    return SyntheticSeries(
        x=x,
        y=y,
        is_anomaly=torch.zeros(series_length, dtype=torch.bool),
        anomaly_type=[""] * series_length,
        anomalies=[],
        meta={
            "function_type": "composite",
            "components": components,
            "noise_std": noise_std,
        },
    )
