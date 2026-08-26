"""Synthetic time series dataset generation.

Defines the SyntheticSeries dataclass and entry-point functions for
generating synthetic 1D time series datasets.
"""

from __future__ import annotations

import copy
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

    def split(self, fractions: tuple[float, float, float]) -> SyntheticSeriesSplits:
        """Split the series along the time axis into train, val, and test.

        The timeline is partitioned contiguously *from the beginning*: the
        first ``fractions[0]`` of the ``T`` timesteps go to ``train``, the
        next ``fractions[1]`` to ``val``, and the remainder to ``test``.
        Each subset's ``anomalies`` audit-log entries are clipped to that
        window and re-based to local sample positions; entries that fall
        entirely outside the window are dropped.

        Parameters
        ----------
        fractions : tuple[float, float, float]
            ``(train, val, test)`` fractions; must sum to ``1`` (within a
            small tolerance).

        Returns
        -------
        SyntheticSeriesSplits
            The three contiguous time segments, each a :class:`SyntheticSeries`.

        Raises
        ------
        ValueError
            If the fractions do not sum to ``1``.
        """
        f_train, f_val, f_test = fractions
        if abs(f_train + f_val + f_test - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1, got {fractions}")

        t_len = len(self)
        n_train = round(f_train * t_len)
        n_val = round(f_val * t_len)

        def _subset(lo: int, hi: int) -> SyntheticSeries:
            return SyntheticSeries(
                x=self.x[lo:hi],
                y=self.y[lo:hi],
                is_anomaly=self.is_anomaly[lo:hi],
                anomaly_type=self.anomaly_type[lo:hi],
                anomalies=_rebase_anomalies(self.anomalies, lo, hi),
                meta=copy.deepcopy(self.meta),
            )

        return SyntheticSeriesSplits(
            train=_subset(0, n_train),
            val=_subset(n_train, n_train + n_val),
            test=_subset(n_train + n_val, t_len),
        )


@dataclass
class SyntheticSeriesSplits:
    """Train/validation/test partition of a :class:`SyntheticSeries`.

    Attributes
    ----------
    train, val, test : SyntheticSeries
        The three disjoint time segments, cut contiguously from the
        beginning of the timeline.
    """

    train: SyntheticSeries
    val: SyntheticSeries
    test: SyntheticSeries


def _rebase_anomalies(
    anomalies: list[dict[str, Any]], lo: int, hi: int
) -> list[dict[str, Any]]:
    """Clip and re-base every anomaly audit-log entry to a ``[lo, hi)`` window.

    Entries with an ``"indices"`` key (used by point and spike anomalies)
    keep only the indices that fall inside the window. Entries with both
    ``"start_idx"`` and ``"end_idx"`` keys (used by every span-style
    injector, e.g. level shifts, trend changes, dropout) are clipped to the
    window. Entries that fall entirely outside the window are dropped from
    the result.

    Parameters
    ----------
    anomalies : list[dict[str, Any]]
        The full audit log, as stored on :attr:`SyntheticSeries.anomalies`.
    lo, hi : int
        Half-open ``[lo, hi)`` window, in original sample positions.

    Returns
    -------
    list[dict[str, Any]]
        Re-based entries whose support overlaps the window, in their
        original relative order.

    Raises
    ------
    ValueError
        If an entry has neither an ``"indices"`` key nor both ``"start_idx"``
        and ``"end_idx"`` keys, since its extent can't be determined.
    """
    rebased: list[dict[str, Any]] = []
    for original_entry in anomalies:
        entry = copy.deepcopy(original_entry)
        if "indices" in entry:
            indices = [i - lo for i in entry["indices"] if lo <= i < hi]
            if not indices:
                continue
            entry["indices"] = indices
        elif "start_idx" in entry and "end_idx" in entry:
            start = max(entry["start_idx"], lo)
            end = min(entry["end_idx"], hi)
            if start >= end:
                continue
            entry["start_idx"] = start - lo
            entry["end_idx"] = end - lo
        else:
            raise ValueError(
                f"Cannot split anomaly log entry {entry!r}: expected an "
                "'indices' key or both 'start_idx' and 'end_idx' keys."
            )
        rebased.append(entry)
    return rebased


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
