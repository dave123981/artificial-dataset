"""
Anomaly injectors for the base functions created with `make_series`.

Provides labeled anomaly injectors for SyntheticSeries instances.
All injectors accept a SyntheticSeries, perform a non-destructive copy,
and return a new SyntheticSeries with modified values and updated label metadata.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from artificial_dataset._components import compose
from artificial_dataset.series import SyntheticSeries

# Trend shapes recognised by `add_trend_change`, mirroring the keys understood
# by `artificial_dataset._components.compose`.
_KNOWN_TREND_TYPES = (
    "constant",
    "linear",
    "exponential",
    "logarithmic",
    "periodic_seasonal",
    "polynomial",
    "sinusoidal",
)


# ---------- Internal Helpers ----------


def _copy_series(series: SyntheticSeries) -> SyntheticSeries:
    """Create a deep copy of a SyntheticSeries to guarantee immutability."""
    return SyntheticSeries(
        x=series.x.clone(),
        y=series.y.clone(),
        is_anomaly=series.is_anomaly.clone(),
        anomaly_type=list(series.anomaly_type),
        anomalies=copy.deepcopy(series.anomalies),
        meta=copy.deepcopy(series.meta),
    )


def _mark(series: SyntheticSeries, idx: torch.Tensor, tag: str) -> None:
    """Flag indices as anomalous and append tag to labels."""
    series.is_anomaly[idx] = True
    idx_list = idx.tolist() if isinstance(idx, torch.Tensor) else list(idx)
    for i in idx_list:
        curr = series.anomaly_type[i]
        if curr == "":
            series.anomaly_type[i] = tag
        elif tag not in curr.split("|"):
            series.anomaly_type[i] = f"{curr}|{tag}"


def _log(series: SyntheticSeries, entry: dict[str, Any]) -> None:
    """Append entry to the series anomaly audit log."""
    series.anomalies.append(entry)


def _resolve_indices(
    n: int,
    length: int,
    existing_mask: torch.Tensor | None,
    avoid_existing: bool,
    gen: torch.Generator,
) -> torch.Tensor:
    """Pick `n` distinct indices, optionally avoiding existing anomalies."""
    if avoid_existing and existing_mask is not None:
        pool = torch.where(~existing_mask)[0]
    else:
        pool = torch.arange(length)

    if n > len(pool):
        raise ValueError(
            f"Requested {n} anomaly points, but only {len(pool)} available candidate \
                indices."
        )

    perm = torch.randperm(len(pool), generator=gen)
    chosen = pool[perm[:n]]
    sorted_idx, _ = torch.sort(chosen)
    return sorted_idx


# ---------- o ----------
# SpikeParams Dataclass
@dataclass(frozen=True)
class SpikeParams:
    """Configuration parameters for generating synthetic spike anomalies."""

    amplitude_range: tuple[float, float] = (4.0, 7.0)
    width_range: tuple[int, int] = (3, 6)
    margin: int = 20


# ---------- o ----------
# Injectors
def add_point_anomalies(
    series: SyntheticSeries,
    n_anomalies: int = 5,
    magnitude: float | tuple[float, float] = (3.0, 6.0),
    direction: str = "both",
    avoid_existing: bool = True,
    random_state: int | None = None,
) -> SyntheticSeries:
    """Inject single-point spikes or dips."""
    series = _copy_series(series)
    series_length = len(series)

    gen = torch.Generator()
    if random_state is not None:
        gen.manual_seed(random_state)

    idx = _resolve_indices(
        n_anomalies, series_length, series.is_anomaly, avoid_existing, gen
    )
    y_std = torch.std(series.y).item() or 1.0

    for i in idx:
        if isinstance(magnitude, tuple):
            r = torch.rand(1, generator=gen).item()
            mag = magnitude[0] + (magnitude[1] - magnitude[0]) * r
        else:
            mag = magnitude

        if direction == "up":
            sign = 1.0
        elif direction == "down":
            sign = -1.0
        else:
            sign = 1.0 if torch.rand(1, generator=gen).item() > 0.5 else -1.0

        series.y[i] += sign * mag * y_std

    _mark(series, idx, "point")
    _log(series, {"type": "point", "indices": idx.tolist()})
    return series


def add_spike_anomalies(
    series: SyntheticSeries,
    n_anomalies: int = 5,
    spike_params: SpikeParams | None = None,
    random_state: int | None = None,
) -> SyntheticSeries:
    """Inject triangular positive spike events."""
    series = _copy_series(series)
    params = spike_params or SpikeParams()
    t_len = len(series)

    gen = torch.Generator()
    if random_state is not None:
        gen.manual_seed(random_state)

    centres = torch.randint(
        params.margin,
        max(params.margin + 1, t_len - params.margin),
        (n_anomalies,),
        generator=gen,
    )

    for centre in centres:
        c = centre.item()
        w = torch.randint(
            params.width_range[0], params.width_range[1] + 1, (1,), generator=gen
        ).item()
        idx = torch.arange(max(0, c - w), min(t_len, c + w + 1))

        profile = 1.0 - torch.abs(idx.float() - float(c)) / (w + 1.0)
        amp_min, amp_max = params.amplitude_range
        amp = amp_min + (amp_max - amp_min) * torch.rand(1, generator=gen).item()

        series.y[idx] += amp * profile
        _mark(series, idx, "spike")

    _log(series, {"type": "spike", "indices": centres.tolist()})
    return series


def add_collective_anomaly(
    series: SyntheticSeries,
    start_idx: int,
    length: int = 20,
    pattern: str = "noise",
    magnitude: float = 3.0,
    random_state: int | None = None,
) -> SyntheticSeries:
    """Replace a subsequence with a collective anomaly pattern."""
    series = _copy_series(series)
    end_idx = min(start_idx + length, len(series))
    idx = torch.arange(start_idx, end_idx)

    gen = torch.Generator()
    if random_state is not None:
        gen.manual_seed(random_state)

    seg = series.y[idx].clone()
    seg_mean = torch.mean(seg)
    seg_std = torch.std(seg).item() or 1.0

    if pattern == "noise":
        series.y[idx] = seg_mean + torch.randn(len(idx), generator=gen) * (
            magnitude * seg_std
        )
    elif pattern == "flat":
        series.y[idx] = seg_mean
    elif pattern == "reverse":
        series.y[idx] = torch.flip(seg, dims=[0])
    elif pattern == "scale":
        series.y[idx] = seg_mean + magnitude * (seg - seg_mean)
    elif pattern == "constant":
        series.y[idx] = magnitude
    else:
        raise ValueError(
            f"Unknown pattern '{pattern}'. Choose from: noise, flat, reverse, scale, \
                constant."
        )

    _mark(series, idx, "collective")
    _log(
        series,
        {
            "type": "collective",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "pattern": pattern,
        },
    )
    return series


def add_level_shift(
    series: SyntheticSeries,
    start_idx: int,
    shift_magnitude: float | tuple[float, float] = (3.0, 5.0),
    duration: int | None = None,
    random_state: int | None = None,
) -> SyntheticSeries:
    """Apply a step shift in the mean value."""
    series = _copy_series(series)
    end_idx = (
        len(series) if duration is None else min(start_idx + duration, len(series))
    )
    idx = torch.arange(start_idx, end_idx)

    gen = torch.Generator()
    if random_state is not None:
        gen.manual_seed(random_state)

    if isinstance(shift_magnitude, tuple):
        mag = (
            shift_magnitude[0]
            + (shift_magnitude[1] - shift_magnitude[0])
            * torch.rand(1, generator=gen).item()
        )
    else:
        mag = shift_magnitude

    sign = 1.0 if torch.rand(1, generator=gen).item() > 0.5 else -1.0
    y_std = torch.std(series.y).item() or 1.0
    series.y[idx] += sign * mag * y_std

    _mark(series, idx, "level_shift")
    _log(series, {"type": "level_shift", "start_idx": start_idx, "end_idx": end_idx})
    return series


def add_trend_change(
    series: SyntheticSeries,
    start_idx: int,
    new_function_type: str,  # "constant" | "linear" | "exponential" |
    # "logarithmic" | "periodic_seasonal" |
    # "polynomial" | "sinusoidal"
    new_function_params: dict[str, Any] | None = None,
    duration: int | None = None,
    continuity: bool = True,
) -> SyntheticSeries:
    """Replace a segment's trend with a different known trend shape.

    Simulates a concept-drift-style anomaly: over ``[start_idx, end_idx)`` the
    series stops following its original generative shape (e.g. sinusoidal)
    and instead follows *new_function_type*, evaluated with
    *new_function_params*, on the segment's own time values. The new shape is
    computed by :func:`~artificial_dataset._components.compose`, so any
    single component recognised there (with all of its own parameters) can be
    used as the anomalous trend.

    Parameters
    ----------
    series : SyntheticSeries
        The base series to modify.
    start_idx : int
        Index (inclusive) where the trend change begins.
    new_function_type : str
        Name of the replacement trend shape. One of: constant, linear,
        exponential, logarithmic, periodic_seasonal, polynomial, sinusoidal.
    new_function_params : dict, optional
        Keyword arguments forwarded to the chosen trend function (e.g.
        ``{"slope": 0.5, "intercept": 0.0}`` for ``"linear"``). Defaults to
        that function's own defaults when omitted.
    duration : int, optional
        Length of the affected segment. Defaults to the rest of the series.
    continuity : bool, default True
        If True, the new trend is vertically shifted so its first value
        matches the series value immediately before *start_idx*, avoiding an
        artificial level jump at the boundary while still exposing the
        change in shape/slope. If False, the new trend is used exactly as
        computed, which may introduce a visible jump.

    Returns
    -------
    SyntheticSeries
        A new series with the segment's trend replaced.

    Raises
    ------
    ValueError
        If *new_function_type* is not a recognised trend shape.

    Examples
    --------
    >>> from artificial_dataset.series import make_series
    >>> base = make_series(
    ...     200, "sinusoidal", {"amplitude": 2.0, "frequency": 0.05}
    ... )
    >>> anomalous = add_trend_change(
    ...     base, start_idx=100, new_function_type="linear",
    ...     new_function_params={"slope": 0.05}, duration=50,
    ... )
    >>> bool(anomalous.is_anomaly[100:150].all())
    True
    """
    if new_function_type not in _KNOWN_TREND_TYPES:
        raise ValueError(
            f"Unknown new_function_type '{new_function_type}'. Choose from: "
            f"{', '.join(_KNOWN_TREND_TYPES)}."
        )

    series = _copy_series(series)
    series_length = len(series)
    end_idx = (
        series_length if duration is None else min(start_idx + duration, series_length)
    )
    idx = torch.arange(start_idx, end_idx)

    params = {new_function_type: new_function_params or {}}
    new_values = compose(series.x[idx], params)

    if continuity and start_idx > 0:
        offset = series.y[start_idx - 1] - new_values[0]
        new_values = new_values + offset

    series.y[idx] = new_values

    _mark(series, idx, "trend_change")
    _log(
        series,
        {
            "type": "trend_change",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "new_function_type": new_function_type,
            "new_function_params": new_function_params or {},
            "continuity": continuity,
        },
    )
    return series


def add_variance_change(
    series: SyntheticSeries,
    start_idx: int,
    duration: int = 20,
    scale_factor: float = 4.0,
    random_state: int | None = None,
) -> SyntheticSeries:
    """Inject extra Gaussian noise variance into a segment."""
    series = _copy_series(series)
    end_idx = min(start_idx + duration, len(series))
    idx = torch.arange(start_idx, end_idx)

    gen = torch.Generator()
    if random_state is not None:
        gen.manual_seed(random_state)

    y_std = torch.std(series.y).item() or 1.0
    noise = torch.randn(len(idx), generator=gen) * (scale_factor * y_std)
    series.y[idx] += noise

    _mark(series, idx, "variance_change")
    _log(
        series,
        {
            "type": "variance_change",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "scale_factor": scale_factor,
        },
    )
    return series


def add_dropout(
    series: SyntheticSeries,
    start_idx: int,
    duration: int = 10,
    mode: str = "flatline",
) -> SyntheticSeries:
    """Simulate missing or frozen sensor signal."""
    series = _copy_series(series)
    end_idx = min(start_idx + duration, len(series))
    idx = torch.arange(start_idx, end_idx)

    if mode == "flatline":
        fill_val = series.y[max(start_idx - 1, 0)].item()
        series.y[idx] = fill_val
    elif mode == "zero":
        series.y[idx] = 0.0
    elif mode == "nan":
        series.y[idx] = float("nan")
    else:
        raise ValueError("mode must be one of: flatline, zero, nan.")

    _mark(series, idx, "dropout")
    _log(
        series,
        {"type": "dropout", "start_idx": start_idx, "end_idx": end_idx, "mode": mode},
    )
    return series


def add_seasonal_distortion(
    series: SyntheticSeries,
    start_idx: int,
    duration: int = 30,
    mode: str = "stretch",
    factor: float = 2.0,
) -> SyntheticSeries:
    """Distort periodic pattern in a time series segment.

    Applies distortion via stretching, compressing, damping, or phase shifting.
    """
    series = _copy_series(series)
    end_idx = min(start_idx + duration, len(series))
    idx = torch.arange(start_idx, end_idx)
    seg = series.y[idx].clone()
    src_len = len(seg)

    if mode in ("stretch", "compress"):
        warped_len = max(
            2, round(src_len * factor if mode == "stretch" else src_len / factor)
        )
        seg_reshaped = seg.view(1, 1, -1)

        # Resample onto new grid, then interpolate back to original segment length
        warped = F.interpolate(
            seg_reshaped, size=warped_len, mode="linear", align_corners=True
        )
        resampled = F.interpolate(
            warped, size=src_len, mode="linear", align_corners=True
        )
        series.y[idx] = resampled.squeeze()
    elif mode == "damp":
        seg_mean = torch.mean(seg)
        series.y[idx] = seg_mean + (seg - seg_mean) / factor
    elif mode == "phase_shift":
        shift_val = round(factor)
        series.y[idx] = torch.roll(seg, shifts=shift_val)
    else:
        raise ValueError("mode must be one of: stretch, compress, damp, phase_shift.")

    _mark(series, idx, "seasonal_distortion")
    _log(
        series,
        {
            "type": "seasonal_distortion",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "mode": mode,
            "factor": factor,
        },
    )
    return series


# ---------- o ----------
# Summary
def anomaly_summary(series: SyntheticSeries) -> list[dict[str, Any]]:
    """Return the audit log list stored in series.anomalies."""
    return series.anomalies
