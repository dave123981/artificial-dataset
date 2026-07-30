"""
injectors.py

Provides labeled anomaly injectors for SyntheticSeries instances.
All injectors accept a SyntheticSeries, perform a non-destructive copy,
and return a new SyntheticSeries with modified values and updated label metadata.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from artificial_dataset.series import SyntheticSeries


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


def _log(series: SyntheticSeries, entry: Dict[str, Any]) -> None:
    """Append entry to the series anomaly audit log."""
    series.anomalies.append(entry)


def _resolve_indices(
    n: int,
    length: int,
    existing_mask: Optional[torch.Tensor],
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
            f"Requested {n} anomaly points, but only {len(pool)} available candidate indices."
        )

    perm = torch.randperm(len(pool), generator=gen)
    chosen = pool[perm[:n]]
    sorted_idx, _ = torch.sort(chosen)
    return sorted_idx


# ---------- o ----------
# SpikeParams Dataclass
@dataclass(frozen=True)
class SpikeParams:
    amplitude_range: tuple[float, float] = (4.0, 7.0)
    width_range: tuple[int, int] = (3, 6)
    margin: int = 20


# ---------- o ----------
# Injectors
def add_point_anomalies(
    series: SyntheticSeries,
    n_anomalies: int = 5,
    magnitude: Union[float, tuple[float, float]] = (3.0, 6.0),
    direction: str = "both",
    avoid_existing: bool = True,
    random_state: Optional[int] = None,
) -> SyntheticSeries:
    """Inject single-point spikes or dips."""
    series = _copy_series(series)
    N = len(series)

    gen = torch.Generator()
    if random_state is not None:
        gen.manual_seed(random_state)

    idx = _resolve_indices(n_anomalies, N, series.is_anomaly, avoid_existing, gen)
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
    spike_params: Optional[SpikeParams] = None,
    random_state: Optional[int] = None,
) -> SyntheticSeries:
    """Inject triangular positive spike events."""
    series = _copy_series(series)
    params = spike_params or SpikeParams()
    t_len = len(series)

    gen = torch.Generator()
    if random_state is not None:
        gen.manual_seed(random_state)

    centres = torch.randint(
        params.margin, max(params.margin + 1, t_len - params.margin), (n_anomalies,), generator=gen
    )

    for centre in centres:
        c = centre.item()
        w = torch.randint(params.width_range[0], params.width_range[1] + 1, (1,), generator=gen).item()
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
    random_state: Optional[int] = None,
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
        series.y[idx] = seg_mean + torch.randn(len(idx), generator=gen) * (magnitude * seg_std)
    elif pattern == "flat":
        series.y[idx] = seg_mean
    elif pattern == "reverse":
        series.y[idx] = torch.flip(seg, dims=[0])
    elif pattern == "scale":
        series.y[idx] = seg_mean + magnitude * (seg - seg_mean)
    elif pattern == "constant":
        series.y[idx] = magnitude
    else:
        raise ValueError(f"Unknown pattern '{pattern}'. Choose from: noise, flat, reverse, scale, constant.")

    _mark(series, idx, "collective")
    _log(series, {"type": "collective", "start_idx": start_idx, "end_idx": end_idx, "pattern": pattern})
    return series


def add_level_shift(
    series: SyntheticSeries,
    start_idx: int,
    shift_magnitude: Union[float, tuple[float, float]] = (3.0, 5.0),
    duration: Optional[int] = None,
    random_state: Optional[int] = None,
) -> SyntheticSeries:
    """Apply a step shift in the mean value."""
    series = _copy_series(series)
    end_idx = len(series) if duration is None else min(start_idx + duration, len(series))
    idx = torch.arange(start_idx, end_idx)

    gen = torch.Generator()
    if random_state is not None:
        gen.manual_seed(random_state)

    if isinstance(shift_magnitude, tuple):
        mag = shift_magnitude[0] + (shift_magnitude[1] - shift_magnitude[0]) * torch.rand(1, generator=gen).item()
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
    new_slope: float,
    duration: Optional[int] = None,
) -> SyntheticSeries:
    """Add a linear ramp changing local slope."""
    series = _copy_series(series)
    N = len(series)
    end_idx = N if duration is None else min(start_idx + duration, N)
    idx = torch.arange(start_idx, end_idx)

    ramp = new_slope * torch.arange(len(idx), dtype=torch.float32)
    series.y[idx] += ramp

    _mark(series, idx, "trend_change")
    _log(series, {"type": "trend_change", "start_idx": start_idx, "end_idx": end_idx, "new_slope": new_slope})
    return series


def add_variance_change(
    series: SyntheticSeries,
    start_idx: int,
    duration: int = 20,
    scale_factor: float = 4.0,
    random_state: Optional[int] = None,
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
    _log(series, {"type": "variance_change", "start_idx": start_idx, "end_idx": end_idx, "scale_factor": scale_factor})
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
    _log(series, {"type": "dropout", "start_idx": start_idx, "end_idx": end_idx, "mode": mode})
    return series


def add_seasonal_distortion(
    series: SyntheticSeries,
    start_idx: int,
    duration: int = 30,
    mode: str = "stretch",
    factor: float = 2.0,
) -> SyntheticSeries:
    """Distort periodic pattern via stretching, compressing, damping, or phase shifting."""
    series = _copy_series(series)
    end_idx = min(start_idx + duration, len(series))
    idx = torch.arange(start_idx, end_idx)
    seg = series.y[idx].clone()
    src_len = len(seg)

    if mode in ("stretch", "compress"):
        warped_len = max(2, int(round(src_len * factor if mode == "stretch" else src_len / factor)))
        seg_reshaped = seg.view(1, 1, -1)
        
        # Resample onto new grid, then interpolate back to original segment length
        warped = F.interpolate(seg_reshaped, size=warped_len, mode="linear", align_corners=True)
        resampled = F.interpolate(warped, size=src_len, mode="linear", align_corners=True)
        series.y[idx] = resampled.squeeze()
    elif mode == "damp":
        seg_mean = torch.mean(seg)
        series.y[idx] = seg_mean + (seg - seg_mean) / factor
    elif mode == "phase_shift":
        shift_val = int(round(factor))
        series.y[idx] = torch.roll(seg, shifts=shift_val)
    else:
        raise ValueError("mode must be one of: stretch, compress, damp, phase_shift.")

    _mark(series, idx, "seasonal_distortion")
    _log(series, {"type": "seasonal_distortion", "start_idx": start_idx, "end_idx": end_idx, "mode": mode, "factor": factor})
    return series

# ---------- o ----------
# Summary
def anomaly_summary(series: SyntheticSeries) -> List[Dict[str, Any]]:
    """Return the audit log list stored in series.anomalies."""
    return series.anomalies