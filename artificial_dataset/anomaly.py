"""Anomaly detection dataset generator built from signal components.

The generator produces a single multivariate time series of shape ``(m, T)``:
``m`` channels, each of length ``T``.  Every channel shares a smooth baseline
(a superposition of :mod:`~artificial_dataset._components` signals) plus
Gaussian measurement noise.  A configurable number of positive triangular
spikes are added on top: each spike event shares its centre and width across
all channels, while every channel responds with its own random amplitude.
Spikes are always additive and positive, so an anomaly is always a peak that
rises above the baseline.
"""

import math
from dataclasses import dataclass
from typing import Any, overload

import torch

from artificial_dataset._components import compose, gaussian_noise

_DEFAULT_CHANNEL_PARAMS: list[dict[str, Any]] = [
    {"sinusoidal": {"amplitude": 0.6, "frequency": 1.0, "phase": 0.0}}
]


@dataclass(frozen=True)
class SpikeParams:
    """Configuration of the random positive anomaly spikes.

    The series receives a random number of spike *events*.  Every event has a
    shared centre and width across channels, while its amplitude is sampled
    independently per channel, so all channels react to the same event with
    different magnitudes.  Every value is drawn uniformly from the inclusive
    range it controls, making the spikes random yet fully configurable.

    Attributes
    ----------
    amplitude_range : tuple[float, float]
        Inclusive ``(min, max)`` peak height of a spike, sampled per channel.
        Values are positive so spikes always rise above the baseline.
    width_range : tuple[int, int]
        Inclusive ``(min, max)`` half-width ``w`` of a spike in samples; the
        triangular bump spans ``[centre - w, centre + w]``.
    count_range : tuple[int, int]
        Inclusive ``(min, max)`` number of spike events in the series.
    margin : int
        Minimum distance (in samples) kept between a spike centre and either
        end of the series, so spikes are not clipped at the boundaries.
    """

    amplitude_range: tuple[float, float] = (4.0, 7.0)
    width_range: tuple[int, int] = (3, 6)
    count_range: tuple[int, int] = (3, 8)
    margin: int = 20


@dataclass
class AnomalyDataset:
    """A single multivariate anomaly-detection time series.

    Attributes
    ----------
    y : torch.Tensor, shape (m, T)
        Observed signal values for ``m`` channels, each of length ``T``.
        ``float32``.
    labels : torch.Tensor, shape (T,)
        Per-timestep anomaly mask: ``1`` where the timestep falls inside the
        support of a spike, ``0`` otherwise, dtype ``torch.long``.
    t : torch.Tensor, shape (T,)
        Shared time grid on which every channel baseline is evaluated.
    peak_indices : torch.Tensor, shape (k,)
        Ground-truth spike-event centres as sample positions into the series,
        dtype ``torch.long`` and sorted ascending.
    """

    y: torch.Tensor
    labels: torch.Tensor
    t: torch.Tensor
    peak_indices: torch.Tensor

    def split(self, fractions: tuple[float, float, float]) -> "AnomalySplits":
        """Split the series along the time axis into train, val, and test.

        The timeline is partitioned contiguously *from the beginning*: the
        first ``fractions[0]`` of the ``T`` timesteps go to ``train``, the next
        ``fractions[1]`` to ``val``, and the remainder to ``test``.  Each
        subset's ``peak_indices`` are filtered to the spikes whose centre falls
        in that window and re-based to local sample positions.

        Parameters
        ----------
        fractions : tuple[float, float, float]
            ``(train, val, test)`` fractions; must sum to ``1`` (within a
            small tolerance).

        Returns
        -------
        AnomalySplits
            The three contiguous time segments, each an :class:`AnomalyDataset`.

        Raises
        ------
        ValueError
            If the fractions do not sum to ``1``.
        """
        f_train, f_val, f_test = fractions
        if abs(f_train + f_val + f_test - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1, got {fractions}")

        t_len = int(self.y.shape[1])
        n_train = round(f_train * t_len)
        n_val = round(f_val * t_len)

        def _subset(lo: int, hi: int) -> AnomalyDataset:
            peaks = self.peak_indices
            in_window = (peaks >= lo) & (peaks < hi)
            return AnomalyDataset(
                y=self.y[:, lo:hi],
                labels=self.labels[lo:hi],
                t=self.t[lo:hi],
                peak_indices=peaks[in_window] - lo,
            )

        return AnomalySplits(
            train=_subset(0, n_train),
            val=_subset(n_train, n_train + n_val),
            test=_subset(n_train + n_val, t_len),
        )


@dataclass
class AnomalySplits:
    """Train/validation/test partition of an :class:`AnomalyDataset`.

    Attributes
    ----------
    train, val, test : AnomalyDataset
        The three disjoint time segments, cut contiguously from the beginning
        of the timeline.
    """

    train: AnomalyDataset
    val: AnomalyDataset
    test: AnomalyDataset


@overload
def make_anomaly_dataset(
    series_length: int = ...,
    noise_std: float = ...,
    x_range: tuple[float, float] = ...,
    channel_params: list[dict[str, Any]] | None = ...,
    spike_params: SpikeParams | None = ...,
    split: None = ...,
    random_state: int | None = ...,
) -> AnomalyDataset: ...


@overload
def make_anomaly_dataset(
    series_length: int = ...,
    noise_std: float = ...,
    x_range: tuple[float, float] = ...,
    channel_params: list[dict[str, Any]] | None = ...,
    spike_params: SpikeParams | None = ...,
    *,
    split: tuple[float, float, float],
    random_state: int | None = ...,
) -> AnomalySplits: ...


def make_anomaly_dataset(
    series_length: int = 1000,
    noise_std: float = 0.4,
    x_range: tuple[float, float] = (0.0, 6.0 * math.pi),
    channel_params: list[dict[str, Any]] | None = None,
    spike_params: SpikeParams | None = None,
    split: tuple[float, float, float] | None = None,
    random_state: int | None = None,
) -> AnomalyDataset | AnomalySplits:
    """Generate a single multivariate anomaly-detection time series.

    Every channel shares a smooth baseline built from
    :func:`~artificial_dataset._components.compose` plus Gaussian measurement
    noise.  A random number of *positive* triangular spikes are then added: each
    spike event shares its centre and width across all channels, while every
    channel responds with its own random amplitude sampled from *spike_params*.

    Parameters
    ----------
    series_length : int, optional
        Length ``T`` of the time series.
    noise_std : float, optional
        Standard deviation of the additive Gaussian measurement noise applied
        to every channel.
    x_range : tuple[float, float], optional
        Closed interval ``[min, max]`` spanned by the shared time grid on which
        the baselines are evaluated.
    channel_params : list[dict[str, Any]] | None, optional
        One signal-parameter dict per channel, each passed to
        :func:`~artificial_dataset._components.compose`.  The number of entries
        sets the channel count ``m``.  When *None*, a single default
        sinusoidal channel is used.
    spike_params : SpikeParams | None, optional
        Configuration of the random positive anomaly spikes.  When *None*, the
        defaults of :class:`SpikeParams` are used.
    split : tuple[float, float, float] | None, optional
        When given, the ``(train, val, test)`` fractions used to split the
        series along the time axis *from the beginning*; the function then
        returns an :class:`AnomalySplits`.  When *None*, a single
        :class:`AnomalyDataset` is returned.
    random_state : int | None, optional
        Seed passed to :func:`torch.manual_seed` for reproducibility.

    Returns
    -------
    AnomalyDataset or AnomalySplits
        An :class:`AnomalyDataset` when *split* is *None*, otherwise an
        :class:`AnomalySplits` holding the three time segments.

    Examples
    --------
    >>> data = make_anomaly_dataset(series_length=200, random_state=0)
    >>> data.y.shape, data.labels.shape, data.t.shape
    (torch.Size([1, 200]), torch.Size([200]), torch.Size([200]))
    >>> bool((data.labels[data.peak_indices] == 1).all())
    True
    >>> splits = make_anomaly_dataset(
    ...     series_length=200, split=(0.5, 0.25, 0.25), random_state=0
    ... )
    >>> splits.train.y.shape[1], splits.val.y.shape[1], splits.test.y.shape[1]
    (100, 50, 50)
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    if channel_params is None:
        channel_params = _DEFAULT_CHANNEL_PARAMS
    if spike_params is None:
        spike_params = SpikeParams()

    t_len = series_length
    x_min, x_max = x_range

    t = torch.linspace(x_min, x_max, t_len)
    baselines = torch.stack([compose(t, params) for params in channel_params])

    y = baselines + gaussian_noise(baselines.shape, mean=0.0, std=noise_std)
    centres, anomaly_mask = _add_spikes(y, spike_params)

    labels = anomaly_mask.to(torch.long)
    dataset = AnomalyDataset(y=y, labels=labels, t=t, peak_indices=centres)

    if split is None:
        return dataset
    return dataset.split(split)


def _randint(low: int, high: int) -> int:
    """Draw a single integer uniformly from the inclusive range ``[low, high]``.

    Parameters
    ----------
    low, high : int
        Inclusive bounds of the range.

    Returns
    -------
    int
        The sampled integer.
    """
    return int(torch.randint(low, high + 1, (1,)).item())


def _add_spikes(
    series: torch.Tensor, spike_params: SpikeParams
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add random positive triangular spikes to the series in place.

    Parameters
    ----------
    series : torch.Tensor, shape (m, T)
        The multivariate series, modified in place.
    spike_params : SpikeParams
        Ranges controlling the random spike amplitude, width, and count.

    Returns
    -------
    centres : torch.Tensor, shape (k,)
        Sorted ``LongTensor`` of spike-event centres.
    mask : torch.Tensor, shape (T,)
        Boolean per-timestep mask, ``True`` inside the support of any spike.
    """
    m, t_len = series.shape
    amp_lo, amp_hi = spike_params.amplitude_range
    margin = spike_params.margin

    mask = torch.zeros(t_len, dtype=torch.bool)
    n_events = _randint(*spike_params.count_range)
    centres = torch.randint(margin, t_len - margin, (n_events,)).sort().values

    for centre in centres.tolist():
        width = _randint(*spike_params.width_range)
        idx = torch.arange(centre - width, centre + width + 1)
        idx = idx[(idx >= 0) & (idx < t_len)]
        profile = 1.0 - (idx - centre).abs().to(torch.float32) / (width + 1)
        mask[idx] = True
        for channel in range(m):
            amplitude = amp_lo + torch.rand(1).item() * (amp_hi - amp_lo)
            series[channel, idx] += amplitude * profile

    return centres, mask
