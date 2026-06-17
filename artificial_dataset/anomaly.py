"""Anomaly detection dataset generator built from signal components.

The generator produces a collection of multivariate time-series *instances*.
Every instance shares a smooth, multi-channel baseline (a superposition of
:mod:`~artificial_dataset._components` signals) plus Gaussian measurement
noise.  A configurable fraction of the instances are *anomalous*: they carry
one or more positive triangular spikes whose amplitude, width, count, and
location are drawn at random within user-controlled ranges.  Spikes are always
additive and positive, so an anomaly is always a peak that rises above the
baseline.
"""

import math
from dataclasses import dataclass, field
from typing import Any, overload

import torch

from artificial_dataset._components import compose, gaussian_noise

_DEFAULT_CHANNEL_PARAMS: list[dict[str, Any]] = [
    {"sinusoidal": {"amplitude": 0.6, "frequency": 1.0, "phase": 0.0}}
]


@dataclass(frozen=True)
class SpikeParams:
    """Configuration of the random positive anomaly spikes.

    Each anomalous instance receives a random number of spike *events*.  Every
    event has a shared centre and width across channels, while its amplitude is
    sampled independently per channel, so all channels react to the same event
    with different magnitudes.  Every value is drawn uniformly from the
    inclusive range it controls, making the spikes random yet fully
    configurable.

    Attributes
    ----------
    amplitude_range : tuple[float, float]
        Inclusive ``(min, max)`` peak height of a spike, sampled per channel.
        Values are positive so spikes always rise above the baseline.
    width_range : tuple[int, int]
        Inclusive ``(min, max)`` half-width ``w`` of a spike in samples; the
        triangular bump spans ``[centre - w, centre + w]``.
    count_range : tuple[int, int]
        Inclusive ``(min, max)`` number of spike events per anomalous instance.
    margin : int
        Minimum distance (in samples) kept between a spike centre and either
        end of the series, so spikes are not clipped at the boundaries.
    """

    amplitude_range: tuple[float, float] = (4.0, 7.0)
    width_range: tuple[int, int] = (3, 6)
    count_range: tuple[int, int] = (1, 4)
    margin: int = 20


@dataclass
class AnomalyDataset:
    """A bundle of multivariate anomaly-detection time-series instances.

    Attributes
    ----------
    y : torch.Tensor, shape (N, m, T)
        Observed signal values for ``N`` instances of ``m`` channels, each of
        length ``T``.  ``float32``.
    labels : torch.Tensor, shape (N,)
        Per-instance class labels: ``0`` for normal, ``1`` for anomalous,
        dtype ``torch.long``.
    t : torch.Tensor, shape (T,)
        Shared time grid on which every channel baseline is evaluated.
    peak_indices : list[torch.Tensor]
        Ground-truth spike-event centres, one ``LongTensor`` per instance
        (empty for normal instances).  ``peak_indices[n]`` gives the sample
        positions of the spikes in instance ``n``.
    """

    y: torch.Tensor
    labels: torch.Tensor
    t: torch.Tensor
    peak_indices: list[torch.Tensor] = field(default_factory=list)

    def split(self, fractions: tuple[float, float, float]) -> "AnomalySplits":
        """Split the instances into train, validation, and test subsets.

        Instances are partitioned contiguously *from the beginning*: the first
        ``fractions[0]`` go to ``train``, the next ``fractions[1]`` to
        ``val``, and the remainder to ``test``.

        Parameters
        ----------
        fractions : tuple[float, float, float]
            ``(train, val, test)`` fractions; must sum to ``1`` (within a
            small tolerance).

        Returns
        -------
        AnomalySplits
            The three subsets, each an :class:`AnomalyDataset` sharing the same
            time grid ``t``.

        Raises
        ------
        ValueError
            If the fractions do not sum to ``1``.
        """
        f_train, f_val, f_test = fractions
        if abs(f_train + f_val + f_test - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1, got {fractions}")

        n = int(self.y.shape[0])
        n_train = round(f_train * n)
        n_val = round(f_val * n)

        def _subset(lo: int, hi: int) -> AnomalyDataset:
            return AnomalyDataset(
                y=self.y[lo:hi],
                labels=self.labels[lo:hi],
                t=self.t,
                peak_indices=self.peak_indices[lo:hi],
            )

        return AnomalySplits(
            train=_subset(0, n_train),
            val=_subset(n_train, n_train + n_val),
            test=_subset(n_train + n_val, n),
        )


@dataclass
class AnomalySplits:
    """Train/validation/test partition of an :class:`AnomalyDataset`.

    Attributes
    ----------
    train, val, test : AnomalyDataset
        The three disjoint subsets, cut contiguously from the beginning of the
        original instance ordering.
    """

    train: AnomalyDataset
    val: AnomalyDataset
    test: AnomalyDataset


@overload
def make_anomaly_dataset(
    n_instances: int = ...,
    series_length: int = ...,
    anomaly_fraction: float = ...,
    noise_std: float = ...,
    x_range: tuple[float, float] = ...,
    channel_params: list[dict[str, Any]] | None = ...,
    spike_params: SpikeParams | None = ...,
    split: None = ...,
    random_state: int | None = ...,
) -> AnomalyDataset: ...


@overload
def make_anomaly_dataset(
    n_instances: int = ...,
    series_length: int = ...,
    anomaly_fraction: float = ...,
    noise_std: float = ...,
    x_range: tuple[float, float] = ...,
    channel_params: list[dict[str, Any]] | None = ...,
    spike_params: SpikeParams | None = ...,
    *,
    split: tuple[float, float, float],
    random_state: int | None = ...,
) -> AnomalySplits: ...


def make_anomaly_dataset(
    n_instances: int = 100,
    series_length: int = 1000,
    anomaly_fraction: float = 0.05,
    noise_std: float = 0.4,
    x_range: tuple[float, float] = (0.0, 6.0 * math.pi),
    channel_params: list[dict[str, Any]] | None = None,
    spike_params: SpikeParams | None = None,
    split: tuple[float, float, float] | None = None,
    random_state: int | None = None,
) -> AnomalyDataset | AnomalySplits:
    """Generate a multivariate anomaly-detection time-series dataset.

    Every instance shares a smooth ``m``-channel baseline (each channel built
    from :func:`~artificial_dataset._components.compose`) plus Gaussian
    measurement noise.  A fraction of the instances are anomalous: they carry
    one or more *positive* triangular spikes whose amplitude, width, count, and
    location are sampled at random within the ranges of *spike_params*.
    Anomalous instances are scattered randomly through the collection, so a
    contiguous :meth:`AnomalyDataset.split` still yields representative
    subsets.

    Parameters
    ----------
    n_instances : int, optional
        Number of time-series instances ``N`` to generate.
    series_length : int, optional
        Length ``T`` of every channel time series.
    anomaly_fraction : float, optional
        Fraction of instances marked as anomalies; must be in ``(0, 1)``.
    noise_std : float, optional
        Standard deviation of the additive Gaussian measurement noise applied
        to every channel of every instance.
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
        instances *from the beginning*; the function then returns an
        :class:`AnomalySplits`.  When *None*, a single :class:`AnomalyDataset`
        is returned.
    random_state : int | None, optional
        Seed passed to :func:`torch.manual_seed` for reproducibility.

    Returns
    -------
    AnomalyDataset or AnomalySplits
        An :class:`AnomalyDataset` when *split* is *None*, otherwise an
        :class:`AnomalySplits` holding the three subsets.

    Raises
    ------
    ValueError
        If *anomaly_fraction* is not strictly inside ``(0, 1)``.

    Examples
    --------
    >>> data = make_anomaly_dataset(
    ...     n_instances=20, series_length=200, random_state=0
    ... )
    >>> data.y.shape, data.labels.shape, data.t.shape
    (torch.Size([20, 1, 200]), torch.Size([20]), torch.Size([200]))
    >>> splits = make_anomaly_dataset(
    ...     n_instances=20, split=(0.5, 0.25, 0.25), random_state=0
    ... )
    >>> splits.train.y.shape[0], splits.val.y.shape[0], splits.test.y.shape[0]
    (10, 5, 5)
    """
    if not 0.0 < anomaly_fraction < 1.0:
        raise ValueError(f"anomaly_fraction must be in (0, 1), got {anomaly_fraction}")

    if random_state is not None:
        torch.manual_seed(random_state)

    if channel_params is None:
        channel_params = _DEFAULT_CHANNEL_PARAMS
    if spike_params is None:
        spike_params = SpikeParams()

    n = n_instances
    m = len(channel_params)
    t_len = series_length
    x_min, x_max = x_range

    t = torch.linspace(x_min, x_max, t_len)
    baselines = torch.stack([compose(t, params) for params in channel_params])

    y = baselines.unsqueeze(0).expand(n, m, t_len).clone()
    y = y + gaussian_noise((n, m, t_len), mean=0.0, std=noise_std)

    n_anomalies = round(n * anomaly_fraction)
    is_anomaly = torch.zeros(n, dtype=torch.bool)
    is_anomaly[torch.randperm(n)[:n_anomalies]] = True

    peak_indices: list[torch.Tensor] = []
    for instance in range(n):
        if is_anomaly[instance]:
            centres = _add_spikes(y[instance], spike_params)
            peak_indices.append(centres)
        else:
            peak_indices.append(torch.empty(0, dtype=torch.long))

    labels = is_anomaly.to(torch.long)
    dataset = AnomalyDataset(y=y, labels=labels, t=t, peak_indices=peak_indices)

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


def _add_spikes(instance: torch.Tensor, spike_params: SpikeParams) -> torch.Tensor:
    """Add random positive triangular spikes to one instance in place.

    Parameters
    ----------
    instance : torch.Tensor, shape (m, T)
        Channels of a single instance, modified in place.
    spike_params : SpikeParams
        Ranges controlling the random spike amplitude, width, and count.

    Returns
    -------
    torch.Tensor
        Sorted ``LongTensor`` of spike-event centres added to the instance.
    """
    m, t_len = instance.shape
    amp_lo, amp_hi = spike_params.amplitude_range
    margin = spike_params.margin

    n_events = _randint(*spike_params.count_range)
    centres = torch.randint(margin, t_len - margin, (n_events,)).sort().values

    for centre in centres.tolist():
        width = _randint(*spike_params.width_range)
        idx = torch.arange(centre - width, centre + width + 1)
        valid = (idx >= 0) & (idx < t_len)
        idx = idx[valid]
        profile = 1.0 - (idx - centre).abs().to(torch.float32) / (width + 1)
        for channel in range(m):
            amplitude = amp_lo + torch.rand(1).item() * (amp_hi - amp_lo)
            instance[channel, idx] += amplitude * profile

    return centres
