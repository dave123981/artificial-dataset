"""Artificial dataset generation using signal component primitives.

The package exposes two high-level generators:

* :func:`~artificial_dataset.classification.make_classification` - labelled
  multi-class data where each class follows a distinct signal shape.
* :func:`~artificial_dataset.anomaly.make_anomaly_dataset` - a single
  multivariate time series with positive spike anomalies added to a smooth
  baseline.
* :func:`~artificial_dataset.series.make_series` plus the composable
  injectors in :mod:`~artificial_dataset.injectors` (``add_point_anomalies``,
  ``add_level_shift``, ``add_dropout``, and others) - build a clean
  univariate series, then stack one or more labeled anomaly types onto it.

All return ``torch.Tensor``-backed objects so the results integrate directly
with PyTorch training loops.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("artificial-dataset")
except PackageNotFoundError:
    __version__ = "unknown"

from artificial_dataset._components import (
    compose,
    compose_weighted,
    constant,
    exponential,
    gaussian_noise,
    linear,
    logarithmic,
    periodic_seasonal,
    polynomial,
    sinusoidal,
)
from artificial_dataset.anomaly import (
    AnomalyDataset,
    AnomalySplits,
    SpikeParams,
    make_anomaly_dataset,
)
from artificial_dataset.classification import make_classification
from artificial_dataset.injectors import (
    add_collective_anomaly,
    add_dropout,
    add_level_shift,
    add_point_anomalies,
    add_seasonal_distortion,
    add_spike_anomalies,
    add_trend_change,
    add_variance_change,
    anomaly_summary,
)
from artificial_dataset.io import save_series
from artificial_dataset.metrics import ClassifierMetrics
from artificial_dataset.series import (
    SyntheticSeries,
    SyntheticSeriesSplits,
    make_series,
)
from artificial_dataset.visualize import plot_series

__all__ = [
    "AnomalyDataset",
    "AnomalySplits",
    "ClassifierMetrics",
    "SpikeParams",
    "SyntheticSeries",
    "SyntheticSeriesSplits",
    "add_collective_anomaly",
    "add_dropout",
    "add_level_shift",
    "add_point_anomalies",
    "add_seasonal_distortion",
    "add_spike_anomalies",
    "add_trend_change",
    "add_variance_change",
    "anomaly_summary",
    "compose",
    "compose_weighted",
    "constant",
    "exponential",
    "gaussian_noise",
    "linear",
    "logarithmic",
    "make_anomaly_dataset",
    "make_classification",
    "make_series",
    "periodic_seasonal",
    "plot_series",
    "polynomial",
    "save_series",
    "sinusoidal",
]
