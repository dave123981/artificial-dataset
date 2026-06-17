"""Artificial dataset generation using signal component primitives.

The package exposes two high-level generators:

* :func:`~artificial_dataset.classification.make_classification` - labelled
  multi-class data where each class follows a distinct signal shape.
* :func:`~artificial_dataset.anomaly.make_anomaly_dataset` - a single
  multivariate time series with positive spike anomalies added to a smooth
  baseline.

Both return ``torch.Tensor`` objects so the results integrate directly with
PyTorch training loops.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("artificial-dataset")
except PackageNotFoundError:
    __version__ = "unknown"

from artificial_dataset._components import (
    compose,
    gaussian_noise,
    linear,
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
from artificial_dataset.metrics import ClassifierMetrics

__all__ = [
    "AnomalyDataset",
    "AnomalySplits",
    "ClassifierMetrics",
    "SpikeParams",
    "compose",
    "gaussian_noise",
    "linear",
    "make_anomaly_dataset",
    "make_classification",
    "polynomial",
    "sinusoidal",
]
