"""Artificial dataset generation using signal component primitives.

The package exposes two high-level generators:

* :func:`~artificial_dataset.classification.make_classification` - labelled
  multi-class data where each class follows a distinct signal shape.
* :func:`~artificial_dataset.anomaly.make_anomaly_dataset` - normal versus
  anomalous samples drawn from the same signal with different noise levels.

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
from artificial_dataset.anomaly import make_anomaly_dataset
from artificial_dataset.classification import make_classification

__all__ = [
    "compose",
    "gaussian_noise",
    "linear",
    "make_anomaly_dataset",
    "make_classification",
    "polynomial",
    "sinusoidal",
]
