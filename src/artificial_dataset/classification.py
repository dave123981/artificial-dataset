"""Classification dataset generator built from signal components."""

import math
from typing import Any

import torch

from artificial_dataset._components import compose, gaussian_noise

_DEFAULT_CLASS_PARAMS: list[dict[str, Any]] = [
    {
        "sinusoidal": {"amplitude": 1.0, "frequency": 1.0, "phase": 0.0},
        "linear": {"slope": 0.3, "intercept": 0.0},
    },
    {
        "sinusoidal": {"amplitude": 0.5, "frequency": 2.0, "phase": math.pi / 2},
        "linear": {"slope": -0.3, "intercept": 2.0},
        "polynomial": {"coefficients": [0.0, 0.0, 0.05]},
    },
]


def make_classification(
    n_samples: int = 1000,
    n_classes: int = 2,
    noise_std: float = 0.1,
    x_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    class_params: list[dict[str, Any]] | None = None,
    random_state: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a classification dataset from signal components.

    Each class is characterised by a unique combination of linear,
    polynomial, and sinusoidal components.  Samples are produced by
    evaluating the class signal at uniformly drawn *x* values and adding
    Gaussian noise.

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples across all classes.
    n_classes : int, optional
        Number of distinct classes.
    noise_std : float, optional
        Standard deviation of the additive Gaussian noise applied to every
        sample.
    x_range : tuple[float, float], optional
        Closed interval ``[min, max]`` from which input values are drawn
        uniformly.
    class_params : list[dict[str, Any]] | None, optional
        Per-class component configuration.  Each entry is a ``params`` dict
        understood by :func:`~artificial_dataset._components.compose`.
        Must have exactly *n_classes* entries when provided.  When *None*,
        a sensible default configuration is used for up to two classes;
        for more classes parameters are generated automatically.
    random_state : int | None, optional
        Seed passed to :func:`torch.manual_seed` for reproducibility.

    Returns
    -------
    X : torch.Tensor, shape (n_samples, 2)
        Feature matrix.  Column 0 contains the sampled *x* values; column 1
        contains the corresponding signal value (components + noise).
    y : torch.Tensor, shape (n_samples,)
        Integer class labels in ``[0, n_classes)``, dtype ``torch.long``.

    Raises
    ------
    ValueError
        If ``len(class_params) != n_classes``.

    Examples
    --------
    >>> X, y = make_classification(n_samples=200, n_classes=2, random_state=0)
    >>> X.shape, y.shape
    (torch.Size([200, 2]), torch.Size([200]))
    >>> y.unique().tolist()
    [0, 1]
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    if class_params is None:
        class_params = _build_default_class_params(n_classes)

    if len(class_params) != n_classes:
        raise ValueError(
            f"len(class_params)={len(class_params)} must equal n_classes={n_classes}"
        )

    x_min, x_max = x_range
    samples_per_class = _split_samples(n_samples, n_classes)

    x_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []

    for cls, (n, params) in enumerate(zip(samples_per_class, class_params)):
        x = torch.rand(n) * (x_max - x_min) + x_min
        signal = compose(x, params) + gaussian_noise((n,), mean=0.0, std=noise_std)
        x_parts.append(torch.stack([x, signal], dim=1))
        y_parts.append(torch.full((n,), cls, dtype=torch.long))

    return torch.cat(x_parts, dim=0), torch.cat(y_parts, dim=0)


def _split_samples(n_samples: int, n_classes: int) -> list[int]:
    base = n_samples // n_classes
    remainder = n_samples % n_classes
    return [base + (1 if i < remainder else 0) for i in range(n_classes)]


def _build_default_class_params(n_classes: int) -> list[dict[str, Any]]:
    if n_classes <= len(_DEFAULT_CLASS_PARAMS):
        return _DEFAULT_CLASS_PARAMS[:n_classes]
    params: list[dict[str, Any]] = list(_DEFAULT_CLASS_PARAMS)
    for c in range(len(_DEFAULT_CLASS_PARAMS), n_classes):
        params.append(
            {
                "linear": {
                    "slope": (c - n_classes / 2) * 0.4,
                    "intercept": float(c),
                },
                "sinusoidal": {
                    "amplitude": 1.0 + 0.3 * c,
                    "frequency": 1.0 + 0.5 * c,
                    "phase": c * math.pi / n_classes,
                },
            }
        )
    return params
