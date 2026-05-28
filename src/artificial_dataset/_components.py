"""Primitive signal components for artificial dataset generation."""

from typing import Any

import torch


def gaussian_noise(size: tuple[int, ...], mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Generate a tensor of Gaussian noise.

    Parameters
    ----------
    size : tuple[int, ...]
        Shape of the output tensor.
    mean : float, optional
        Mean of the Gaussian distribution.
    std : float, optional
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    torch.Tensor
        Noise tensor with the requested shape.
    """
    return torch.normal(mean=mean, std=std, size=size)


def linear(x: torch.Tensor, slope: float = 1.0, intercept: float = 0.0) -> torch.Tensor:
    """Compute a linear signal: ``y = slope * x + intercept``.

    Parameters
    ----------
    x : torch.Tensor
        Input values.
    slope : float, optional
        Slope coefficient.
    intercept : float, optional
        Intercept (bias) term.

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as *x*.
    """
    return slope * x + intercept


def polynomial(x: torch.Tensor, coefficients: list[float]) -> torch.Tensor:
    """Compute a polynomial signal: ``y = sum(c_i * x^i)``.

    Parameters
    ----------
    x : torch.Tensor
        Input values.
    coefficients : list[float]
        Coefficients ``[c_0, c_1, ..., c_n]`` where ``c_i`` multiplies ``x**i``.

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as *x*.
    """
    result = torch.zeros_like(x)
    for i, c in enumerate(coefficients):
        result = result + c * x.pow(i)
    return result


def sinusoidal(
    x: torch.Tensor,
    amplitude: float = 1.0,
    frequency: float = 1.0,
    phase: float = 0.0,
) -> torch.Tensor:
    """Compute a sinusoidal signal: ``y = amplitude * sin(frequency * x + phase)``.

    Parameters
    ----------
    x : torch.Tensor
        Input values (in radians when using default frequency).
    amplitude : float, optional
        Peak amplitude.
    frequency : float, optional
        Angular frequency in rad/unit.
    phase : float, optional
        Phase offset in radians.

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as *x*.
    """
    return amplitude * torch.sin(frequency * x + phase)


def compose(x: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    """Evaluate a superposition of signal components at *x*.

    Recognised keys in *params* and their expected value types:

    * ``"linear"``      – ``dict`` of keyword arguments for :func:`linear`
    * ``"polynomial"``  – ``dict`` of keyword arguments for :func:`polynomial`
    * ``"sinusoidal"``  – ``dict`` of keyword arguments for :func:`sinusoidal`

    Unknown keys are silently ignored so callers can attach metadata to the
    same dict without interfering with signal generation.

    Parameters
    ----------
    x : torch.Tensor
        Input values, shape ``(n,)``.
    params : dict[str, Any]
        Component specifications as described above.

    Returns
    -------
    torch.Tensor
        Superposed signal of the same shape as *x*.
    """
    signal = torch.zeros_like(x)
    if "linear" in params:
        signal = signal + linear(x, **params["linear"])
    if "polynomial" in params:
        signal = signal + polynomial(x, **params["polynomial"])
    if "sinusoidal" in params:
        signal = signal + sinusoidal(x, **params["sinusoidal"])
    return signal
