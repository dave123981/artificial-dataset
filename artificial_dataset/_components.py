"""Primitive signal components for artificial dataset generation."""

from typing import Any

import torch


def gaussian_noise(
    size: tuple[int, ...], mean: float = 0.0, std: float = 1.0
) -> torch.Tensor:
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


# ---------- o ----------
# Basic trend functions
def constant(x: torch.Tensor, value: float = 1.0) -> torch.Tensor:
    """Compute a constant signal: ``y = value``.

    Parameters
    ----------
    x : torch.Tensor
        Input values, used only to determine the output shape.
    value : float, optional
        Constant value.

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as *x*, filled with *value*.
    """
    return torch.full_like(x, fill_value=value, dtype=torch.float32)


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


def exponential(
    x: torch.Tensor,
    initial_value: float = 1.0,
    growth_rate: float = 0.05,
) -> torch.Tensor:
    """Compute an exponential signal: ``y = initial_value * exp(growth_rate * x)``.

    Parameters
    ----------
    x : torch.Tensor
        Input values.
    initial_value : float, optional
        Value at ``x = 0``.
    growth_rate : float, optional
        Exponential growth (positive) or decay (negative) rate.

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as *x*.
    """
    return initial_value * torch.exp(growth_rate * x)


def logarithmic(
    x: torch.Tensor,
    scale: float = 1.0,
    shift: float = 1.0,
) -> torch.Tensor:
    """Compute a logarithmic signal: ``y = scale * log(x + shift)``.

    Parameters
    ----------
    x : torch.Tensor
        Input values.
    scale : float, optional
        Multiplicative scale applied to the logarithm.
    shift : float, optional
        Additive shift applied to *x* before taking the log, keeping the
        argument positive when *x* starts at ``0``.

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as *x*.

    Raises
    ------
    ValueError
        If *shift* is not strictly positive.
    """
    if shift <= 0:
        raise ValueError("shift must be > 0 to keep log(x + shift) defined at x=0.")
    return scale * torch.log(x + shift)


def periodic_seasonal(
    x: torch.Tensor,
    period: float = 10.0,
    amplitude: float = 1.0,
    offset: float = 0.0,
    waveform: str = "sine",
) -> torch.Tensor:
    """Compute a repeating pattern with an explicit period, in samples.

    Unlike :func:`sinusoidal`, which is parameterised by frequency, this is
    parameterised by *period* and supports a few common waveform shapes.

    Parameters
    ----------
    x : torch.Tensor
        Input values.
    period : float, optional
        Period of the pattern, in the same units as *x*.
    amplitude : float, optional
        Peak amplitude.
    offset : float, optional
        Constant vertical offset.
    waveform : str, optional
        One of ``"sine"``, ``"square"``, ``"triangle"``, ``"sawtooth"``.

    Returns
    -------
    torch.Tensor
        Output tensor of the same shape as *x*.

    Raises
    ------
    ValueError
        If *period* is not strictly positive, or *waveform* is unrecognised.
    """
    if period <= 0:
        raise ValueError("period must be > 0.")

    phase_fraction = (x % period) / period  # in [0, 1)

    if waveform == "sine":
        values = torch.sin(2 * torch.pi * phase_fraction)
    elif waveform == "square":
        values = torch.sign(torch.sin(2 * torch.pi * phase_fraction))
    elif waveform == "triangle":
        values = 2 * torch.abs(2 * (phase_fraction - torch.floor(phase_fraction + 0.5))) - 1
    elif waveform == "sawtooth":
        values = 2 * phase_fraction - 1
    else:
        raise ValueError(
            f"Unknown waveform '{waveform}'. Choose from: sine, square, triangle, sawtooth."
        )

    return amplitude * values + offset


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


# ---------- o ----------
# Combination of basic functions
def compose(x: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    """Evaluate a superposition of signal components at *x*.

    Recognised keys in *params* and their expected value types:

    * ``"constant"``          - ``dict`` of keyword arguments for :func:`constant`
    * ``"linear"``            - ``dict`` of keyword arguments for :func:`linear`
    * ``"exponential"``       - ``dict`` of keyword arguments for :func:`exponential`
    * ``"logarithmic"``       - ``dict`` of keyword arguments for :func:`logarithmic`
    * ``"periodic_seasonal"`` - ``dict`` of keyword arguments for :func:`periodic_seasonal`
    * ``"polynomial"``        - ``dict`` of keyword arguments for :func:`polynomial`
    * ``"sinusoidal"``        - ``dict`` of keyword arguments for :func:`sinusoidal`

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
    if "constant" in params:
        signal = signal + constant(x, **params["constant"])
    if "linear" in params:
        signal = signal + linear(x, **params["linear"])
    if "exponential" in params:
        signal = signal + exponential(x, **params["exponential"])
    if "logarithmic" in params:
        signal = signal + logarithmic(x, **params["logarithmic"])
    if "periodic_seasonal" in params:
        signal = signal + periodic_seasonal(x, **params["periodic_seasonal"])
    if "polynomial" in params:
        signal = signal + polynomial(x, **params["polynomial"])
    if "sinusoidal" in params:
        signal = signal + sinusoidal(x, **params["sinusoidal"])
    return signal


def compose_weighted(x: torch.Tensor, components: list[dict[str, Any]]) -> torch.Tensor:
    """Evaluate a weighted sum of several composed signals at *x*.

    Unlike :func:`compose`, which sums at most one instance of each
    component type (dict keys must be unique), *compose_weighted* accepts a
    list, so the same component type can appear more than once — e.g. two
    sinusoids at different frequencies — each contributing with its own
    weight.

    Parameters
    ----------
    x : torch.Tensor
        Input values, shape ``(n,)``.
    components : list[dict[str, Any]]
        One entry per term in the sum. Each entry is a ``params`` dict as
        understood by :func:`compose` (e.g. ``{"sinusoidal": {...}}``), plus
        an optional ``"weight"`` key (default ``1.0``) scaling that term.

    Returns
    -------
    torch.Tensor
        Superposed signal of the same shape as *x*.

    Raises
    ------
    ValueError
        If *components* is empty.

    Examples
    --------
    >>> import torch
    >>> x = torch.linspace(0, 10, steps=50)
    >>> y = compose_weighted(x, [
    ...     {"linear": {"slope": 0.2}},
    ...     {"sinusoidal": {"amplitude": 1.0, "frequency": 0.5}, "weight": 0.5},
    ... ])
    >>> y.shape
    torch.Size([50])
    """
    if not components:
        raise ValueError("components must contain at least one entry.")

    total = torch.zeros_like(x)
    for entry in components:
        weight = entry.get("weight", 1.0)
        params = {key: value for key, value in entry.items() if key != "weight"}
        total = total + weight * compose(x, params)
    return total