"""Demonstration of add_trend_change: a base trend switching to another known trend."""

from artificial_dataset.injectors import add_trend_change
from artificial_dataset.series import make_series
from artificial_dataset.visualize import plot_series

# A sinusoidal baseline...
dataset = make_series(
    series_length=400,
    function_type="sinusoidal",
    function_params={"amplitude": 2.0, "frequency": 0.05},
    noise_std=0.1,
    random_state=0,
)

# ...that drifts into a linear trend for one segment (continuity=True keeps the
# level from jumping at the boundary, so only the shape/slope changes)...
dataset = add_trend_change(
    dataset,
    start_idx=150,
    new_function_type="linear",
    new_function_params={"slope": 0.08, "intercept": 0.0},
    duration=60,
)

plot_series(
    dataset, title="Demonstration: Sinusoidal baseline with trend-change anomalies"
)
