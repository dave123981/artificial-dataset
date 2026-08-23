"""Use demonstration of the visualization of datasets."""

from artificial_dataset.injectors import add_level_shift
from artificial_dataset.io import save_series
from artificial_dataset.series import make_series
from artificial_dataset.visualize import plot_series

dataset = make_series(
    500,
    "Sinusodial trend with anomalies",
    {"amplitude": 2.0, "frequency": 0.05},
    noise_std=0.1,
)
dataset = add_level_shift(dataset, start_idx=200, shift_magnitude=4.0, duration=40)

plot_series(dataset)  # draws the signal with anomalies marked
save_series(dataset, "dataset.csv")
