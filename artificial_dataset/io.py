"""Serialization utilities for SyntheticSeries instances.

Uses the standard-library `csv` module rather than pandas, keeping the
package torch-native with no additional runtime dependency.
"""

from __future__ import annotations

import csv
import os

from artificial_dataset.series import SyntheticSeries


def save_series(series: SyntheticSeries, path: str | os.PathLike[str]) -> None:
    """
    Save a SyntheticSeries to a CSV file.

    Writes one row per timestep with columns: `x`, `y`, `label`,
    `anomaly_type`. `label` is the binary classification target (1 =
    anomalous, matching `series.label`); `anomaly_type` is the pipe-delimited
    tag string for that timestep (empty string when not anomalous).

    Parameters
    ----------
    series : SyntheticSeries
        The series to export.
    path : str or os.PathLike
        Destination file path. Parent directories are not created
        automatically.

    Returns
    -------
    None
    """
    x = series.x.detach().cpu().tolist()
    y = series.y.detach().cpu().tolist()
    label = series.label.detach().cpu().tolist()

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "anomaly_type"])
        for xi, yi, li, tag in zip(x, y, label, series.anomaly_type, strict=True):
            writer.writerow([xi, yi, li, tag])
