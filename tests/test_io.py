"""Unit tests for CSV serialization of SyntheticSeries instances."""

import csv
from pathlib import Path

import pytest

from artificial_dataset.injectors import add_level_shift, add_point_anomalies
from artificial_dataset.io import save_series
from artificial_dataset.series import SyntheticSeries, make_series


@pytest.fixture
def base_series() -> SyntheticSeries:
    """Fixture returning a standard, non-anomalous synthetic series."""
    return make_series(
        series_length=20,
        function_type="sinusoidal",
        function_params={"amplitude": 2.0, "frequency": 0.05},
        noise_std=0.0,
    )


def test_save_series_writes_header_row(
    base_series: SyntheticSeries, tmp_path: Path
) -> None:
    """The CSV file starts with the expected column header."""
    path = tmp_path / "series.csv"
    save_series(base_series, path)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    assert header == ["x", "y", "label", "anomaly_type"]


def test_save_series_row_count_matches_series_length(
    base_series: SyntheticSeries, tmp_path: Path
) -> None:
    """One data row is written per timestep, plus the header row."""
    path = tmp_path / "series.csv"
    save_series(base_series, path)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == len(base_series) + 1


def test_save_series_values_round_trip(
    base_series: SyntheticSeries, tmp_path: Path
) -> None:
    """Written x, y, and label columns match the source series numerically."""
    path = tmp_path / "series.csv"
    save_series(base_series, path)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = list(reader)

    x_written = [float(r[0]) for r in rows]
    y_written = [float(r[1]) for r in rows]
    label_written = [int(r[2]) for r in rows]

    assert x_written == pytest.approx(base_series.x.tolist())
    assert y_written == pytest.approx(base_series.y.tolist())
    assert label_written == base_series.label.tolist()


def test_save_series_anomaly_type_column_empty_when_no_anomalies(
    base_series: SyntheticSeries, tmp_path: Path
) -> None:
    """anomaly_type is written as an empty string for normal timesteps."""
    path = tmp_path / "series.csv"
    save_series(base_series, path)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    assert all(r[3] == "" for r in rows)
    assert all(r[2] == "0" for r in rows)


def test_save_series_writes_single_anomaly_tag(
    base_series: SyntheticSeries, tmp_path: Path
) -> None:
    """A single injected anomaly type is written verbatim in its column."""
    series = add_point_anomalies(base_series, n_anomalies=1, random_state=0)
    path = tmp_path / "series.csv"
    save_series(series, path)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    tags_written = [r[3] for r in rows]
    assert tags_written == series.anomaly_type
    assert tags_written.count("point") == 1


def test_save_series_writes_pipe_delimited_tag_for_overlapping_anomalies(
    base_series: SyntheticSeries, tmp_path: Path
) -> None:
    """Overlapping anomalies produce a pipe-joined tag string in the CSV."""
    series = add_point_anomalies(
        base_series, n_anomalies=3, avoid_existing=False, random_state=1
    )
    series = add_level_shift(series, start_idx=0, duration=len(series), random_state=1)
    path = tmp_path / "series.csv"
    save_series(series, path)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    tags_written = [r[3] for r in rows]
    assert tags_written == series.anomaly_type
    assert any("|" in tag for tag in tags_written)


def test_save_series_accepts_pathlike(
    base_series: SyntheticSeries, tmp_path: Path
) -> None:
    """save_series accepts a pathlib.Path, not just a str, for the destination."""
    path = tmp_path / "subdir_exists" / "series.csv"
    path.parent.mkdir()
    save_series(base_series, path)
    assert path.exists()


def test_save_series_missing_parent_directory_raises(
    base_series: SyntheticSeries, tmp_path: Path
) -> None:
    """Parent directories are not created automatically, per the docstring."""
    path = tmp_path / "does_not_exist" / "series.csv"
    with pytest.raises(FileNotFoundError):
        save_series(base_series, path)


def test_save_series_length_mismatch_raises(
    base_series: SyntheticSeries, tmp_path: Path
) -> None:
    """A malformed series whose anomaly_type length disagrees with x/y raises.

    save_series zips x, y, label, and anomaly_type with strict=True, so any
    length mismatch between them must surface as a ValueError rather than
    silently truncating the shorter sequence.
    """
    series = SyntheticSeries(
        x=base_series.x,
        y=base_series.y,
        is_anomaly=base_series.is_anomaly,
        anomaly_type=base_series.anomaly_type[:-1],
    )
    with pytest.raises(ValueError):
        save_series(series, tmp_path / "series.csv")


def test_save_series_empty_series(tmp_path: Path) -> None:
    """A single-timestep series is still written correctly (header + 1 row)."""
    series = make_series(series_length=1, function_type="constant")
    path = tmp_path / "series.csv"
    save_series(series, path)

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 2


def test_save_series_handles_gpu_or_grad_tensors_via_detach_cpu(
    base_series: SyntheticSeries, tmp_path: Path
) -> None:
    """x/y tensors requiring grad are detached before export, not raising."""
    series = SyntheticSeries(
        x=base_series.x.clone().requires_grad_(True),
        y=base_series.y.clone().requires_grad_(True),
        is_anomaly=base_series.is_anomaly,
        anomaly_type=base_series.anomaly_type,
    )
    path = tmp_path / "series.csv"
    save_series(series, path)
    assert path.exists()


def test_save_series_uses_utf8_encoding(
    base_series: SyntheticSeries, tmp_path: Path
) -> None:
    """The written file is valid UTF-8 (matches the encoding save_series declares)."""
    path = tmp_path / "series.csv"
    save_series(base_series, path)
    with open(path, encoding="utf-8") as f:
        f.read()  # raises UnicodeDecodeError if encoding is wrong
