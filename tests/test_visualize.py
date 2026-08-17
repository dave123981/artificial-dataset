"""Unit tests for plotting utilities."""

from collections.abc import Generator

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.figure import Figure

matplotlib.use("Agg")

from artificial_dataset.injectors import add_level_shift, add_point_anomalies
from artificial_dataset.series import SyntheticSeries, make_series
from artificial_dataset.visualize import _anomaly_spans, plot_series


@pytest.fixture
def base_series() -> SyntheticSeries:
    """Fixture returning a standard, non-anomalous synthetic series."""
    return make_series(
        series_length=100,
        function_type="sinusoidal",
        function_params={"amplitude": 2.0, "frequency": 0.05},
        noise_std=0.05,
        random_state=0,
    )


@pytest.fixture(autouse=True)
def _close_figures() -> Generator[None, None, None]:
    """Close any figures left open by a test to avoid matplotlib state leaks."""
    yield
    plt.close("all")


# ---------- _anomaly_spans ----------


def test_anomaly_spans_empty_mask_returns_no_spans() -> None:
    """An all-False mask yields an empty span list."""
    mask = torch.zeros(10, dtype=torch.bool)
    assert _anomaly_spans(mask) == []


def test_anomaly_spans_single_point() -> None:
    """A single True index yields one (start, end) span equal to itself."""
    mask = torch.zeros(10, dtype=torch.bool)
    mask[4] = True
    assert _anomaly_spans(mask) == [(4, 4)]


def test_anomaly_spans_single_contiguous_run() -> None:
    """A contiguous run of True values collapses into one span."""
    mask = torch.zeros(10, dtype=torch.bool)
    mask[2:6] = True
    assert _anomaly_spans(mask) == [(2, 5)]


def test_anomaly_spans_multiple_disjoint_runs() -> None:
    """Non-adjacent True runs are returned as separate spans, in order."""
    mask = torch.zeros(20, dtype=torch.bool)
    mask[1:3] = True
    mask[10] = True
    mask[15:18] = True
    assert _anomaly_spans(mask) == [(1, 2), (10, 10), (15, 17)]


def test_anomaly_spans_all_true() -> None:
    """A fully anomalous mask yields a single span covering the whole range."""
    mask = torch.ones(6, dtype=torch.bool)
    assert _anomaly_spans(mask) == [(0, 5)]


def test_anomaly_spans_adjacent_indices_merge() -> None:
    """Two runs separated by a single gap stay separate; touching ones merge."""
    mask = torch.zeros(10, dtype=torch.bool)
    mask[0] = True
    mask[1] = True  # adjacent to index 0 -> should merge into one span
    mask[3] = True  # gap at index 2 -> separate span
    assert _anomaly_spans(mask) == [(0, 1), (3, 3)]


# ---------- plot_series ----------


def test_plot_series_returns_figure(base_series: SyntheticSeries) -> None:
    """plot_series returns a matplotlib Figure instance."""
    fig = plot_series(base_series, show=False)
    assert isinstance(fig, Figure)


def test_plot_series_no_anomalies_has_no_scatter(base_series: SyntheticSeries) -> None:
    """With no anomalies, no scatter collection or shaded span is drawn."""
    fig = plot_series(base_series, show=False)
    ax = fig.axes[0]
    assert len(ax.collections) == 0


def test_plot_series_with_point_anomalies_adds_scatter(
    base_series: SyntheticSeries,
) -> None:
    """Injected point anomalies are drawn as a scatter collection."""
    series = add_point_anomalies(base_series, n_anomalies=3, random_state=0)
    fig = plot_series(series, show=False)
    ax = fig.axes[0]
    assert len(ax.collections) >= 1


def test_plot_series_with_contiguous_anomaly_adds_shaded_span(
    base_series: SyntheticSeries,
) -> None:
    """A contiguous anomaly (level shift) adds an axvspan patch next to the scatter."""
    series = add_level_shift(base_series, start_idx=10, duration=15, random_state=0)
    fig = plot_series(series, show=False)
    ax = fig.axes[0]
    assert len(ax.patches) >= 1


def test_plot_series_default_title_uses_function_type(
    base_series: SyntheticSeries,
) -> None:
    """When no title is given, the axes title falls back to meta['function_type']."""
    fig = plot_series(base_series, show=False)
    ax = fig.axes[0]
    assert ax.get_title() == base_series.meta["function_type"]


def test_plot_series_explicit_title_overrides_default(
    base_series: SyntheticSeries,
) -> None:
    """An explicit title argument takes precedence over series.meta."""
    fig = plot_series(base_series, title="Custom Title", show=False)
    ax = fig.axes[0]
    assert ax.get_title() == "Custom Title"


def test_plot_series_missing_function_type_falls_back_to_default_string() -> None:
    """When meta has no 'function_type' key, the hardcoded fallback title is used."""
    series = make_series(series_length=20, function_type="constant")
    series.meta.pop("function_type", None)
    fig = plot_series(series, show=False)
    ax = fig.axes[0]
    assert ax.get_title() == "Synthetic series"


def test_plot_series_creates_new_figure_when_ax_omitted(
    base_series: SyntheticSeries,
) -> None:
    """Without an ax argument, plot_series creates its own figure."""
    n_figures_before = len(plt.get_fignums())
    fig = plot_series(base_series, show=False)
    assert len(plt.get_fignums()) == n_figures_before + 1
    assert fig.number in plt.get_fignums()


def test_plot_series_draws_into_provided_axes(base_series: SyntheticSeries) -> None:
    """When ax is provided, plot_series draws into it and returns its parent figure."""
    fig, ax = plt.subplots()
    returned_fig = plot_series(base_series, ax=ax, show=False)
    assert returned_fig is fig
    assert len(ax.lines) == 1


def test_plot_series_line_data_matches_series_values(
    base_series: SyntheticSeries,
) -> None:
    """The plotted line's x/y data matches the series' x/y tensors."""
    fig = plot_series(base_series, show=False)
    ax = fig.axes[0]
    line = ax.lines[0]
    x_plotted, y_plotted = line.get_data()

    assert x_plotted == pytest.approx(base_series.x.detach().cpu().numpy())
    assert y_plotted == pytest.approx(base_series.y.detach().cpu().numpy())


def test_plot_series_sets_axis_labels(base_series: SyntheticSeries) -> None:
    """X and y axis labels are set to the expected fixed strings."""
    fig = plot_series(base_series, show=False)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "time (a.u.)"
    assert ax.get_ylabel() == "value (a.u.)"


def test_plot_series_show_false_does_not_raise(base_series: SyntheticSeries) -> None:
    """show=False skips plt.show() and returns normally under a headless backend."""
    fig = plot_series(base_series, show=False)
    assert fig is not None


def test_plot_series_show_true_calls_plt_show(
    base_series: SyntheticSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    """show=True triggers a call to plt.show()."""
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(True))
    plot_series(base_series, show=True)
    assert calls == [True]
