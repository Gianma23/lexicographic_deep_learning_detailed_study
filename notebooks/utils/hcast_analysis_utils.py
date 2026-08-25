from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import colors as mcolors

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
try:
    from notebooks.utils.multiseed_utils import (
        aggregate_parsed_seed_runs,
        discover_seed_dirs,
        has_seed_runs,
        metric_series_with_std,
        normalized_config_for_seed_comparison,
        sample_stats,
        seed_value_from_dir,
    )
except ModuleNotFoundError:
    from multiseed_utils import (
        aggregate_parsed_seed_runs,
        discover_seed_dirs,
        has_seed_runs,
        metric_series_with_std,
        normalized_config_for_seed_comparison,
        sample_stats,
        seed_value_from_dir,
    )


load_dotenv(
    Path(os.environ.get("PROJECT_ENV_FILE", REPO_ROOT / ".env")).expanduser(),
    override=False,
)

try:
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover - notebook runtime dependent
    Markdown = None
    display = None


try:
    from notebooks.utils.thesis_style import (
        DATASET_SLUGS,
        PANEL_HEIGHT_2COL_IN,
        PANEL_HEIGHT_IN,
        TEXT_WIDTH_IN,
        dataset_display_name,
        dataset_slug,
        grid_page_rows,
        legend_height_in,
        level_display_name,
        level_index_symbol,
        level_loss_symbol,
        loss_display_name,
        save_figure,
        use_paper_style,
    )
except ModuleNotFoundError:
    from thesis_style import (
        DATASET_SLUGS,
        PANEL_HEIGHT_2COL_IN,
        PANEL_HEIGHT_IN,
        TEXT_WIDTH_IN,
        dataset_display_name,
        dataset_slug,
        grid_page_rows,
        legend_height_in,
        level_display_name,
        level_index_symbol,
        level_loss_symbol,
        loss_display_name,
        save_figure,
        use_paper_style,
    )


use_paper_style()

# Thesis mode drops the in-figure suptitle, because in the document the title
# belongs to the LaTeX caption. Set to False in a notebook to get the titles back
# for on-screen reading; nothing else changes.
THESIS_STYLE = True

# Thesis mode takes the title off the canvas, which leaves an unlabelled figure
# on screen: a notebook that renders three datasets times four sections shows a
# dozen panels with nothing saying which dataset or which comparison any of them
# is. The title is therefore written above the figure as notebook text instead
# of onto it, so it is there while reading and still absent from the exported
# PDF the document includes. Set to False for bare figures.
SHOW_FIGURE_CAPTIONS = True

# Where figures are written. Notebooks set this to a directory under
# /scratch/g.saggini1/outputs; None renders without saving.
FIGURE_DIR: Optional[Path] = None

# Every figure saved since the last reset, as (stem, width_in, height_in). The
# notebooks print this at the end to confirm nothing drifted off the page.
EXPORTED_FIGURES: List[Tuple[str, float, float]] = []

# Ordered purple ramp for hierarchy levels, shared with datasets_analysis.ipynb.
# Colour means "run" almost everywhere in these notebooks, so the level ramp is
# used only inside panels where a single run owns the whole panel.
LEVEL_RAMP = ("#CBC0DD", "#8C6BB1", "#4D004B")

# Line weights. A panel here is half the text width and routinely carries six to
# ten curves, so the curves are drawn thin: at 1.3 pt the strokes of neighbouring
# runs touch and the colour of each becomes hard to read. Anything that is not a
# run curve (a rule, a raw overlay) sits below these.
CURVE_LINEWIDTH = 0.95
CURVE_LINEWIDTH_SECONDARY = 0.9  # the "before" trace in a before/after pair
RAW_OVERLAY_LINEWIDTH = 0.6

# Dash pattern for the "before" trace of a before/after pair. A plain dotted
# line does not survive at this size: matplotlib's ":" is a dot as long as the
# stroke is wide with a gap 1.65 times that, so at 0.9 pt it prints as about a
# third ink and the colour of the curve stops being readable - and the legend
# sample, only a fifth of an inch wide, becomes a row of specks. This pattern
# is two thirds ink, so the curve keeps its colour and still reads as broken.
# The numbers are in points before matplotlib scales them by the linewidth.
BEFORE_TRACE_DASHES = (0, (2.6, 1.3))

# Decoding mode reads through the line style, colour still means "run". Solid is
# independent decoding, dashed is top-down: the solid curve is the one the level
# heads actually produce, and the dashed one is what the top-down decoder makes
# of it, so the derived quantity gets the broken line.
MODE_LINESTYLES = {"independent": "-", "topdown": "--"}

# Which decoders a notebook wants to see. Drawing both is the default because
# the two decoders do not in general select the same epoch and a comparison that
# silently mixes them is wrong; but a section being read for one decoder alone
# is half as many curves, so it is worth being able to ask for that.
#
# A notebook sets this once - `hu.set_decoder_view('independent')` - and every
# section follows: the curves lose the other line style, the tables lose the
# rows belonging to the other decoder, and the diagnostic panels lose the keys
# that exist only under it. Individual calls can still override it with their
# own `decoder_view=` argument.
DECODER_VIEWS: Tuple[str, ...] = ("both", "independent", "topdown")
DECODER_VIEW: str = "both"

_DECODER_VIEW_ALIASES = {
    "both": "both",
    "all": "both",
    "topdownandindependent": "both",
    "independentandtopdown": "both",
    "independent": "independent",
    "ind": "independent",
    "topdown": "topdown",
    "td": "topdown",
}

_DECODER_VIEW_PHRASES = {
    "both": "top-down and independent",
    "independent": "independent",
    "topdown": "top-down",
}


def _normalize_decoder_view(view: Optional[Union[str, bool]] = None) -> str:
    """One of ``DECODER_VIEWS``; ``None`` means the module-level setting.

    ``True``/``False`` are accepted so the older `include_topdown=` spelling
    keeps working: it said "both" or "independent" and nothing else.
    """
    if view is None:
        view = DECODER_VIEW
    if isinstance(view, bool):
        return "both" if view else "independent"
    key = str(view).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key not in _DECODER_VIEW_ALIASES:
        raise ValueError(
            f"Unknown decoder view {view!r}; expected one of {DECODER_VIEWS}."
        )
    return _DECODER_VIEW_ALIASES[key]


def set_decoder_view(view: Union[str, bool]) -> str:
    """Set the decoders every figure and table in this module shows.

    ``'both'`` (default), ``'independent'`` or ``'topdown'``. Returns the
    normalized value so a notebook can print what it set.
    """
    global DECODER_VIEW
    DECODER_VIEW = _normalize_decoder_view(view)
    return DECODER_VIEW


def decoder_modes(view: Optional[Union[str, bool]] = None) -> Tuple[str, ...]:
    """The decoder keys a view selects, in the order figures draw them."""
    resolved = _normalize_decoder_view(view)
    if resolved == "both":
        return ("independent", "topdown")
    return (resolved,)


def decoder_view_phrase(view: Optional[Union[str, bool]] = None) -> str:
    """How a caption names the view: 'top-down and independent', or one of them."""
    return _DECODER_VIEW_PHRASES[_normalize_decoder_view(view)]


# One output-space diagnostic section for every family, so a reader can put the
# H-CAST page next to the HRN page and compare panel for panel. The list is the
# union of what any family logs, ordered as the mechanism reads: did it fire,
# how large was the constraint violation, how far did it move the scores, and
# what did that do to the fine decision and to the hierarchy around it.
#
# A union is safe because `plot_projection_diagnostics` drops a spec that no
# selected run logs and a spec that every selected run pins at exactly zero, so
# each family renders only the panels its mechanism actually moved. That makes
# the panel count itself informative: a family whose mechanism never fired keeps
# the trailing structural panels and loses the projection ones.
MECHANISM_DIAGNOSTIC_SPECS: Tuple[Tuple[str, str, bool, str], ...] = (
    ("proj_constraint_alpha", "Constraint activation flag", False, "val"),
    ("proj_temperature", "Projection temperature", False, "val"),
    ("proj_logit_residual_before_l1", "Constraint residual before projection", False, "val"),
    ("proj_logit_residual_after_l1", "Constraint residual after projection", False, "val"),
    ("proj_logit_residual_reduction", "Constraint residual reduction", False, "val"),
    ("proj_logit_delta_l1_level_2", "Total fine logit shift", False, "val"),
    ("proj_gt_logit_delta_level_2", "True-class fine logit shift", False, "val"),
    ("proj_delta_l1_level_2", "Total fine probability shift", False, "val"),
    ("proj_gt_prob_delta_level_2", "True-class fine probability shift", False, "val"),
    ("proj_flip_rate_level_2", "Fine prediction flip rate", True, "val"),
    ("gt_parent_mass_pre_l2", "Fine mass under the true parent, before", True, "val"),
    ("gt_parent_mass_post_l2", "Fine mass under the true parent, after", True, "val"),
    ("gt_child_rank_within_parent_pre_l2", "True-class rank among siblings, before", False, "val"),
    ("gt_child_rank_within_parent_post_l2", "True-class rank among siblings, after", False, "val"),
    # `support_l1_*_correct` is not listed: it is exactly the middle-level
    # accuracy, which already has its own panel in the per-level figure.
    ("acc_l2_ind_given_l1_correct", "Fine accuracy given a correct parent (independent)", True, "val"),
    ("acc_l2_td_given_l1_correct", "Fine accuracy given a correct parent (top-down)", True, "val"),
)


def set_figure_dir(figure_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    """Point every plot in this module at one export directory."""
    global FIGURE_DIR
    FIGURE_DIR = None if figure_dir is None else Path(figure_dir)
    EXPORTED_FIGURES.clear()
    return FIGURE_DIR


def _export(fig, stem: str) -> None:
    """Save one figure at its authored size and record the geometry."""
    EXPORTED_FIGURES.append(save_figure(fig, FIGURE_DIR, stem))


def _caption_text(title: str, stem: str) -> str:
    """The on-screen caption: the title, plus which page it is when one was split."""
    match = re.search(r"part(\d+)$", str(stem))
    return f"{title} (part {match.group(1)})" if match else str(title)


def _finish(fig, stem: str, suptitle: Optional[str] = None) -> None:
    """Title, export, show.

    Outside thesis mode the title is drawn on the canvas. In thesis mode it is
    not - the document gets it from the LaTeX caption - so it is displayed as a
    line of notebook text immediately above the figure instead, which is where a
    reader looks for it and which the exported PDF does not carry.
    """
    if suptitle:
        if not THESIS_STYLE:
            fig.suptitle(suptitle)
        elif SHOW_FIGURE_CAPTIONS:
            show_markdown(f"**{_caption_text(suptitle, stem)}**")
    # A hair more side padding than the constrained-layout default. The engine
    # occasionally lands a rotated y-axis label a point or two past the canvas
    # (six-digit tick labels next to "Value (%)" is the case seen here), and a
    # clipped first glyph in print is not worth the 0.03 in this costs.
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(w_pad=0.07)
    _export(fig, stem)
    plt.show()


def _panel_columns(n_panels: int) -> int:
    """Panels per row: two, the way most papers lay a figure grid out.

    Two columns is what buys the height - a full-width row has to stay short to
    fit four of them on a page, while a half-width one can be tall enough to
    read. A lone panel keeps the full width, since half a row of white space
    beside it looks like a missing figure.
    """
    return 1 if int(n_panels) <= 1 else 2


def _legend_layout(labels: Sequence[str]) -> Tuple[int, int]:
    """Columns and rows for a bottom legend that fits the printed text width.

    Run labels here are long ("H-CAST lex coarse-first (independent)"), so the
    column count has to come from the widest label rather than from the entry
    count: at 7 pt a character is about 0.049 in and each entry also carries a
    line sample and its padding.
    """
    labels = [str(label) for label in labels]
    if not labels:
        return (1, 0)
    # Line sample plus its padding, measured off the printed legend.
    handle_width = 0.55
    widest = max(_text_width_in(label, plt.rcParams["legend.fontsize"]) for label in labels)
    entry_width = widest + handle_width
    ncol = max(1, min(len(labels), int(TEXT_WIDTH_IN // entry_width)))
    return (ncol, int(np.ceil(len(labels) / ncol)))


def _add_bottom_legend(fig, handles, labels) -> int:
    """One legend under the whole figure. Returns the number of rows it uses."""
    if not handles:
        return 0
    ncol, nrow = _legend_layout(labels)
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False)
    return nrow


def _grid_pages(
    n_panels: int,
    ncols: int,
    legend_labels: Sequence[str],
    panel_height: Optional[float] = None,
) -> Iterable[Tuple[Any, np.ndarray, int, int]]:
    """Yield ``(fig, axes, start, stop)`` pages of at most one text-block height.

    The grid is authored at the printed text width, two panels to a row; a grid
    that would exceed ``PAGE_HEIGHT_IN`` is split across consecutive figures
    rather than compressed, so the panels keep the same size on every page. The
    legend is measured from its labels first, so the space reserved for it is
    the space it actually takes.
    """
    ncols = max(1, int(ncols))
    if panel_height is None:
        panel_height = PANEL_HEIGHT_IN if ncols == 1 else PANEL_HEIGHT_2COL_IN
    _legend_ncol, legend_rows = _legend_layout(legend_labels)
    rows_per_page = grid_page_rows(ncols, legend_rows, panel_height)

    # Balance the split instead of filling each page to the brim: nine panels at
    # two columns is five rows, and 4 + 1 leaves an orphan row on its own page
    # where 3 + 2 does not.
    total_rows = int(np.ceil(n_panels / ncols))
    if total_rows > rows_per_page:
        n_pages = int(np.ceil(total_rows / rows_per_page))
        rows_per_page = int(np.ceil(total_rows / n_pages))

    start = 0
    while start < n_panels:
        stop = min(n_panels, start + rows_per_page * ncols)
        nrows = int(np.ceil((stop - start) / ncols))
        height = nrows * panel_height + legend_height_in(legend_rows)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(TEXT_WIDTH_IN, height), sharex=True, squeeze=False
        )
        yield fig, axes.reshape(-1), start, stop
        start = stop


# Widths are measured on a throwaway Agg canvas rather than estimated from a
# character count. A count is wrong in both directions here: DejaVu Serif is a
# wide face, so a title that "fits" by count overruns the panel, and a mathtext
# run like "$P_{123}$" counts nine characters but prints about three.
_MEASURE_FIGURE = None


def _text_width_in(text: str, fontsize) -> float:
    """Printed width of a string at a given point size, in inches."""
    global _MEASURE_FIGURE
    if _MEASURE_FIGURE is None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        _MEASURE_FIGURE = Figure(figsize=(1.0, 1.0), dpi=100)
        FigureCanvasAgg(_MEASURE_FIGURE)
    artist = _MEASURE_FIGURE.text(0.0, 0.0, str(text), fontsize=fontsize)
    try:
        extent = artist.get_window_extent(renderer=_MEASURE_FIGURE.canvas.get_renderer())
    finally:
        artist.remove()
    return float(extent.width) / float(_MEASURE_FIGURE.dpi)


def _wrap_title(title: str, ncols: int, n_lines: int = 2) -> str:
    """Wrap a panel title to the width of its panel.

    A title wider than its panel is clipped at the figure edge on an outer
    column and overruns its neighbour on an inner one; a second line is better
    than either. The budget is the panel width less the gutter that the y-axis
    label and tick numbers take, since the title is centred on the axes and not
    on the panel.
    """
    panel_width = TEXT_WIDTH_IN / max(1, int(ncols))
    budget = max(1.0, panel_width - 0.45)
    size = plt.rcParams["axes.titlesize"]
    if _text_width_in(title, size) <= budget:
        return title

    words = str(title).split()
    lines: List[str] = [""]
    for word in words:
        candidate = f"{lines[-1]} {word}".strip()
        if lines[-1] and _text_width_in(candidate, size) > budget and len(lines) < n_lines:
            lines.append(word)
        else:
            lines[-1] = candidate
    return "\n".join(lines)


def _label_left_column(axes: np.ndarray, n_used: int, ylabel: str) -> None:
    """Put one y-axis label on the leftmost panel of each row.

    Every panel in these grids measures the same quantity, so repeating the
    label on the right-hand column buys nothing and costs a gutter: at
    half-width, the label plus its tick numbers is about 0.55 in of a 2.95 in
    panel. Labelling the left column keeps the reading and returns the space.
    """
    for index in range(n_used):
        ax = axes[index]
        if ax.get_subplotspec().colspan.start == 0:
            ax.set_ylabel(ylabel)


def _close_unused_axes(axes: np.ndarray, n_used: int, xlabel: Optional[str] = None) -> None:
    """Hide the trailing empty cells of a page and label the bottom-most panels.

    ``sharex=True`` hides the tick labels of every panel that has one below it,
    so switching off a trailing cell would leave the panel above it without an
    axis. The x tick labels are therefore turned back on wherever that happens.
    """
    for ax in axes[n_used:]:
        ax.set_visible(False)
    if not n_used:
        return
    ncols = 1
    positions = [ax.get_subplotspec().colspan.start for ax in axes]
    if positions:
        ncols = max(positions) + 1
    for index in range(n_used):
        is_bottom = index + ncols >= n_used
        if is_bottom:
            axes[index].tick_params(axis="x", labelbottom=True)
            if xlabel:
                axes[index].set_xlabel(xlabel)

    # One wrapped title in a row pushes its own axes down and leaves the row
    # visibly out of line, so every title on the page is padded to the tallest.
    titles = [axes[index].get_title() for index in range(n_used)]
    tallest = max((title.count("\n") for title in titles), default=0)
    if tallest:
        for index, title in enumerate(titles):
            missing = tallest - title.count("\n")
            if missing > 0:
                axes[index].set_title(title + "\n" * missing)


RunSpec = Union[str, Path, Mapping[str, Any]]
RunData = MutableMapping[str, Any]
BEST_SELECTION_MODES = ("topdown", "independent")

def resolve_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "configs").exists():
            return candidate
    return cwd


def show_markdown(text: str) -> None:
    is_jupyter = False
    try:
        from IPython import get_ipython

        shell = get_ipython()
        is_jupyter = bool(shell) and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        is_jupyter = False

    if is_jupyter and Markdown is not None and display is not None:
        display(Markdown(text))
    else:
        print(text)


def _as_float_dict(metrics: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for key, value in metrics.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def normalize_metrics(metrics: Any) -> Dict[str, float]:
    return _as_float_dict(metrics)


def load_jsonl_events(path: Union[str, Path]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_test_results(raw: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, MappingABC):
        return {}

    results: Dict[str, Dict[str, Any]] = {}
    for mode in BEST_SELECTION_MODES:
        section = raw.get(mode)
        if not isinstance(section, MappingABC):
            continue
        results[mode] = {
            "best_checkpoint": section.get("best_checkpoint", ""),
            "best_epoch": section.get("best_epoch", None),
            "best_metric": section.get("best_metric", None),
            "test_metrics": normalize_metrics(section.get("test_metrics", {})),
        }
    if results:
        return results

    # Legacy format: a single best checkpoint/test-metrics payload at root level.
    metadata_keys = {"best_checkpoint", "best_epoch", "best_metric", "test_metrics"}
    legacy_metrics = normalize_metrics(raw.get("test_metrics", {}))
    if not legacy_metrics:
        metric_like = {key: value for key, value in raw.items() if key not in metadata_keys}
        legacy_metrics = normalize_metrics(metric_like)
    if not legacy_metrics and not any(key in raw for key in ("best_checkpoint", "best_epoch", "best_metric")):
        return {}
    results["topdown"] = {
        "best_checkpoint": raw.get("best_checkpoint", ""),
        "best_epoch": raw.get("best_epoch", None),
        "best_metric": raw.get("best_metric", None),
        "test_metrics": legacy_metrics,
    }
    return results


def metric_for_best(metrics: Mapping[str, float], mode: str = "topdown") -> float:
    m = normalize_metrics(metrics)

    if mode not in BEST_SELECTION_MODES:
        raise ValueError(f"Unknown selection mode '{mode}'. Expected one of {BEST_SELECTION_MODES}.")

    fpa_key = f"fpa_{mode}"
    tice_key = f"tice_{mode}"
    wap_key = f"weighted_ap_{mode}"
    has_fpa = fpa_key in m
    has_tice = tice_key in m
    has_wap = wap_key in m

    if has_fpa or has_tice or has_wap:
        fpa = float(m.get(fpa_key, 0.0))
        neg_tice = -float(m.get(tice_key, 1.0))
        wap = float(m.get(wap_key, 0.0))
        return float(fpa + 1e-3 * neg_tice + 1e-6 * wap)

    prefix = f"acc_level_{mode}_"
    deepest = [
        key
        for key in m
        if key.startswith(prefix) and key[len(prefix) :].isdigit()
    ]
    if not deepest:
        return float(m.get(fpa_key, 0.0))

    deepest_key = max(deepest, key=lambda key: int(key.rsplit("_", 1)[-1]))
    primary = float(m.get(deepest_key, 0.0))
    tie = float(m.get(fpa_key, 0.0))
    return float(primary + 1e-3 * tie)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_epoch_event(epoch_events: Sequence[Mapping[str, Any]], epoch: Optional[int]) -> Optional[Mapping[str, Any]]:
    if epoch is None:
        return None
    for event in epoch_events:
        if int(event.get("epoch", -1)) == epoch:
            return event
    return None


def _best_epoch_events_by_mode(
    epoch_events: Sequence[Mapping[str, Any]],
    test_results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Optional[Mapping[str, Any]]]:
    out: Dict[str, Optional[Mapping[str, Any]]] = {}
    for mode in BEST_SELECTION_MODES:
        result = test_results.get(mode, {}) if isinstance(test_results, MappingABC) else {}
        if (
            mode != "topdown"
            and isinstance(test_results, MappingABC)
            and (
                not isinstance(result, MappingABC)
                or (
                    result.get("best_epoch") is None
                    and not normalize_metrics(result.get("test_metrics", {}))
                )
            )
        ):
            result = test_results.get("topdown", {})
        best_event = _find_epoch_event(epoch_events, _coerce_int(result.get("best_epoch")))
        if best_event is None and epoch_events:
            best_event = max(
                epoch_events,
                key=lambda event: float(event.get("rank_scores", {}).get(mode, float("-inf"))),
            )
        out[mode] = best_event
    return out


def _test_metrics_for_mode(run_data: Mapping[str, Any], mode: str) -> Mapping[str, float]:
    test_results = run_data.get("test_results", {})
    if isinstance(test_results, MappingABC):
        section = test_results.get(mode)
        if isinstance(section, MappingABC) and isinstance(section.get("test_metrics"), MappingABC):
            metrics = section["test_metrics"]
            if metrics:
                return metrics
        section = test_results.get("topdown")
        if isinstance(section, MappingABC) and isinstance(section.get("test_metrics"), MappingABC):
            metrics = section["test_metrics"]
            if metrics:
                return metrics
    return {}


def _mode_from_metric_key(metric_key: str) -> str:
    if "_independent" in metric_key or metric_key.startswith("acc_level_independent_"):
        return "independent"
    return "topdown"


def _parse_single_run(run_dir: Union[str, Path]) -> Dict[str, Any]:
    run_path = Path(run_dir)
    events = load_jsonl_events(run_path / "run_log.jsonl")
    epoch_events = [event for event in events if event.get("event") == "epoch"]

    for event in epoch_events:
        event["train_metrics_norm"] = normalize_metrics(event.get("train_metrics", {}))
        event["val_metrics_norm"] = normalize_metrics(event.get("val_metrics", {}))
        event["rank_scores"] = {mode: metric_for_best(event["val_metrics_norm"], mode=mode) for mode in BEST_SELECTION_MODES}
        event["rank_score"] = float(event["rank_scores"]["topdown"])

    # Test metrics are sourced from test_metrics.yaml only.
    test_events = [event for event in events if event.get("event") == "test"]
    test_event = test_events[-1] if test_events else {}
    test_yaml_path = run_path / "test_metrics.yaml"
    test_metrics_yaml: Dict[str, Any] = load_yaml(test_yaml_path) if test_yaml_path.exists() else {}
    test_results = _normalize_test_results(test_metrics_yaml)
    test_metrics = dict(test_results.get("topdown", {}).get("test_metrics", {}))
    if not test_metrics and "independent" in test_results:
        test_metrics = dict(test_results["independent"]["test_metrics"])

    level_names: List[str] = []
    model_name: Optional[str] = None
    model_loss: Optional[str] = None
    weight_mode: Optional[str] = None
    model_transform_mode: Optional[str] = None
    cfg: Dict[str, Any] = {}
    cfg_path = run_path / "config_resolved.yaml"
    if cfg_path.exists():
        cfg = load_yaml(cfg_path) or {}
        level_names = list(cfg.get("dataset", {}).get("levels", []))
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, MappingABC):
            raw_model_name = model_cfg.get("name")
            if raw_model_name is not None:
                if not isinstance(raw_model_name, str):
                    raise ValueError("model.name in config_resolved.yaml must be a string.")
                model_name = raw_model_name
            raw_loss = model_cfg.get("loss")
            if isinstance(raw_loss, str):
                model_loss = raw_loss
            elif raw_loss is not None and model_name == "hiercos":
                raise ValueError("Hier-COS model.loss in config_resolved.yaml must be a string.")
            raw_weight_mode = model_cfg.get("weight_mode")
            if raw_weight_mode is not None:
                if not isinstance(raw_weight_mode, str):
                    raise ValueError("Hier-COS model.weight_mode in config_resolved.yaml must be a string.")
                weight_mode = raw_weight_mode
            raw_transform_mode = model_cfg.get("transform_mode")
            if raw_transform_mode is not None:
                if not isinstance(raw_transform_mode, str):
                    raise ValueError("Hier-COS model.transform_mode in config_resolved.yaml must be a string.")
                model_transform_mode = raw_transform_mode
    best_epoch_events = _best_epoch_events_by_mode(epoch_events, test_results)
    best_epoch_event = best_epoch_events.get("topdown")

    return {
        "run_dir": run_path,
        "run_name": run_path.name,
        "config": cfg,
        "events": events,
        "epoch_events": epoch_events,
        "test_event": test_event,
        "test_metrics": test_metrics,
        "test_results": test_results,
        "test_metrics_yaml": test_metrics_yaml,
        "level_names": level_names,
        "model_name": model_name,
        "model_loss": model_loss,
        "weight_mode": weight_mode,
        "model_transform_mode": model_transform_mode,
        "best_epoch_events": best_epoch_events,
        "best_epoch_event": best_epoch_event,
    }


def parse_run(
    run_dir: Union[str, Path],
    excluded_seeds: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Aggregate every seed of one run, minus the seeds named in `excluded_seeds`.

    A seed is excluded when it was trained under a protocol the rest of the
    group does not share - a different image size, split or schedule - so
    pooling it would average two experiments. The exclusion is printed rather
    than applied silently, because a run's seed count is read off these plots.
    """
    experiment_dir = Path(run_dir)
    seed_dirs = discover_seed_dirs(experiment_dir)
    excluded = {int(seed) for seed in (excluded_seeds or ())}
    if excluded:
        kept = []
        for seed_dir in seed_dirs:
            if seed_value_from_dir(seed_dir) in excluded:
                print(f"Excluding incompatible historical seed: {seed_dir}")
                continue
            kept.append(seed_dir)
        seed_dirs = kept
    if not seed_dirs:
        raise ValueError(f"No completed seed directories found under {experiment_dir}")
    seed_runs = [_parse_single_run(seed_dir) for seed_dir in seed_dirs]
    return aggregate_parsed_seed_runs(experiment_dir, seed_runs)


def get_metric_series(epoch_events: Sequence[Mapping[str, Any]], metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    epochs = [int(event["epoch"]) for event in epoch_events]
    values = [float(event.get("val_metrics_norm", {}).get(metric_key, np.nan)) for event in epoch_events]
    return np.array(epochs, dtype=np.int32), np.array(values, dtype=np.float64)


def get_train_loss_series(epoch_events: Sequence[Mapping[str, Any]], loss_key: str) -> Tuple[np.ndarray, np.ndarray]:
    epochs = [int(event["epoch"]) for event in epoch_events]
    values = [float(event.get("train_losses", {}).get(loss_key, np.nan)) for event in epoch_events]
    return np.array(epochs, dtype=np.int32), np.array(values, dtype=np.float64)


def get_train_metric_series(
    epoch_events: Sequence[Mapping[str, Any]], metric_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    epochs = [int(event["epoch"]) for event in epoch_events]
    values = []
    for event in epoch_events:
        value = np.nan
        for source_key in ("train_metrics_norm", "train_metrics"):
            source = event.get(source_key, {})
            if not isinstance(source, dict):
                continue
            value = _metric_value_from_source(source, metric_key)
            if np.isfinite(value):
                break
        values.append(float(value))
    return np.array(epochs, dtype=np.int32), np.array(values, dtype=np.float64)


def get_train_metric_series_any(
    epoch_events: Sequence[Mapping[str, Any]], metric_keys: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """First of ``metric_keys`` that carries finite values, canonical spelling first.

    Runs written before the canonical block keys existed only carry the trunk
    aliases, so a block metric is looked up under every spelling it may have.
    """
    epochs = np.array([int(event["epoch"]) for event in epoch_events], dtype=np.int32)
    for metric_key in metric_keys:
        _, values = get_train_metric_series(epoch_events, metric_key)
        if np.any(np.isfinite(values)):
            return epochs, values
    return epochs, np.full(epochs.shape, np.nan, dtype=np.float64)


def _metric_value_from_source(metric_source: Mapping[str, Any], metric_key: str) -> float:
    raw_value = metric_source.get(str(metric_key), np.nan)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return float("nan")
    if np.isfinite(value):
        return value
    return float("nan")


def moving_average_ignore_nan(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    win = int(window)
    if win <= 1:
        return arr

    finite_mask = np.isfinite(arr)
    safe_values = np.where(finite_mask, arr, 0.0)
    kernel = np.ones(win, dtype=np.float64)

    rolling_sum = np.convolve(safe_values, kernel, mode="full")[: arr.size]
    rolling_count = np.convolve(finite_mask.astype(np.float64), kernel, mode="full")[: arr.size]

    out = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = rolling_count > 0
    out[valid] = rolling_sum[valid] / rolling_count[valid]
    return out


def has_any_finite(run_series_cache: Sequence[Tuple[RunData, Dict[str, Tuple[np.ndarray, np.ndarray]]]], metric_key: str) -> bool:
    for _, metric_map in run_series_cache:
        _, values = metric_map[metric_key]
        if np.any(np.isfinite(values)):
            return True
    return False


def get_positive_robust_range(
    run_series_cache: Sequence[Tuple[RunData, Dict[str, Tuple[np.ndarray, np.ndarray]]]],
    metric_key: str,
) -> Optional[Tuple[float, float]]:
    chunks = []
    for _, metric_map in run_series_cache:
        _, values = metric_map[metric_key]
        vals = values[np.isfinite(values) & (values > 0)]
        if vals.size > 0:
            chunks.append(vals)

    if not chunks:
        return None

    concat = np.concatenate(chunks)
    lo = float(np.nanpercentile(concat, 2.0))
    hi = float(np.nanpercentile(concat, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= 0:
        return None

    ymin = max(1e-12, lo * 0.6)
    ymax = max(ymin * 5.0, hi * 1.4)
    return ymin, ymax


def get_level_label(level_idx: int, run_data: Mapping[str, Any]) -> str:
    """How the thesis names the level a run's config calls ``level_names[idx]``.

    The name is rewritten rather than read straight out of the config, because
    the config spelling is the dataset adapter's: CIFAR-100 writes
    ``coarse1/coarse2/fine`` where the thesis writes ``coarse/middle/fine``. A
    run whose config carries no level names falls back to the methodology's
    one-based numbering, never to ``L0``.
    """
    names = run_data.get("level_names") or []
    if level_idx < len(names):
        return level_display_name(names[level_idx])
    return f"level {level_idx + 1}"


def level_objective_label(level_idx: int, run_data: Mapping[str, Any]) -> str:
    """One level objective as a legend entry: ``Coarse ($\\ell_1$)``."""
    level_name = get_level_label(level_idx, run_data).capitalize()
    return f"{level_name} ({level_loss_symbol(level_idx)})"


# --------------------------------------------------------------------------- #
# Exact gradient-support blocks
# --------------------------------------------------------------------------- #
#
# The diagnostics are indexed by the blocks of docs/04-methodology.tex
# (ssec:lex_blocks): for parameter coordinate i, S(i) is the set of level
# objectives whose graph reaches it, and P_A = {i : S(i) = A}. `p123` is reached
# by all three levels, `p12` by coarse and middle only, and so on. Which blocks a
# run logs is set by `train.gradient_blocks`, and it differs by family: H-CAST
# branches fine-to-coarse and selects p123/p12/p1, HT-CapsNet runs the reverse
# cascade and selects p123/p23/p3, HRN and Hier-COS have one shared network and
# select p123 alone.
#
# Older runs wrote three of these blocks under H-CAST's private trunk names, so
# they are translated on read (train/lexicographic/gradients.py:21). The two
# union views are deliberately not translated: `t2t1` and `t3t2t1` are unions of
# blocks rather than blocks, and outside H-CAST they duplicate p123 exactly.

_GRADIENT_Y_LABELS = {
    "gradient_norms": "Norm",
    "gradient_alignment": "Cosine",
    "parameter_movement": "Norm",
    "projection_alignment": "Cosine",
    "projection_applied": "",
}

GRADIENT_BLOCK_ORDER = ("p123", "p23", "p13", "p12", "p3", "p2", "p1")
GRADIENT_LEVEL_ORDER = ("coarse", "mid", "fine")
LEGACY_EXACT_BLOCK_ALIASES = {"t1": "p123", "t2": "p12", "t3": "p1"}
LEGACY_UNION_ALIASES = ("t2t1", "t3t2t1")

BLOCK_LEVELS: Dict[str, Tuple[str, ...]] = {
    block: tuple(GRADIENT_LEVEL_ORDER[int(index) - 1] for index in block[1:])
    for block in GRADIENT_BLOCK_ORDER
}

# Ordered so a figure reads middle-against-coarse, then fine against each of its
# predecessors, then fine against their resultant.
CANONICAL_COSINE_PAIRS = (("mid", "coarse"), ("fine", "coarse"), ("fine", "mid"))
HIGHER_REFERENCE = "higher"

_SUBSCRIPT_DIGITS = {"1": "₁", "2": "₂", "3": "₃"}


def block_label(block: str, *, tex: bool = True) -> str:
    """`p123` as it is written in the thesis."""
    digits = str(block)[1:]
    if tex:
        return f"$P_{{{digits}}}$"
    return "P" + "".join(_SUBSCRIPT_DIGITS.get(digit, digit) for digit in digits)


def _canonical_block(raw_block: str) -> Optional[str]:
    """Map a logged block token to a canonical block, or None if it is a union."""
    token = str(raw_block)
    if token in LEGACY_UNION_ALIASES:
        return None
    token = LEGACY_EXACT_BLOCK_ALIASES.get(token, token)
    return token if token in BLOCK_LEVELS else None


def block_metric_keys(canonical_key: str) -> Tuple[str, ...]:
    """Every log spelling of one canonical key, newest first.

    Runs from August 2026 onward write the `p...` keys; everything before that
    wrote only the trunk aliases, so both are tried.
    """
    keys = [canonical_key]
    for legacy_token, canonical_block in LEGACY_EXACT_BLOCK_ALIASES.items():
        marker = f"_{canonical_block}_"
        if marker in canonical_key:
            keys.append(canonical_key.replace(marker, f"_{legacy_token}_", 1))
        elif canonical_key.endswith(f"_{canonical_block}"):
            keys.append(canonical_key[: -len(canonical_block)] + legacy_token)
    return tuple(dict.fromkeys(keys))


def grad_norm_key(block: str, level: str, *, post: bool = False) -> str:
    return f"{'post_' if post else ''}grad_norm_{block}_{level}"


def cosine_key(block: str, target: str, reference: str, *, post: bool = False) -> str:
    if not post:
        return f"cos_{block}_{target}_{reference}"
    # The projected spelling differs between the canonical and the legacy keys;
    # the canonical one is built here and `block_metric_keys` supplies the alias.
    return f"post_cos_{block}_{target}_{reference}"


def projection_flag_key(block: str, target: str, reference: str) -> str:
    return f"post_projection_applied_{block}_{target}_{reference}"


# The legacy projected-cosine names do not follow the canonical pattern - they
# carry a `_proj` suffix on whichever operand was projected - so they are listed
# rather than derived. Keys are (block, target, reference).
_LEGACY_POST_COSINE_NAMES = {
    ("p12", "mid", "coarse"): "post_cos_t2_mid_proj_coarse",
    ("p123", "mid", "coarse"): "post_cos_t1_mid_proj_coarse",
    ("p123", "fine", HIGHER_REFERENCE): "post_cos_t1_fine_proj_higher",
    ("p123", "fine", "coarse"): "post_cos_t1_fine_proj_coarse",
    ("p123", "fine", "mid"): "post_cos_t1_fine_proj_mid_proj",
}
_LEGACY_PROJECTION_FLAG_NAMES = {
    ("p12", "mid", "coarse"): "post_projection_applied_t2_mid_coarse",
    ("p123", "mid", "coarse"): "post_projection_applied_t1_mid_coarse",
    ("p123", "fine", HIGHER_REFERENCE): "post_projection_applied_t1_fine_higher",
    ("p123", "mid", "fine"): "post_projection_applied_t1_mid_fine",
    ("p12", "coarse", "mid"): "post_projection_applied_t2_coarse_mid",
    ("p123", "coarse", HIGHER_REFERENCE): "post_projection_applied_t1_coarse_higher",
}


def post_cosine_keys(block: str, target: str, reference: str) -> Tuple[str, ...]:
    keys = list(block_metric_keys(cosine_key(block, target, reference, post=True)))
    legacy = _LEGACY_POST_COSINE_NAMES.get((block, target, reference))
    if legacy:
        keys.append(legacy)
    return tuple(dict.fromkeys(keys))


def post_projection_flag_keys(block: str, target: str, reference: str) -> Tuple[str, ...]:
    keys = list(block_metric_keys(projection_flag_key(block, target, reference)))
    legacy = _LEGACY_PROJECTION_FLAG_NAMES.get((block, target, reference))
    if legacy:
        keys.append(legacy)
    return tuple(dict.fromkeys(keys))


def _pre_keys_for_post(post_keys: Sequence[str]) -> Tuple[str, ...]:
    """The pre-projection counterpart of each projected-cosine key.

    Canonical keys only drop the `post_` prefix; the legacy spellings also carry
    a `_proj` marker on whichever operand was projected, which the raw key does
    not have.
    """
    pre_keys = []
    for key in post_keys:
        pre = str(key)
        if pre.startswith("post_"):
            pre = pre[len("post_") :]
        pre = pre.replace("_proj_", "_")
        if pre.endswith("_proj"):
            pre = pre[: -len("_proj")]
        pre_keys.append(pre)
    return tuple(dict.fromkeys(pre_keys))


def _level_phrase(level: str) -> str:
    if level == HIGHER_REFERENCE:
        return "higher-priority"
    return level_display_name(level)


def gradient_panel_title(kind: str, block: str, *args: str, post: bool = False) -> str:
    """A panel title a reader can parse without the log glossary."""
    where = f"on {block_label(block)}"
    after = ", after projection" if post else ""
    if kind == "grad_norm":
        level = _level_phrase(args[0]).capitalize()
        return f"{level} gradient {where}{after}"
    if kind == "cos":
        target, reference = _level_phrase(args[0]), _level_phrase(args[1])
        return f"{target.capitalize()} vs {reference} alignment {where}{after}"
    if kind == "param_norm":
        return f"Parameter norm {where}"
    if kind == "delta_param_norm":
        # "parameter" is left implicit: the panel beside it says "Parameter
        # norm", and the longer phrasing wraps to two lines in a half-width
        # panel, which knocks the right column out of line with the left.
        return f"Per-epoch movement {where}"
    if kind == "projection_applied":
        return f"{_level_phrase(args[0]).capitalize()} projection applied {where}"
    raise ValueError(f"Unknown gradient panel kind '{kind}'.")


def _normalize_dataset_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("dataset key must be a string.")
    if value in {"cifar-100", "cub-200-2011", "fgvc-aircraft"}:
        return value
    raise ValueError(
        f"Unsupported dataset key '{value}'. "
        "Expected one of ['cifar-100', 'cub-200-2011', 'fgvc-aircraft']."
    )


def _resolve_run_dir(path_like: Union[str, Path], outputs_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return outputs_root / path


def _metadata_seed_dir(run_dir: Path) -> Path:
    seed_dirs = discover_seed_dirs(run_dir)
    return seed_dirs[0] if seed_dirs else Path(run_dir)


def _read_hcc_cfg(run_dir: Path) -> Dict[str, Any]:
    cfg_path = _metadata_seed_dir(run_dir) / "config_resolved.yaml"
    if not cfg_path.exists():
        return {}
    cfg = load_yaml(cfg_path) or {}

    hcc_cfg = cfg.get("hcc", None)
    if isinstance(hcc_cfg, dict):
        return dict(hcc_cfg)

    hcc_cfg = cfg.get("model", {}).get("hcc", {})
    if not isinstance(hcc_cfg, dict):
        return {}
    return dict(hcc_cfg)


def _read_dataset_name(run_dir: Path) -> Optional[str]:
    cfg_path = _metadata_seed_dir(run_dir) / "config_resolved.yaml"
    if not cfg_path.exists():
        return None
    cfg = load_yaml(cfg_path) or {}
    dataset_cfg = cfg.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return None
    raw_name = dataset_cfg.get("name", None)
    if raw_name is None:
        return None
    if not isinstance(raw_name, str):
        raise ValueError("dataset.name in config_resolved.yaml must be a string.")
    return raw_name


def _read_hcc_temperature(run_dir: Path) -> Optional[float]:
    raw_temp = _read_hcc_cfg(run_dir).get("temperature", None)
    if raw_temp is None:
        return None
    try:
        return float(raw_temp)
    except (TypeError, ValueError):
        return None


def _read_hcc_projection_mode(run_dir: Path) -> Optional[str]:
    mode = _read_hcc_cfg(run_dir).get("projection_mode", None)
    if mode is None:
        return None
    return str(mode)


def _read_hcc_constraint_strength_max(run_dir: Path) -> Optional[float]:
    raw = _read_hcc_cfg(run_dir).get("constraint_strength_max", None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return min(max(value, 0.0), 1.0)


def _run_meta_from_dir(run_dir: Path) -> Dict[str, Any]:
    return {
        "run_dir": Path(run_dir),
        "temperature": _read_hcc_temperature(run_dir),
        "hcc_projection_mode": _read_hcc_projection_mode(run_dir),
        "hcc_constraint_strength_max": _read_hcc_constraint_strength_max(run_dir),
        "dataset_name": _read_dataset_name(run_dir),
    }


def _detect_hiercos_study_family(run_like: Mapping[str, Any], text: str) -> Optional[str]:
    model_name_raw = run_like.get("model_name", None)
    model_name = model_name_raw.strip().lower() if isinstance(model_name_raw, str) else ""
    loss_raw = run_like.get("model_loss", None)
    loss_mode = loss_raw.strip().lower() if isinstance(loss_raw, str) else ""
    transform_raw = run_like.get("model_transform_mode", None)
    transform_mode = transform_raw.strip().lower() if isinstance(transform_raw, str) else ""

    looks_like_hiercos = (
        model_name == "hiercos"
        or "hiercos" in text
        or "hier-cos" in text
        or "global_softmax_ce_reg" in text
        or "level_softmax_ce_reg" in text
    )
    if not looks_like_hiercos:
        return None

    if (
        transform_mode == "final_only"
        or "final_only" in text
        or "final only" in text
        or "final fixed" in text
        or "fixed layer" in text
    ):
        return "hiercos_final_only"

    if loss_mode == "level_softmax_ce_reg" or "level_softmax_ce_reg" in text:
        return "hiercos_loss_level_softmax_ce_reg"
    if loss_mode == "global_softmax_ce_reg" or "global_softmax_ce_reg" in text:
        return "hiercos_loss_global_softmax_ce_reg"
    if loss_mode == "kl_reg":
        return "hiercos_loss_kl_reg"
    return "hiercos_loss_kl_reg"


def _detect_color_family(run_like: Mapping[str, Any]) -> str:
    if bool(run_like.get("is_baseline", False)):
        return "baseline"

    label = str(run_like.get("label", "")).lower()
    run_dir_name = ""
    run_dir = run_like.get("run_dir", None)
    if run_dir is not None:
        run_dir_name = Path(run_dir).name.lower()

    text = f"{label} {run_dir_name}"
    hiercos_family = _detect_hiercos_study_family(run_like, text)
    if hiercos_family is not None:
        return hiercos_family

    if ("ce " in text) or ("ce_" in text):
        return "ce_weight"
    if "lex" in text:
        return "lex"
    if "hcc" in text or "step" in text or "inverse" in text:
        return "hcc"
    return "other"


def _gradient_hexes(cmap_name: str, n: int, lo: float = 0.40, hi: float = 0.88) -> List[str]:
    if int(n) <= 0:
        return []
    if int(n) == 1:
        positions = np.array([(float(lo) + float(hi)) / 2.0], dtype=np.float64)
    else:
        positions = np.linspace(float(lo), float(hi), int(n))
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(float(pos))) for pos in positions]


def _palette_gradient_hexes(anchor_hexes: Sequence[str], n: int) -> List[str]:
    """Interpolate between hand-picked high-contrast anchors."""
    if int(n) <= 0:
        return []
    anchors = list(anchor_hexes)
    if int(n) == 1:
        return [anchors[min(len(anchors) // 2, len(anchors) - 1)]]
    cmap = mcolors.LinearSegmentedColormap.from_list("semantic_run_palette", anchors)
    return [mcolors.to_hex(cmap(float(pos))) for pos in np.linspace(0.0, 1.0, int(n))]


def _apply_semantic_color_gradients(run_data_by_dataset: Mapping[str, List[RunData]]) -> None:
    family_palettes = {
        # Baseline is intentionally stable and blue across datasets.
        "baseline": ["#1f77b4", "#0b4f8a"],
        # Hier-COS paper-aligned KL + regularization family.
        "hiercos_loss_kl_reg": ["#9ca3af", "#6b7280", "#374151"],
        # Hier-COS global-softmax CE + regularization family.
        "hiercos_loss_global_softmax_ce_reg": ["#86efac", "#22c55e", "#166534"],
        # Hier-COS level-softmax CE + regularization family.
        "hiercos_loss_level_softmax_ce_reg": ["#fdba74", "#f97316", "#c2410c"],
        # Hier-COS final fixed-layer ablation (`transform_mode=final_only`).
        "hiercos_final_only": ["#c4b5fd", "#8b5cf6", "#5b21b6"],
        # HCC variants use a perceptually clear warm ramp instead of similar greens.
        "hcc": ["#f6d32d", "#f59e0b", "#ef4444", "#991b1b"],
        # Lexicographic runs use a green/teal ramp, separated from baseline blue and HCC warm colors.
        "lex": ["#006d5b", "#009e73", "#20c997"],
        # Hier-COS CE weight variants use a visible blue ramp instead of gray.
        "ce_weight": ["#93c5fd", "#3b82f6", "#1d4ed8"],
        # Keep a colorful fallback for uncategorized runs.
        "other": ["#f97316", "#c026d3"],
    }

    for dataset_runs in run_data_by_dataset.values():
        ordered = sorted(dataset_runs, key=lambda run: run.get("_run_order", 0))
        buckets: Dict[str, List[RunData]] = {key: [] for key in family_palettes.keys()}

        for run in ordered:
            if bool(run.get("color_locked", False)):
                continue
            family = _detect_color_family(run)
            if family not in buckets:
                family = "other"
            buckets[family].append(run)

        for family, runs in buckets.items():
            colors = _palette_gradient_hexes(family_palettes[family], len(runs))
            for run, color in zip(runs, colors):
                run["color"] = color


def _positive_robust_range_for_keys(
    run_series_cache: Sequence[Tuple[RunData, Dict[str, Tuple[np.ndarray, np.ndarray]]]],
    metric_keys: Sequence[str],
) -> Optional[Tuple[float, float]]:
    chunks = []
    for _, metric_map in run_series_cache:
        for metric_key in metric_keys:
            _, values = metric_map[metric_key]
            vals = values[np.isfinite(values) & (values > 0)]
            if vals.size > 0:
                chunks.append(vals)

    if not chunks:
        return None

    concat = np.concatenate(chunks)
    lo = float(np.nanpercentile(concat, 2.0))
    hi = float(np.nanpercentile(concat, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= 0:
        return None

    ymin = max(1e-12, lo * 0.6)
    ymax = max(ymin * 5.0, hi * 1.4)
    return ymin, ymax


def _build_train_metric_map(
    epoch_events: Sequence[Mapping[str, Any]], metric_keys: Sequence[str]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return {metric_key: get_train_metric_series(epoch_events, metric_key) for metric_key in metric_keys}


def _canonical_hiercos_loss_mode(raw_mode: Optional[str], default: str = "kl_reg") -> str:
    mode = default if raw_mode is None else raw_mode
    if not isinstance(mode, str):
        raise ValueError("Hier-COS model.loss must be a string.")
    if mode == "kl_reg":
        return "kl_reg"
    if mode == "global_softmax_ce_reg":
        return "global_softmax_ce_reg"
    if mode == "level_softmax_ce_reg":
        return "level_softmax_ce_reg"
    raise ValueError(
        f"Unsupported Hier-COS model.loss '{mode}'. "
        "Expected one of ['kl_reg', 'global_softmax_ce_reg', 'level_softmax_ce_reg']."
    )


def _fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{100.0 * x:.2f}%"


def _fmt_delta_pp(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    sign = "+" if x >= 0 else ""
    return f"{sign}{100.0 * x:.2f} pp"


def _fmt_edges(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.3f}"


def _fmt_delta_edges(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.3f}"


def _metric_goal_is_lower_better(metric_key: str) -> bool:
    return metric_key.startswith("tice_") or metric_key.startswith("ahd_")


def _metric_goal_arrow(metric_key: str) -> str:
    return "↓" if _metric_goal_is_lower_better(metric_key) else "↑"


def _is_distance_metric(metric_key: str) -> bool:
    return metric_key.startswith("ahd_")


def _fmt_value(metric_key: str, value: float) -> str:
    return _fmt_edges(value) if _is_distance_metric(metric_key) else _fmt_pct(value)


def _fmt_delta(metric_key: str, delta_value: float) -> str:
    return _fmt_delta_edges(delta_value) if _is_distance_metric(metric_key) else _fmt_delta_pp(delta_value)


def _merged_comp_cell(metric_key: str, comp_value: float, base_value: float) -> str:
    comp_txt = _fmt_value(metric_key, comp_value)
    if not np.isfinite(base_value) or not np.isfinite(comp_value):
        return comp_txt
    delta_txt = _fmt_delta(metric_key, comp_value - base_value)
    return f"{comp_txt} ({delta_txt})"


def _test_metric_stats(run_data: Mapping[str, Any], metric_key: str) -> Tuple[float, float, int]:
    mode = _mode_from_metric_key(metric_key)
    section = run_data.get("test_results", {}).get(mode, {})
    if not isinstance(section, MappingABC):
        return float("nan"), float("nan"), 0
    mean = float(section.get("test_metrics", {}).get(metric_key, np.nan))
    std = float(section.get("test_metrics_std", {}).get(metric_key, np.nan))
    count = int(section.get("test_metrics_count", {}).get(metric_key, 0))
    return mean, std, count


def _fmt_value_stats(metric_key: str, mean: float, std: float, count: int) -> str:
    mean_text = _fmt_value(metric_key, mean)
    if count > 1 and np.isfinite(std):
        return f"{mean_text} ± {_fmt_value(metric_key, std)}"
    return mean_text


def _best_and_second_best_indices(metric_key: str, values: Sequence[float]) -> Tuple[set[int], set[int]]:
    finite_pairs = [(idx, val) for idx, val in enumerate(values) if np.isfinite(val)]
    if not finite_pairs:
        return set(), set()

    lower_is_better = _metric_goal_is_lower_better(metric_key)
    sorted_pairs = sorted(finite_pairs, key=lambda item: item[1], reverse=not lower_is_better)

    grouped_indices: List[List[int]] = []
    grouped_values: List[float] = []
    for idx, value in sorted_pairs:
        if grouped_values and np.isclose(value, grouped_values[-1], rtol=1e-9, atol=1e-12):
            grouped_indices[-1].append(idx)
        else:
            grouped_values.append(value)
            grouped_indices.append([idx])

    best_indices = set(grouped_indices[0]) if grouped_indices else set()
    second_best_indices = set(grouped_indices[1]) if len(grouped_indices) > 1 else set()
    return best_indices, second_best_indices


# (mode key, line style, legend label, best-epoch marker) for each decoder.
_MODE_SPECS: Dict[str, Tuple[str, str, str, str]] = {
    "independent": ("independent", MODE_LINESTYLES["independent"], "independent", "o"),
    "topdown": ("topdown", MODE_LINESTYLES["topdown"], "top-down", "x"),
}


def _default_mode_specs(
    view: Optional[Union[str, bool]] = None
) -> List[Tuple[str, str, str, str]]:
    return [_MODE_SPECS[mode] for mode in decoder_modes(view)]


def _filter_mode_specs(
    mode_specs: Sequence[Tuple[str, str, str, str]],
    view: Optional[Union[str, bool]] = None,
) -> List[Tuple[str, str, str, str]]:
    wanted = set(decoder_modes(view))
    return [spec for spec in mode_specs if spec[0] in wanted]


# Markers that tie a log key to one decoder. Each decoder is spelled two ways in
# the metric names: `_topdown`/`_independent` on the decoded metrics, and the
# shorter `_td_`/`_ind_` the conditional diagnostics use
# (`acc_l2_td_given_l1_correct`). A key matching neither belongs to both views.
_TOPDOWN_KEY_MARKERS = ("_topdown", "_td_correct", "_td_given_")
_INDEPENDENT_KEY_MARKERS = ("_independent", "_ind_correct", "_ind_given_")


def _metric_key_decoder(metric_key: str) -> Optional[str]:
    """Which decoder a log key belongs to, or None if it belongs to neither."""
    key = str(metric_key)
    if any(marker in key for marker in _TOPDOWN_KEY_MARKERS):
        return "topdown"
    if any(marker in key for marker in _INDEPENDENT_KEY_MARKERS):
        return "independent"
    return None


def _include_metric_for_view(
    metric_key: str, view: Optional[Union[str, bool]] = None
) -> bool:
    decoder = _metric_key_decoder(metric_key)
    return decoder is None or decoder in decoder_modes(view)


@dataclass
class HCastAnalysisConfig:
    outputs_root: Path = Path(os.environ.get("OUTPUTS_ROOT", "/scratch/g.saggini1/outputs"))
    include_baselines: bool = True
    # Keep a dataset in the analysis when its configured baseline is the only
    # available run, instead of dropping the dataset from every plot.
    include_baseline_only_datasets: bool = True
    # Which decoders this analysis shows: 'both', 'independent' or 'topdown'.
    # None follows the module-level `DECODER_VIEW`, which is what a notebook
    # sets with `set_decoder_view(...)`.
    decoder_view: Optional[str] = None
    baseline_color: str = "#1f77b4"
    baseline_by_dataset: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "cifar-100": {"run_dir": "hcast_cifar100", "label": "H-CAST"},
            "cub-200-2011": {"run_dir": "hcast_cub200", "label": "H-CAST"},
            "fgvc-aircraft": {"run_dir": "hcast_aircraft", "label": "H-CAST"},
        }
    )
    manual_runs: List[RunSpec] = field(default_factory=list)
    # Seeds to leave out of one run's aggregate, keyed by run directory name.
    # Used for a seed trained under a protocol the rest of its group does not
    # share, which would otherwise be averaged into the mean and the band.
    excluded_seeds_by_run: Dict[str, Sequence[int]] = field(default_factory=dict)
    # Terms of the aggregate objective to plot, in reading order. Set from the
    # family profile; terms the family never logs are dropped downstream.
    aggregate_loss_keys: Tuple[str, ...] = ()
    # Prefix of every exported figure stem, e.g. 'hcast' -> hcast_cifar100_validation_curves.
    family_slug: str = "model"
    temperature_color_palette: List[str] = field(
        default_factory=lambda: [
            "#d62728",
            "#2ca02c",
            "#ff7f0e",
            "#9467bd",
            "#8c564b",
            "#17becf",
            "#e377c2",
            "#bcbd22",
        ]
    )
    auto_recolor_unlocked_runs: bool = True


# --------------------------------------------------------------------- #
# Family registry
# --------------------------------------------------------------------- #
#
# Every model-analysis notebook renders the same sections in the same order, so
# the only thing that legitimately varies between them is which runs to read and
# what the aggregate objective is called. Those two things live here rather than
# in the notebooks, because a difference that lives in five notebooks drifts:
# the previous versions had five hand-written diagnostic spec lists, two
# spellings of the same figure stem, and one family silently rendering half its
# curves. A notebook is now `HCastAnalysis.for_family('hrn')`.
#
# Whatever a family does NOT have is still left to the data: the library drops a
# loss key nothing logs, a diagnostic panel nothing moved, and a gradient block
# the architecture does not contain. So a run matrix may list an arm that has
# not been launched yet - it is reported as missing and picked up automatically
# once it lands.

# Dataset order and slugs come from thesis_style, so the figures, the file names
# and the family profiles cannot disagree about what a dataset is called.
DATASETS: Tuple[str, ...] = tuple(DATASET_SLUGS.values())

# Colour is per run within a dataset; the baseline keeps its own fixed colour so
# it is the same blue in every family's figures.
BASELINE_COLOR = "#243ab4"
RUN_COLOR_PALETTE: Tuple[str, ...] = (
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#e377c2",
    "#bcbd22",
)


# The mechanisms a family can carry on top of its baseline, and how a figure
# names them. A mechanism is cross-family only if more than one family has an arm
# for it; `MechanismDeltaComparison` reports which do, rather than assuming.
MECHANISM_LABELS: Dict[str, str] = {
    "hcc": "HCC",
    "lex_coarse_first": "Lexicographic (coarse-first)",
    "lex_fine_first": "Lexicographic (fine-first)",
}


@dataclass(frozen=True)
class FamilyProfile:
    """What one model family contributes to an otherwise identical notebook."""

    slug: str
    display_name: str
    # `{dataset_slug}` is substituted per dataset.
    baseline_template: str
    baseline_label: str
    # (run_dir_template, label) in the order they should appear in a legend.
    comparison_templates: Tuple[Tuple[str, str], ...] = ()
    # Runs that exist for one dataset only, as (dataset_slug, run_dir, label).
    extra_runs: Tuple[Tuple[str, str, str], ...] = ()
    # Aggregate objective terms, in the order they should be read. Terms the
    # family never logs are dropped, and two keys that resolve to one display
    # name (HRN's `hier_loss`/`tree_loss`) collapse to a single panel.
    aggregate_loss_keys: Tuple[str, ...] = ("total",)
    excluded_seeds_by_run: Mapping[str, Sequence[int]] = field(default_factory=dict)
    # Mechanism key -> run-dir template, for the arms that exist in more than one
    # family and can therefore be compared across architectures. Every template
    # here must also appear in `comparison_templates`, which is checked at import:
    # the two lists describe the same runs and must not drift apart.
    mechanism_arms: Mapping[str, str] = field(default_factory=dict)

    def baseline_by_dataset(self) -> Dict[str, Dict[str, Any]]:
        return {
            dataset_key: {
                "run_dir": self.baseline_template.format(dataset=slug),
                "label": self.baseline_label,
            }
            for dataset_key, slug in DATASET_SLUGS.items()
        }

    def manual_runs(self) -> List[RunSpec]:
        runs: List[RunSpec] = [
            {"run_dir": template.format(dataset=slug), "label": label}
            for template, label in self.comparison_templates
            for slug in DATASETS
        ]
        runs.extend({"run_dir": run_dir, "label": label} for _slug, run_dir, label in self.extra_runs)
        return runs


FAMILY_PROFILES: Dict[str, FamilyProfile] = {
    "hcast": FamilyProfile(
        slug="hcast",
        display_name="H-CAST",
        baseline_template="hcast_{dataset}",
        baseline_label="H-CAST baseline (global KL)",
        comparison_templates=(
            ("hcast_{dataset}_hcc", "H-CAST + HCC"),
            ("hcast_{dataset}_lex_coarse_first", "H-CAST lex coarse-first"),
            ("hcast_{dataset}_lex_fine_first", "H-CAST lex fine-first"),
        ),
        aggregate_loss_keys=("total", "level_ce", "gk_loss"),
        mechanism_arms={
            "hcc": "hcast_{dataset}_hcc",
            "lex_coarse_first": "hcast_{dataset}_lex_coarse_first",
            "lex_fine_first": "hcast_{dataset}_lex_fine_first",
        },
    ),
    "lhdnn": FamilyProfile(
        slug="lhdnn",
        display_name="LH-DNN",
        baseline_template="lhdnn_{dataset}",
        baseline_label="LH-DNN baseline",
        # LH-DNN's branch-point projection is backward-only, so the family has
        # no separate on/off arm: the baseline is the method. Its output-space
        # section is correctly empty, which is the finding, not a gap.
        comparison_templates=(),
        aggregate_loss_keys=("total", "level_ce"),
    ),
    "ht_capsnet": FamilyProfile(
        slug="ht_capsnet",
        display_name="HT-CapsNet",
        baseline_template="capsnet_{dataset}",
        baseline_label="HT-CapsNet baseline (dynamic weights)",
        comparison_templates=(
            ("capsnet_{dataset}_hcc", "HT-CapsNet + HCC (exploratory)"),
            ("ht_capsnet_{dataset}_lex_coarse_first", "HT-CapsNet lex coarse-first (unit weights)"),
        ),
        # Trained at batch_size=256 before the switch to the upstream 32, with
        # the same optimizer, split_seed and 100-epoch horizon. Kept as its own
        # run so it is compared against the batch_size=32 seeds, not pooled.
        extra_runs=(("cifar100", "capsnet_cifar100_bs256", "HT-CapsNet CIFAR-100 (batch 256)"),),
        aggregate_loss_keys=("total", "margin"),
        mechanism_arms={
            "hcc": "capsnet_{dataset}_hcc",
            "lex_coarse_first": "ht_capsnet_{dataset}_lex_coarse_first",
        },
        # seed_42 predates the corrected attention projections and used a
        # different image size, split and schedule.
        excluded_seeds_by_run={"capsnet_cifar100": (42,)},
    ),
    "hrn": FamilyProfile(
        slug="hrn",
        display_name="HRN",
        baseline_template="hrn_{dataset}",
        baseline_label="HRN baseline",
        # HRN needs model.loss=level_marginal for the objective to decompose
        # into the three per-level terms HCC and the lexicographic step are
        # defined on, so every comparison arm carries that suffix. The plain
        # baseline consequently logs no per-level loss and no per-level
        # gradient, and those sections report themselves empty for it.
        comparison_templates=(
            ("hrn_{dataset}_level_marginal_hcc", "HRN level_marginal + HCC"),
            ("hrn_{dataset}_level_marginal_lex_coarse_first", "HRN level_marginal lex coarse-first"),
            ("hrn_{dataset}_level_marginal_lex_fine_first", "HRN level_marginal lex fine-first"),
        ),
        aggregate_loss_keys=("total", "tree_loss", "hier_loss", "ce_loss_leaf", "fine_ce"),
        # Every HRN arm also switches the objective to `level_marginal`, which the
        # plain baseline does not use, so an HRN delta carries that second change
        # with it. `MechanismDeltaComparison` prints this rather than hiding it.
        mechanism_arms={
            "hcc": "hrn_{dataset}_level_marginal_hcc",
            "lex_coarse_first": "hrn_{dataset}_level_marginal_lex_coarse_first",
            "lex_fine_first": "hrn_{dataset}_level_marginal_lex_fine_first",
        },
    ),
    "hiercos": FamilyProfile(
        slug="hiercos",
        display_name="Hier-COS",
        baseline_template="hiercos_{dataset}_global_softmax_ce_reg_baseline_kl_leaf",
        baseline_label="Hier-COS baseline (global softmax, kl_leaf)",
        comparison_templates=(
            ("hiercos_{dataset}_global_softmax_ce_reg_hcc", "Hier-COS + HCC"),
            (
                "hiercos_{dataset}_global_softmax_ce_reg_lex_coarse_first_kl_leaf",
                "Hier-COS lex coarse-first",
            ),
            (
                "hiercos_{dataset}_global_softmax_ce_reg_lex_fine_first_kl_leaf",
                "Hier-COS lex fine-first",
            ),
            ("hiercos_{dataset}_subspace", "Hier-COS subspace supervision"),
            (
                "hiercos_{dataset}_level_softmax_ce_reg_projection_kl_leaf_identity",
                "Hier-COS LH-projection (identity)",
            ),
        ),
        aggregate_loss_keys=(
            "total",
            "ce",
            "reg",
            "kl",
            "subspace_soft_cross_entropy",
            "subspace_target_kl",
            "subspace_score_l2",
        ),
        mechanism_arms={
            "hcc": "hiercos_{dataset}_global_softmax_ce_reg_hcc",
            "lex_coarse_first": "hiercos_{dataset}_global_softmax_ce_reg_lex_coarse_first_kl_leaf",
            "lex_fine_first": "hiercos_{dataset}_global_softmax_ce_reg_lex_fine_first_kl_leaf",
        },
    ),
}


# The families of the cross-model comparison, in the order they should appear in
# a legend. The first one is the delta baseline of every test table there.
#
# The comparison reads each family's baseline out of `FAMILY_PROFILES` rather
# than naming run directories again, so the cross-model notebook and the family
# notebook cannot end up disagreeing about which run is a family's baseline -
# which is the failure the family registry was written to prevent in the first
# place, one level up.
COMPARISON_FAMILIES: Tuple[str, ...] = ("hcast", "lhdnn", "hrn", "ht_capsnet", "hiercos")


def _validate_mechanism_arms() -> None:
    """Every mechanism arm must be a run the family notebook already plots.

    `mechanism_arms` and `comparison_templates` name the same runs from two
    directions - by mechanism and in legend order - so a run renamed in one and
    not the other would give the cross-model page a run the family page does not
    have. Checking at import turns that into an error at the first cell instead
    of a silently missing column.
    """
    for family, profile in FAMILY_PROFILES.items():
        known = {template for template, _label in profile.comparison_templates}
        for mechanism, template in profile.mechanism_arms.items():
            if mechanism not in MECHANISM_LABELS:
                raise KeyError(
                    f"{family}: mechanism '{mechanism}' has no entry in MECHANISM_LABELS."
                )
            if template not in known:
                raise KeyError(
                    f"{family}: mechanism arm '{template}' is not in comparison_templates."
                )


_validate_mechanism_arms()


class HCastAnalysis:
    def __init__(self, config: HCastAnalysisConfig, run_data_list: List[RunData]) -> None:
        self.config = config
        self.run_data_list = run_data_list
        # Set by `for_family`; None when a notebook built the config by hand.
        self.profile: Optional[FamilyProfile] = None
        self.run_data_by_dataset: Dict[str, List[RunData]] = {}
        for run_data in run_data_list:
            self.run_data_by_dataset.setdefault(run_data["dataset_key"], []).append(run_data)

        for dataset_runs in self.run_data_by_dataset.values():
            dataset_runs.sort(key=lambda run: (0 if run.get("is_baseline") else 1, run.get("_run_order", 0)))

        if bool(config.auto_recolor_unlocked_runs):
            _apply_semantic_color_gradients(self.run_data_by_dataset)
        # Fixed dataset order, so the CIFAR-100 figure is the first one in every
        # family's notebook. Discovery order would otherwise depend on which run
        # happened to resolve first, which differs per family.
        canonical = list(DATASET_SLUGS)
        self.dataset_keys = sorted(
            self.run_data_by_dataset,
            key=lambda key: (canonical.index(key) if key in canonical else len(canonical), str(key)),
        )

    @classmethod
    def from_config(cls, config: HCastAnalysisConfig) -> "HCastAnalysis":
        runs = cls._build_runs_from_manual(config)
        if config.include_baselines:
            cls._append_dataset_baselines(config, runs)
        if not runs:
            print("No runs found. Check MANUAL_RUNS/BASELINE_BY_DATASET and output paths.")
            return cls(config=config, run_data_list=[])

        run_data_list: List[RunData] = []
        for run in runs:
            run_dir = Path(run["run_dir"])
            parsed = parse_run(run_dir, config.excluded_seeds_by_run.get(run_dir.name))
            parsed.update(run)
            dataset_name = parsed.get("dataset_name")
            if not dataset_name:
                level_names = [str(name) for name in (parsed.get("level_names") or [])]
                dataset_name = "/".join(level_names) if level_names else parsed["run_dir"].name
            parsed["dataset_name"] = str(dataset_name)
            parsed_dataset_key = _normalize_dataset_key(parsed["dataset_name"])
            parsed["dataset_key"] = parsed_dataset_key
            parsed["_run_order"] = len(run_data_list)
            run_data_list.append(parsed)

        return cls(config=config, run_data_list=run_data_list)

    @classmethod
    def for_family(
        cls,
        family: str,
        outputs_root: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> "HCastAnalysis":
        """The analysis for one registered family, with every section identical.

        This is what a model-analysis notebook calls. Anything a specific
        question needs on top - an extra ablation arm, a narrower dataset set -
        goes through `overrides`, which are applied to the config after the
        profile, so the profile stays the shared definition.
        """
        try:
            profile = FAMILY_PROFILES[str(family)]
        except KeyError:
            known = ", ".join(sorted(FAMILY_PROFILES))
            raise KeyError(f"Unknown model family '{family}'. Registered families: {known}.") from None

        settings: Dict[str, Any] = {
            "family_slug": profile.slug,
            "baseline_by_dataset": profile.baseline_by_dataset(),
            "manual_runs": profile.manual_runs(),
            "aggregate_loss_keys": profile.aggregate_loss_keys,
            "excluded_seeds_by_run": dict(profile.excluded_seeds_by_run),
            "baseline_color": BASELINE_COLOR,
            "temperature_color_palette": list(RUN_COLOR_PALETTE),
            # Every family logs both decoders, so every family shows both.
        }
        if outputs_root is not None:
            settings["outputs_root"] = Path(outputs_root)
        settings.update(overrides)

        analysis = cls.from_config(HCastAnalysisConfig(**settings))
        analysis.profile = profile
        return analysis

    @classmethod
    def for_baseline_comparison(
        cls,
        families: Sequence[str] = COMPARISON_FAMILIES,
        outputs_root: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> "HCastAnalysis":
        """One run per family - each family's baseline - on the same axes.

        This is the transpose of `for_family`. There a run is an arm of one
        family and the comparison is between mechanisms; here a run is a whole
        family and the comparison is between architectures, so a dataset panel
        carries one curve per model and the sections read the same way.

        Which run is a family's baseline comes from `FAMILY_PROFILES`, together
        with the seeds that profile excludes, so a baseline redefined for a
        family notebook is redefined here in the same commit.

        The first family is the delta baseline of every test table and keeps the
        fixed `BASELINE_COLOR`; the rest take `RUN_COLOR_PALETTE` in the given
        order and hold that colour across all three datasets.

        No aggregate-objective section is defined for this analysis: `total` is
        a different functional in every family - H-CAST sums level
        cross-entropies and a global KL term, HRN a tree loss and a leaf
        cross-entropy, HT-CapsNet capsule margins - so the curves would share an
        axis without sharing a quantity. The per-level objectives are still
        per-run panels, where no cross-family axis is implied.
        """
        names = [str(family) for family in families]
        if not names:
            raise ValueError("for_baseline_comparison needs at least one family.")
        unknown = [name for name in names if name not in FAMILY_PROFILES]
        if unknown:
            known = ", ".join(sorted(FAMILY_PROFILES))
            raise KeyError(
                f"Unknown model family: {', '.join(unknown)}. Registered families: {known}."
            )

        profiles = [FAMILY_PROFILES[name] for name in names]
        base_profile = profiles[0]

        manual_runs: List[RunSpec] = []
        for position, profile in enumerate(profiles[1:]):
            color = RUN_COLOR_PALETTE[position % len(RUN_COLOR_PALETTE)]
            for slug in DATASETS:
                manual_runs.append(
                    {
                        "run_dir": profile.baseline_template.format(dataset=slug),
                        # The family name alone: in a figure where every run is a
                        # different architecture, "baseline" is true of all five
                        # and distinguishes none of them.
                        "label": profile.display_name,
                        "color": color,
                    }
                )

        excluded_seeds: Dict[str, Sequence[int]] = {}
        for profile in profiles:
            excluded_seeds.update(profile.excluded_seeds_by_run)

        settings: Dict[str, Any] = {
            "family_slug": "comparison",
            "baseline_by_dataset": {
                dataset_key: {"run_dir": spec["run_dir"], "label": base_profile.display_name}
                for dataset_key, spec in base_profile.baseline_by_dataset().items()
            },
            "manual_runs": manual_runs,
            "excluded_seeds_by_run": excluded_seeds,
            "baseline_color": BASELINE_COLOR,
            "temperature_color_palette": list(RUN_COLOR_PALETTE),
        }
        if outputs_root is not None:
            settings["outputs_root"] = Path(outputs_root)
        settings.update(overrides)

        # `profile` is left unset: this analysis belongs to no single family, and
        # a section that read one family's profile off it would be wrong here.
        return cls.from_config(HCastAnalysisConfig(**settings))

    def report_missing_runs(self) -> None:
        """Name the configured arms that produced no seed directory.

        A run matrix is written as the intended experiment, so an arm that has
        not been launched yet - or one whose directory was renamed - is listed
        rather than dropped. Without this the difference between "this family
        has no lexicographic arm" and "its lexicographic arm is not on disk"
        is invisible in the figures.
        """
        requested = []
        for spec in self.config.manual_runs:
            run_dir = spec["run_dir"] if isinstance(spec, dict) else spec
            requested.append(_resolve_run_dir(run_dir, self.config.outputs_root))
        if self.config.include_baselines:
            for spec in self.config.baseline_by_dataset.values():
                requested.append(_resolve_run_dir(spec["run_dir"], self.config.outputs_root))

        resolved = {Path(run_data["run_dir"]).resolve() for run_data in self.run_data_list}
        missing = [path for path in dict.fromkeys(requested) if path.resolve() not in resolved]
        if not missing:
            return
        print(
            f"{len(missing)} configured run(s) have no readable seed directory and are "
            "absent from every figure below:"
        )
        for path in missing:
            print(f"    {path.name}")

    @staticmethod
    def _build_runs_from_manual(config: HCastAnalysisConfig) -> List[RunData]:
        if not config.manual_runs:
            return []

        runs: List[RunData] = []
        for idx, spec in enumerate(config.manual_runs):
            normalized_spec: Dict[str, Any]
            if isinstance(spec, (str, Path)):
                normalized_spec = {"run_dir": spec}
            elif isinstance(spec, dict):
                normalized_spec = dict(spec)
            else:
                raise TypeError(f"Unsupported run spec type: {type(spec)!r}")

            run_dir = _resolve_run_dir(normalized_spec["run_dir"], config.outputs_root)
            if not run_dir.exists() or not run_dir.is_dir() or not has_seed_runs(run_dir):
                print(f"Skipping missing/manual-invalid run: {run_dir}")
                continue

            meta = _run_meta_from_dir(run_dir)
            temp = meta["temperature"]
            default_label = f"H-CAST HCC T={temp:g}" if temp is not None else run_dir.name
            has_explicit_color = ("color" in normalized_spec) and (normalized_spec.get("color") is not None)
            runs.append(
                {
                    "label": normalized_spec.get("label", default_label),
                    "run_dir": run_dir,
                    "color": normalized_spec.get(
                        "color", config.temperature_color_palette[idx % len(config.temperature_color_palette)]
                    ),
                    "color_locked": bool(has_explicit_color),
                    "temperature": temp,
                    "hcc_projection_mode": meta["hcc_projection_mode"],
                    "hcc_constraint_strength_max": meta["hcc_constraint_strength_max"],
                    "dataset_name": normalized_spec.get("dataset_name", meta["dataset_name"]),
                    "is_baseline": bool(normalized_spec.get("is_baseline", False)),
                }
            )
        return runs

    @staticmethod
    def _append_dataset_baselines(config: HCastAnalysisConfig, runs: List[RunData]) -> None:
        baseline_lookup: Dict[str, Mapping[str, Any]] = {}
        for key, value in config.baseline_by_dataset.items():
            normalized_key = _normalize_dataset_key(key)
            baseline_lookup[normalized_key] = value

        selected_dataset_keys = []
        for run in runs:
            key = _normalize_dataset_key(run.get("dataset_name"))
            if key and key not in selected_dataset_keys:
                selected_dataset_keys.append(key)

        # A dataset with no selected comparison run is still analyzable from its
        # configured baseline alone, so consider every configured dataset key.
        if config.include_baseline_only_datasets:
            baseline_only_keys = [key for key in sorted(baseline_lookup) if key not in selected_dataset_keys]
            if baseline_only_keys:
                scope = (
                    "No manual runs found; attempting baseline-only analysis for"
                    if not selected_dataset_keys
                    else "Baseline-only datasets (no selected comparison runs):"
                )
                print(f"{scope} {', '.join(baseline_only_keys)}")
            selected_dataset_keys.extend(baseline_only_keys)
        elif not selected_dataset_keys:
            selected_dataset_keys = sorted(baseline_lookup.keys())
            if selected_dataset_keys:
                print("No manual runs found; attempting baseline-only analysis from BASELINE_BY_DATASET.")

        for dataset_key in selected_dataset_keys:
            baseline_spec = baseline_lookup.get(dataset_key)
            if not baseline_spec:
                available_keys = ", ".join(sorted(baseline_lookup.keys()))
                print(f"No baseline configured for dataset key: {dataset_key}. Available keys: {available_keys}")
                continue

            baseline_run_dir = _resolve_run_dir(baseline_spec["run_dir"], config.outputs_root)
            if (
                not baseline_run_dir.exists()
                or not baseline_run_dir.is_dir()
                or not has_seed_runs(baseline_run_dir)
            ):
                print(f"Skipping missing baseline run: {baseline_run_dir}")
                continue

            baseline_resolved = baseline_run_dir.resolve()
            already_present = any(Path(run["run_dir"]).resolve() == baseline_resolved for run in runs)
            if already_present:
                for run in runs:
                    if Path(run["run_dir"]).resolve() == baseline_resolved:
                        run["is_baseline"] = True
                        # Preserve notebook-configured baseline color from semantic auto-recolor.
                        if not bool(run.get("color_locked", False)):
                            run["color"] = baseline_spec.get("color", config.baseline_color)
                            run["color_locked"] = True
                continue

            baseline_meta = _run_meta_from_dir(baseline_run_dir)
            baseline_color = baseline_spec.get("color", config.baseline_color)
            runs.append(
                {
                    "label": baseline_spec.get("label", "H-CAST"),
                    "run_dir": baseline_run_dir,
                    "color": baseline_color,
                    # Baseline color should stay user-controlled (BASELINE_COLOR or per-dataset override).
                    "color_locked": True,
                    "temperature": baseline_meta["temperature"],
                    "hcc_projection_mode": baseline_meta["hcc_projection_mode"],
                    "hcc_constraint_strength_max": baseline_meta["hcc_constraint_strength_max"],
                    "dataset_name": baseline_spec.get("dataset_name", baseline_meta["dataset_name"]),
                    "is_baseline": True,
                }
            )

    def _stem(self, dataset_key: str, section: str, suffix: str = "") -> str:
        """Figure file name: <family>_<dataset>_<section>, e.g. hcast_cifar100_gradient_norms."""
        parts = [str(self.config.family_slug), dataset_slug(dataset_key), str(section)]
        if suffix:
            parts.append(str(suffix))
        return "_".join(part for part in parts if part)

    def print_run_summary(self) -> None:
        self.report_missing_runs()
        for dataset_key in self.dataset_keys:
            for run_data in self.run_data_by_dataset[dataset_key]:
                epoch_count = len(run_data["epoch_events"])
                best_events = run_data.get("best_epoch_events", {})
                best_td = best_events.get("topdown") if isinstance(best_events, MappingABC) else run_data.get("best_epoch_event")
                best_ind = best_events.get("independent") if isinstance(best_events, MappingABC) else run_data.get("best_epoch_event")
                best_td_epoch = best_td["epoch"] if best_td is not None else None
                best_ind_epoch = best_ind["epoch"] if best_ind is not None else None
                temperature = run_data.get("temperature", None)
                projection_mode = run_data.get("hcc_projection_mode", None)
                strength_max = run_data.get("hcc_constraint_strength_max", None)
                dataset_name = run_data.get("dataset_name", "unknown")
                is_baseline = bool(run_data.get("is_baseline", False))
                seeds = list(run_data.get("seeds", []))

                temp_txt = "" if temperature is None else f", T={temperature:g}"
                mode_txt = "" if projection_mode is None else f", proj_mode={projection_mode}"
                if strength_max is None or not np.isfinite(strength_max):
                    strength_txt = ""
                else:
                    strength_txt = f", strength_max={strength_max:g}"
                baseline_txt = ", baseline" if is_baseline else ""
                seeds_txt = f", seeds={seeds}" if seeds else ""

                modes = decoder_modes(self.config.decoder_view)
                best_parts = []
                if "topdown" in modes:
                    best_parts.append(f"best_td_epoch={best_td_epoch}")
                if "independent" in modes:
                    best_parts.append(f"best_ind_epoch={best_ind_epoch}")
                best_txt = ", ".join(best_parts)

                print(
                    f"[{dataset_name}] {run_data['label']}: epochs={epoch_count}, "
                    f"{best_txt}{temp_txt}{mode_txt}{strength_txt}{baseline_txt}{seeds_txt}"
                )

    def plot_validation_curves(
        self,
        metric_families: Optional[Sequence[Tuple[str, str, bool]]] = None,
        mode_specs: Optional[Sequence[Tuple[str, str, str, str]]] = None,
        show_best_errorbars: bool = False,
        decoder_view: Optional[str] = None,
    ) -> None:
        view = _normalize_decoder_view(
            decoder_view if decoder_view is not None else self.config.decoder_view
        )
        metric_families = metric_families or [
            ("fpa", "Validation FPA (%)", True),
            ("weighted_ap", "Validation wAP (%)", True),
            ("tice", "Validation TICE (%)", True),
            ("ahd", "Validation AHD (edges)", False),
        ]
        mode_specs = (
            _default_mode_specs(view)
            if mode_specs is None
            else _filter_mode_specs(mode_specs, view)
        )

        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

            legend_labels = [
                f"{run_data['label']} ({mode_label})"
                for run_data in dataset_runs
                for _mode_key, _style, mode_label, _marker in mode_specs
            ]
            ncols = _panel_columns(len(metric_families))
            pages = list(_grid_pages(len(metric_families), ncols, legend_labels))
            for page_idx, (fig, axes, start, stop) in enumerate(pages, start=1):
                page_families = metric_families[start:stop]
                for ax, (metric_prefix, metric_title, is_percent) in zip(axes, page_families):
                    for run_data in dataset_runs:
                        for mode_key, line_style, mode_label, marker in mode_specs:
                            metric_key = f"{metric_prefix}_{mode_key}"
                            epochs, values, stds, counts = metric_series_with_std(
                                run_data["epoch_events"], metric_key
                            )
                            plot_values = values * 100.0 if is_percent else values
                            ax.plot(
                                epochs,
                                plot_values,
                                label=f"{run_data['label']} ({mode_label})",
                                color=run_data["color"],
                                linestyle=line_style,
                                linewidth=CURVE_LINEWIDTH,
                            )
                            plot_stds = stds * 100.0 if is_percent else stds
                            band_mask = np.isfinite(plot_values) & np.isfinite(plot_stds) & (counts > 1)
                            if np.any(band_mask):
                                ax.fill_between(
                                    epochs,
                                    plot_values - plot_stds,
                                    plot_values + plot_stds,
                                    where=band_mask,
                                    color=run_data["color"],
                                    alpha=0.14,
                                    linewidth=0,
                                )

                            best_events = run_data.get("best_epoch_events", {})
                            best_event = (
                                best_events.get(mode_key)
                                if isinstance(best_events, MappingABC)
                                else run_data.get("best_epoch_event")
                            )
                            if best_event is not None:
                                best_value = float(best_event["val_metrics_norm"].get(metric_key, np.nan))
                                if np.isfinite(best_value):
                                    best_plot_value = best_value * 100.0 if is_percent else best_value
                                    ax.scatter(
                                        [best_event["epoch"]],
                                        [best_plot_value],
                                        color=run_data["color"],
                                        marker=marker,
                                        s=14,
                                        zorder=4,
                                    )
                                    best_epoch_std = float(best_event.get("epoch_std", np.nan))
                                    best_value_std = float(
                                        best_event.get("val_metrics_norm_std", {}).get(metric_key, np.nan)
                                    )
                                    if show_best_errorbars and int(best_event.get("epoch_count", 0)) > 1:
                                        yerr = best_value_std * 100.0 if is_percent else best_value_std
                                        ax.errorbar(
                                            [best_event["epoch"]],
                                            [best_plot_value],
                                            xerr=[[best_epoch_std], [best_epoch_std]]
                                            if np.isfinite(best_epoch_std)
                                            else None,
                                            yerr=[[yerr], [yerr]] if np.isfinite(yerr) else None,
                                            color=run_data["color"],
                                            linewidth=1.0,
                                            capsize=2,
                                            alpha=0.8,
                                            zorder=3,
                                        )

                    # No y label: the title already carries the metric and its
                    # unit, so "Score (%)" under "Validation FPA (%)" is a
                    # repetition that costs a gutter on every panel.
                    ax.set_title(_wrap_title(metric_title, ncols))

                _close_unused_axes(axes, len(page_families), "Epoch")
                handles, labels = axes[0].get_legend_handles_labels()
                _add_bottom_legend(fig, handles, labels)
                decoders = decoder_view_phrase(view)
                suffix = f"part{page_idx}" if len(pages) > 1 else ""
                _finish(
                    fig,
                    self._stem(dataset_key, "validation_curves", suffix),
                    f"{dataset_display_name(dataset_key)}: validation metrics ({decoders} decoding)",
                )

    def plot_projection_diagnostics(
        self,
        base_diag_specs: Optional[Sequence[Union[Tuple[str, str, bool], Tuple[str, str, bool, str]]]] = None,
        section_stem: str = "mechanism_diagnostics",
        section_title: str = "output-space constraint diagnostics",
        decoder_view: Optional[str] = None,
    ) -> None:
        # Default to the shared spec so every family's section is the same
        # section; pass `base_diag_specs` only for a family-specific appendix.
        view = _normalize_decoder_view(
            decoder_view if decoder_view is not None else self.config.decoder_view
        )
        base_diag_specs = base_diag_specs or MECHANISM_DIAGNOSTIC_SPECS

        diag_specs: List[Tuple[str, str, bool, str]] = []
        for spec in base_diag_specs:
            if len(spec) == 3:
                metric_key, title, is_percent = spec
                source = "val"
            elif len(spec) == 4:
                metric_key, title, is_percent, source = spec
            else:
                raise ValueError(
                    "Projection diagnostic specs must be tuples of length 3 "
                    "(metric_key, title, is_percent) or 4 (+source)."
                )
            source_norm = str(source).strip().lower()
            if source_norm not in {"val", "train"}:
                raise ValueError(
                    "Projection diagnostic source must be 'val' or 'train': "
                    f"got '{source}'."
                )
            if not _include_metric_for_view(str(metric_key), view):
                continue
            diag_specs.append((str(metric_key), str(title), bool(is_percent), source_norm))

        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

            # Two filters, both stated rather than applied silently.
            #
            # A run without the mechanism still writes several of these keys as
            # an exact 0.0 - "nothing was rewritten" - and a binary switch held
            # on for a whole run writes an exact 1.0 at every epoch. Neither is
            # a curve, so a spec earns a panel only when its value actually
            # varies somewhere in the selected runs. The constant is named in
            # the message, so "held at 1.0" still confirms the mechanism was
            # active without spending a panel on a flat line.
            #
            # Then: two specs that trace the identical curve in every selected
            # run are one panel's worth of information, so the second is
            # dropped. This is what removes `..._post_l2` where nothing rewrote
            # the scores and it is a copy of `..._pre_l2`, and the residual
            # reduction where the post-projection residual is at numerical zero
            # and the reduction is therefore the residual before it.
            name = dataset_display_name(dataset_key)
            active_diag_specs: List[Tuple[str, str, bool, str]] = []
            active_curves: List[List[np.ndarray]] = []
            constant_notes: List[str] = []
            duplicate_notes: List[str] = []
            for metric_key, title, is_percent, source in diag_specs:
                curves = []
                for run_data in dataset_runs:
                    if source == "train":
                        _, values = get_train_metric_series(run_data["epoch_events"], metric_key)
                    else:
                        _, values = get_metric_series(run_data["epoch_events"], metric_key)
                    curves.append(values)
                finite = np.concatenate([values[np.isfinite(values)] for values in curves]) if curves else np.array([])
                if not finite.size:
                    continue

                unique_values = np.unique(finite)
                if unique_values.size == 1:
                    constant_notes.append(f"{title} (held at {unique_values[0]:g})")
                    continue

                duplicate_of = next(
                    (
                        kept_title
                        for (_key, kept_title, _pct, _src), kept_curves in zip(active_diag_specs, active_curves)
                        if all(
                            np.array_equal(a, b) or np.allclose(a, b, equal_nan=True)
                            for a, b in zip(curves, kept_curves)
                        )
                    ),
                    None,
                )
                if duplicate_of is not None:
                    duplicate_notes.append(f"{title} (identical to {duplicate_of})")
                    continue

                active_diag_specs.append((metric_key, title, is_percent, source))
                active_curves.append(curves)

            if constant_notes:
                print(
                    f"[{name}] constant in every selected run, so not plotted: "
                    f"{'; '.join(constant_notes)}."
                )
            if duplicate_notes:
                print(
                    f"[{name}] duplicates a panel already on this page, so not plotted: "
                    f"{'; '.join(duplicate_notes)}."
                )
            if not active_diag_specs:
                print(
                    f"[{name}] no output-space diagnostic varies: none of the "
                    "selected runs applied the mechanism."
                )
                continue

            ncols = _panel_columns(len(active_diag_specs))
            run_labels = [str(run_data["label"]) for run_data in dataset_runs]
            pages = list(_grid_pages(len(active_diag_specs), ncols, run_labels))
            for page_idx, (fig, axes, start, stop) in enumerate(pages, start=1):
                page_specs = active_diag_specs[start:stop]
                for ax, (metric_key, title, is_percent, source) in zip(axes, page_specs):
                    for run_data in dataset_runs:
                        if source == "train":
                            epochs, values = get_train_metric_series(run_data["epoch_events"], metric_key)
                        else:
                            epochs, values = get_metric_series(run_data["epoch_events"], metric_key)
                        plot_values = values * 100.0 if is_percent else values
                        ax.plot(
                            epochs,
                            plot_values,
                            label=run_data["label"],
                            color=run_data["color"],
                            linewidth=CURVE_LINEWIDTH,
                        )

                        if source == "val":
                            best_events = run_data.get("best_epoch_events", {})
                            best_event = (
                                best_events.get("topdown")
                                if isinstance(best_events, MappingABC)
                                else run_data.get("best_epoch_event")
                            )
                            if best_event is not None:
                                best_value = float(best_event["val_metrics_norm"].get(metric_key, np.nan))
                                if np.isfinite(best_value):
                                    best_plot_value = best_value * 100.0 if is_percent else best_value
                                    ax.scatter(
                                        [best_event["epoch"]],
                                        [best_plot_value],
                                        color=run_data["color"],
                                        marker="o",
                                        s=14,
                                        zorder=4,
                                    )

                    # As in the validation-curve figure, the title carries the
                    # quantity and its unit and there is no y label: "Value"
                    # under a named panel says nothing and costs a gutter on a
                    # half-width panel.
                    panel_title = f"{title} (%)" if is_percent else title
                    if source == "train":
                        panel_title = f"{panel_title} (train)"
                    ax.set_title(_wrap_title(panel_title, ncols))

                _close_unused_axes(axes, len(page_specs), "Epoch")
                handles, labels = axes[0].get_legend_handles_labels()
                _add_bottom_legend(fig, handles, labels)
                suffix = f"part{page_idx}" if len(pages) > 1 else ""
                _finish(
                    fig,
                    self._stem(dataset_key, section_stem, suffix),
                    f"{dataset_display_name(dataset_key)}: {section_title}",
                )

    def plot_training_losses(
        self,
        aggregate_loss_keys: Optional[Sequence[str]] = None,
    ) -> None:
        aggregate_loss_keys = (
            aggregate_loss_keys
            or self.config.aggregate_loss_keys
            or ["total", "ce", "reg", "kl", "level_ce", "gk_loss"]
        )

        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

            level_loss_ids = sorted(
                {
                    int(key.rsplit("_", 1)[-1])
                    for run_data in dataset_runs
                    for event in run_data["epoch_events"]
                    for key in event.get("train_losses", {}).keys()
                    if key.startswith("loss_level_") and key.rsplit("_", 1)[-1].isdigit()
                }
            )
            # Aggregate objectives and the per-level terms are separate figures:
            # families such as HRN expose eight aggregate losses plus three level
            # losses, which is more than one page at a readable panel height.
            aggregate_specs = [(key, loss_display_name(key)) for key in aggregate_loss_keys]
            level_specs = []
            for level_idx in level_loss_ids:
                level_name = get_level_label(level_idx, dataset_runs[0]).capitalize()
                level_specs.append(
                    (
                        f"loss_level_{level_idx}",
                        f"{level_name} objective ({level_loss_symbol(level_idx)})",
                    )
                )

            for specs, stem, title in (
                (aggregate_specs, "training_losses_aggregate", "terms of the training objective"),
                (level_specs, "training_losses_per_level", "per-level training objectives"),
            ):
                # Several families log one quantity under two keys - HRN writes
                # `hier_loss`/`tree_loss` and `ce_loss_leaf`/`fine_ce` for the
                # same value, so a launcher lists both to cover older runs. The
                # raw keys hid that; the readable names do not, so the second
                # spelling of a name already on the page is dropped. Presence is
                # tested first, so it is the alias that carries data that stays.
                active_specs = []
                seen_titles = set()
                for spec in specs:
                    has_data = any(
                        np.any(np.isfinite(get_train_loss_series(run_data["epoch_events"], spec[0])[1]))
                        for run_data in dataset_runs
                    )
                    if not has_data or spec[1] in seen_titles:
                        continue
                    seen_titles.add(spec[1])
                    active_specs.append(spec)
                if not active_specs:
                    continue

                ncols = _panel_columns(len(active_specs))
                run_labels = [str(run_data["label"]) for run_data in dataset_runs]
                pages = list(_grid_pages(len(active_specs), ncols, run_labels))
                for page_idx, (fig, axes, start, stop) in enumerate(pages, start=1):
                    page_specs = active_specs[start:stop]
                    for ax, (metric_key, metric_title) in zip(axes, page_specs):
                        for run_data in dataset_runs:
                            epochs, values = get_train_loss_series(run_data["epoch_events"], metric_key)
                            ax.plot(
                                epochs,
                                values,
                                label=run_data["label"],
                                color=run_data["color"],
                                linewidth=CURVE_LINEWIDTH,
                            )
                        ax.set_title(_wrap_title(metric_title, ncols))

                    _close_unused_axes(axes, len(page_specs), "Epoch")
                    _label_left_column(axes, len(page_specs), "Loss")
                    handles, labels = axes[0].get_legend_handles_labels()
                    _add_bottom_legend(fig, handles, labels)
                    suffix = f"part{page_idx}" if len(pages) > 1 else ""
                    _finish(
                        fig,
                        self._stem(dataset_key, stem, suffix),
                        f"{dataset_display_name(dataset_key)}: {title}",
                    )

    def plot_level_loss_weights(self) -> None:
        """The weight each level objective actually carried, epoch by epoch.

        Only some families weight their level losses, and those that do can
        weight them dynamically: HT-CapsNet's native schedule recomputes the
        weights from the next batch, while its lexicographic launcher sets
        `model.loss.weight_mode: none` and leaves them at one. Any comparison
        between those two arms therefore mixes the mechanism under test with a
        change in level weighting, and this section is what makes that second
        variable visible instead of implicit.

        Families that log no `loss_weight_level_*` key say so and are skipped,
        so the section can stand in every notebook.
        """
        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

            level_ids = sorted(
                {
                    int(key.rsplit("_", 1)[-1])
                    for run_data in dataset_runs
                    for event in run_data["epoch_events"]
                    for key in event.get("train_losses", {}).keys()
                    if key.startswith("loss_weight_level_") and key.rsplit("_", 1)[-1].isdigit()
                }
            )
            name = dataset_display_name(dataset_key)
            if not level_ids:
                print(f"[{name}] no level-loss weights logged in the selected runs.")
                continue

            ncols = _panel_columns(len(level_ids))
            run_labels = [str(run_data["label"]) for run_data in dataset_runs]
            pages = list(_grid_pages(len(level_ids), ncols, run_labels))
            for page_idx, (fig, axes, start, stop) in enumerate(pages, start=1):
                page_levels = level_ids[start:stop]
                for ax, level_idx in zip(axes, page_levels):
                    metric_key = f"loss_weight_level_{level_idx}"
                    for run_data in dataset_runs:
                        epochs, means, stds, counts = metric_series_with_std(
                            run_data["epoch_events"], metric_key, source="train_losses"
                        )
                        ax.plot(
                            epochs,
                            means,
                            label=run_data["label"],
                            color=run_data["color"],
                            linewidth=CURVE_LINEWIDTH,
                        )
                        band = np.isfinite(means) & np.isfinite(stds) & (counts > 1)
                        if np.any(band):
                            ax.fill_between(
                                epochs,
                                means - stds,
                                means + stds,
                                where=band,
                                color=run_data["color"],
                                alpha=0.14,
                                linewidth=0,
                            )
                    level_label = get_level_label(level_idx, dataset_runs[0]).capitalize()
                    ax.set_title(
                        _wrap_title(
                            f"{level_label} objective ({level_loss_symbol(level_idx)})",
                            ncols,
                        )
                    )

                _close_unused_axes(axes, len(page_levels), "Epoch")
                _label_left_column(axes, len(page_levels), "Weight")
                handles, labels = axes[0].get_legend_handles_labels()
                _add_bottom_legend(fig, handles, labels)
                suffix = f"part{page_idx}" if len(pages) > 1 else ""
                _finish(
                    fig,
                    self._stem(dataset_key, "level_loss_weights", suffix),
                    f"{name}: effective level-loss weights",
                )

    def plot_per_run_per_level_training_losses(self) -> None:
        """One panel per run, with that run's level objectives inside the panel.

        The previous section compares one objective across runs; this one keeps
        the three levels of a single run together, which is where a run that
        trades shallow loss for deep loss becomes visible.
        """
        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

            plottable = []
            for run_data in dataset_runs:
                level_loss_ids = sorted(
                    {
                        int(key.rsplit("_", 1)[-1])
                        for event in run_data["epoch_events"]
                        for key in event.get("train_losses", {}).keys()
                        if key.startswith("loss_level_") and key.rsplit("_", 1)[-1].isdigit()
                    }
                )
                level_loss_ids = [
                    level_idx
                    for level_idx in level_loss_ids
                    if np.any(
                        np.isfinite(
                            get_train_loss_series(run_data["epoch_events"], f"loss_level_{level_idx}")[1]
                        )
                    )
                ]
                if level_loss_ids:
                    plottable.append((run_data, level_loss_ids))
                else:
                    print(
                        f"[{dataset_display_name(dataset_key)}] no per-level training losses "
                        f"for {run_data['label']}"
                    )

            if not plottable:
                continue

            ncols = _panel_columns(len(plottable))
            level_labels = [
                level_objective_label(level_idx, plottable[0][0])
                for level_idx in plottable[0][1]
            ]
            pages = list(_grid_pages(len(plottable), ncols, level_labels))
            for page_idx, (fig, axes, start, stop) in enumerate(pages, start=1):
                page_runs = plottable[start:stop]
                for ax, (run_data, level_loss_ids) in zip(axes, page_runs):
                    for position, level_idx in enumerate(level_loss_ids):
                        epochs, values = get_train_loss_series(
                            run_data["epoch_events"], f"loss_level_{level_idx}"
                        )
                        ax.plot(
                            epochs,
                            values,
                            label=level_objective_label(level_idx, run_data),
                            color=LEVEL_RAMP[min(position, len(LEVEL_RAMP) - 1)],
                            linewidth=CURVE_LINEWIDTH,
                        )
                    ax.set_title(_wrap_title(str(run_data["label"]), ncols))

                _close_unused_axes(axes, len(page_runs), "Epoch")
                _label_left_column(axes, len(page_runs), "Loss")
                handles, labels = axes[0].get_legend_handles_labels()
                _add_bottom_legend(fig, handles, labels)
                suffix = f"part{page_idx}" if len(pages) > 1 else ""
                _finish(
                    fig,
                    self._stem(dataset_key, "level_losses_per_run", suffix),
                    f"{dataset_display_name(dataset_key)}: level objectives within each run",
                )

    # ----------------------------------------------------------------- #
    # Gradient and parameter diagnostics
    # ----------------------------------------------------------------- #
    #
    # Everything below is indexed by the exact gradient-support blocks of
    # docs/04-methodology.tex, not by a single family's parameter names, because
    # which blocks exist is a property of the architecture: H-CAST branches
    # fine-to-coarse and has P123/P12/P1, HT-CapsNet runs the reverse cascade and
    # has P123/P23/P3, HRN and Hier-COS share one network and have P123 alone.
    # The blocks are discovered from the logs so each family shows only the ones
    # it actually has, instead of an empty panel or - worse - the zero that an
    # empty block writes into its cosine key.

    def discover_gradient_blocks(self, dataset_key: str) -> Dict[str, List[str]]:
        """Blocks with logged gradient norms in this dataset's runs, and their levels.

        A block counts as present only when at least one of its
        ``grad_norm_<block>_<level>`` keys carries a finite value. Cosine and
        parameter panels are then built only for blocks that pass this test,
        which is what keeps the zero that an empty mask writes into
        ``cos_p12_mid_coarse`` out of the figures.
        """
        dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
        present: Dict[str, List[str]] = {}
        for block in GRADIENT_BLOCK_ORDER:
            levels = []
            for level in BLOCK_LEVELS[block]:
                keys = block_metric_keys(grad_norm_key(block, level))
                if any(
                    np.any(np.isfinite(get_train_metric_series_any(run_data["epoch_events"], keys)[1]))
                    for run_data in dataset_runs
                ):
                    levels.append(level)
            if levels:
                present[block] = levels
        return present

    def _configured_gradient_blocks(self, dataset_key: str) -> Dict[str, Tuple[str, ...]]:
        """`train.gradient_blocks` per run, read from `config_resolved.yaml`."""
        configured: Dict[str, Tuple[str, ...]] = {}
        for run_data in self.run_data_by_dataset.get(dataset_key, []):
            run_dir = run_data.get("run_dir")
            if run_dir is None:
                continue
            cfg = load_yaml(_metadata_seed_dir(Path(run_dir)) / "config_resolved.yaml")
            blocks = ((cfg.get("train") or {}) if isinstance(cfg, dict) else {}).get("gradient_blocks")
            if isinstance(blocks, (list, tuple)) and blocks:
                configured[str(run_data["label"])] = tuple(str(block) for block in blocks)
        return configured

    def report_gradient_blocks(self) -> None:
        """State which blocks the logs carry, and where that differs from the configs.

        Runs launched before the canonical block keys existed wrote three blocks
        under H-CAST's trunk names, and a launcher whose block selection was
        widened after those runs finished will not agree with its own logs. Both
        cases are printed rather than silently rendered as a missing section.
        """
        for dataset_key in self.dataset_keys:
            logged = self.discover_gradient_blocks(dataset_key)
            name = dataset_display_name(dataset_key)
            if not logged:
                print(f"[{name}] no gradient diagnostics logged in the selected runs.")
                continue

            summary = ", ".join(
                f"{block_label(block, tex=False)} ({', '.join(level_display_name(l) for l in levels)})"
                for block, levels in logged.items()
            )
            print(f"[{name}] logged blocks: {summary}")

            for label, configured in self._configured_gradient_blocks(dataset_key).items():
                missing = [block for block in configured if block not in logged]
                if missing:
                    print(
                        f"    {label}: config selects "
                        f"{', '.join(block_label(b, tex=False) for b in configured)}, but "
                        f"{', '.join(block_label(b, tex=False) for b in missing)} is absent from the "
                        "logs - that block is either empty for this architecture or the run "
                        "predates the selection."
                    )

    def _gradient_panels(
        self, dataset_key: str
    ) -> Tuple[Dict[str, List[Tuple[str, Tuple[str, ...], str]]], List[str]]:
        """Panel specs per figure group, plus notes about what was left out.

        Returns ``(panels_by_group, notes)`` where a panel is
        ``(title, candidate metric keys, kind)``.
        """
        logged = self.discover_gradient_blocks(dataset_key)
        dataset_runs = self.run_data_by_dataset.get(dataset_key, [])

        def is_present(keys: Sequence[str]) -> bool:
            return any(
                np.any(np.isfinite(get_train_metric_series_any(run_data["epoch_events"], keys)[1]))
                for run_data in dataset_runs
            )

        def flag_state(run_data: RunData, keys: Sequence[str]) -> str:
            """'always', 'never' or 'varies' within one run.

            The code emits a flag for every projection step the operator defines,
            including the ones the active priority order never performs, so a
            flag pinned at 0 means "not part of this priority order" and one
            pinned at 1 means "ran at every epoch". Which steps a run performs is
            a property of its priority mode and belongs in a sentence; only a
            flag that changes over epochs - a projection skipped because its
            reference gradient went degenerate - has anything for a curve to show.
            """
            _, values = get_train_metric_series_any(run_data["epoch_events"], keys)
            finite = values[np.isfinite(values)]
            if not finite.size:
                return "absent"
            if np.min(finite) >= 0.999:
                return "always"
            if np.max(finite) <= 0.001:
                return "never"
            return "varies"

        norms: List[Tuple[str, Tuple[str, ...], str]] = []
        alignment: List[Tuple[str, Tuple[str, ...], str]] = []
        parameters: List[Tuple[str, Tuple[str, ...], str]] = []
        projection_alignment: List[Tuple[str, Tuple[str, ...], str]] = []
        projection_applied: List[Tuple[str, Tuple[str, ...], str]] = []
        notes: List[str] = []
        steps_per_run: Dict[str, List[str]] = {}

        for block, levels in logged.items():
            for level in levels:
                norms.append(
                    (
                        gradient_panel_title("grad_norm", block, level),
                        block_metric_keys(grad_norm_key(block, level)),
                        "norm",
                    )
                )

            for target, reference in CANONICAL_COSINE_PAIRS:
                if target not in levels or reference not in levels:
                    continue
                keys = block_metric_keys(cosine_key(block, target, reference))
                if is_present(keys):
                    alignment.append(
                        (gradient_panel_title("cos", block, target, reference), keys, "cosine")
                    )
            higher_keys = block_metric_keys(cosine_key(block, "fine", HIGHER_REFERENCE))
            if "fine" in levels and len(levels) >= 3 and is_present(higher_keys):
                alignment.append(
                    (
                        gradient_panel_title("cos", block, "fine", HIGHER_REFERENCE),
                        higher_keys,
                        "cosine",
                    )
                )

            # Parameter norm and per-epoch movement sit side by side on one row,
            # so the scale of a block and how far it moved are read together.
            parameters.append(
                (
                    gradient_panel_title("param_norm", block),
                    block_metric_keys(f"param_norm_{block}"),
                    "norm",
                )
            )
            parameters.append(
                (
                    gradient_panel_title("delta_param_norm", block),
                    block_metric_keys(f"delta_param_norm_{block}"),
                    "norm",
                )
            )

            # Lex arms only: the projected cosine against its pre-projection
            # value, plus whether the projection step ran at all.
            for target, reference in (*CANONICAL_COSINE_PAIRS, ("fine", HIGHER_REFERENCE)):
                if target not in levels:
                    continue
                if reference != HIGHER_REFERENCE and reference not in levels:
                    continue
                post_keys = post_cosine_keys(block, target, reference)
                if not is_present(post_keys):
                    continue
                # No ", after projection" suffix here: the panel carries both
                # curves and the legend is what separates before from after.
                projection_alignment.append(
                    (
                        gradient_panel_title("cos", block, target, reference),
                        post_keys,
                        "cosine_prepost",
                    )
                )

            # Only a flag that changes over epochs gets a panel. Which steps a
            # priority order performs is constant within a run, so it is
            # collected per run and stated as text below the section.
            for target, reference in (
                ("mid", "coarse"),
                ("fine", HIGHER_REFERENCE),
                ("coarse", HIGHER_REFERENCE),
                ("mid", "fine"),
                ("coarse", "mid"),
            ):
                flag_keys = post_projection_flag_keys(block, target, reference)
                if not is_present(flag_keys):
                    continue
                step = f"{block_label(block, tex=False)} {_level_phrase(target)} against {_level_phrase(reference)}"
                states = {
                    str(run_data["label"]): flag_state(run_data, flag_keys)
                    for run_data in dataset_runs
                }
                if any(state == "varies" for state in states.values()):
                    projection_applied.append(
                        (
                            gradient_panel_title("projection_applied", block, target),
                            flag_keys,
                            "flag",
                        )
                    )
                    continue
                for label, state in states.items():
                    if state == "always":
                        steps_per_run.setdefault(label, []).append(step)

        if steps_per_run:
            notes.append("projection steps performed at every epoch:")
            for label, steps in steps_per_run.items():
                notes.append(f"    {label}: {'; '.join(steps)}")

        return (
            {
                "gradient_norms": norms,
                "gradient_alignment": alignment,
                "parameter_movement": parameters,
                "projection_alignment": projection_alignment,
                "projection_applied": projection_applied,
            },
            notes,
        )

    def plot_gradient_diagnostics(
        self,
        smooth_window: int = 2,
        n_cols: Optional[int] = None,
        show_raw_overlay: bool = False,
        raw_alpha: float = 0.16,
        log_scale_for_norms: bool = False,
        groups: Optional[Sequence[str]] = None,
    ) -> None:
        """Up to five figures per dataset, all indexed by exact gradient-support block.

        - ``gradient_norms``: how hard each level pushes on each shared block.
        - ``gradient_alignment``: pairwise cosines, so conflict is signed.
        - ``parameter_movement``: block scale and per-epoch movement.
        - ``projection_alignment``: lex arms only, each projected cosine against
          its own pre-projection value.
        - ``projection_applied``: lex arms only, and only for the projection
          steps that were ever skipped for numerical safety.
        """
        group_titles = {
            "gradient_norms": "level gradient norms on the shared blocks",
            "gradient_alignment": "pairwise gradient alignment",
            "parameter_movement": "parameter scale and per-epoch movement",
            "projection_alignment": "gradient alignment before and after projection",
            "projection_applied": "projection steps skipped for numerical safety",
        }
        wanted = tuple(groups) if groups else tuple(group_titles)

        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

            panels_by_group, notes = self._gradient_panels(dataset_key)
            name = dataset_display_name(dataset_key)
            if not any(panels_by_group[group] for group in wanted):
                print(f"[{name}] no gradient or parameter diagnostics in the selected runs.")
                continue
            for note in notes:
                print(f"[{name}] {note}")

            for group in wanted:
                panels = panels_by_group[group]
                if not panels:
                    if group == "projection_alignment":
                        print(
                            f"[{name}] no post-projection diagnostics: none of the selected runs "
                            "were trained with train.lexicographic.enabled."
                        )
                    continue

                # Parameter panels pair (norm, movement) per block, so they
                # only read correctly at two columns.
                ncols = 2 if group == "parameter_movement" else (
                    _panel_columns(len(panels)) if n_cols is None else max(1, int(n_cols))
                )
                is_prepost = group == "projection_alignment"
                legend_labels = [
                    f"{run_data['label']}{suffix}"
                    for run_data in dataset_runs
                    for suffix in ((" (after)", " (before)") if is_prepost else ("",))
                ]

                pages = list(_grid_pages(len(panels), ncols, legend_labels))
                for page_idx, (fig, axes, start, stop) in enumerate(pages, start=1):
                    page_panels = panels[start:stop]
                    for ax, (title, metric_keys, kind) in zip(axes, page_panels):
                        self._draw_gradient_panel(
                            ax=ax,
                            dataset_runs=dataset_runs,
                            title=title,
                            metric_keys=metric_keys,
                            kind=kind,
                            smooth_window=smooth_window,
                            show_raw_overlay=show_raw_overlay,
                            raw_alpha=raw_alpha,
                            log_scale_for_norms=log_scale_for_norms,
                            is_parameter_group=(group == "parameter_movement"),
                            ncols=ncols,
                        )

                    _close_unused_axes(axes, len(page_panels), "Epoch")
                    _label_left_column(axes, len(page_panels), _GRADIENT_Y_LABELS[group])
                    handles, labels = self._gradient_legend(axes[: len(page_panels)])
                    _add_bottom_legend(fig, handles, labels)
                    suffix = f"part{page_idx}" if len(pages) > 1 else ""
                    _finish(
                        fig,
                        self._stem(dataset_key, group, suffix),
                        f"{name}: {group_titles[group]}",
                    )

    @staticmethod
    def _gradient_legend(axes: Sequence[Any]) -> Tuple[List[Any], List[str]]:
        """Collect handles across a page, since not every panel carries every run."""
        handles: List[Any] = []
        labels: List[str] = []
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        return handles, labels

    def _draw_gradient_panel(
        self,
        ax: Any,
        dataset_runs: Sequence[RunData],
        title: str,
        metric_keys: Sequence[str],
        kind: str,
        smooth_window: int,
        show_raw_overlay: bool,
        raw_alpha: float,
        log_scale_for_norms: bool,
        is_parameter_group: bool,
        ncols: int = 2,
    ) -> None:
        is_prepost = kind == "cosine_prepost"
        pre_keys: Tuple[str, ...] = ()
        if is_prepost:
            pre_keys = _pre_keys_for_post(metric_keys)

        drew_any = False
        for run_data in dataset_runs:
            epochs, values = get_train_metric_series_any(run_data["epoch_events"], metric_keys)
            if not np.any(np.isfinite(values)):
                continue
            drew_any = True

            style = (
                {"linestyle": "-" if run_data.get("is_baseline") else "--", "linewidth": CURVE_LINEWIDTH}
                if is_parameter_group
                else {"linestyle": "-", "linewidth": CURVE_LINEWIDTH}
            )
            smoothed = moving_average_ignore_nan(values, smooth_window)
            label = f"{run_data['label']} (after)" if is_prepost else run_data["label"]
            ax.plot(epochs, smoothed, label=label, color=run_data["color"], **style)

            if show_raw_overlay and int(smooth_window) > 1:
                ax.plot(epochs, values, color=run_data["color"], linewidth=RAW_OVERLAY_LINEWIDTH, alpha=raw_alpha)

            if is_prepost and pre_keys:
                pre_epochs, pre_values = get_train_metric_series_any(
                    run_data["epoch_events"], pre_keys
                )
                if np.any(np.isfinite(pre_values)):
                    ax.plot(
                        pre_epochs,
                        moving_average_ignore_nan(pre_values, smooth_window),
                        label=f"{run_data['label']} (before)",
                        color=run_data["color"],
                        linestyle=BEFORE_TRACE_DASHES,
                        linewidth=CURVE_LINEWIDTH_SECONDARY,
                    )

        # The y label goes on the leftmost column only, once the page is built;
        # every panel of a group measures the same quantity.
        ax.set_title(_wrap_title(title, ncols))
        if kind in {"cosine", "cosine_prepost"}:
            ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.7, zorder=1)
            ax.set_ylim(-1.05, 1.05)
        elif kind == "flag":
            ax.set_ylim(-0.08, 1.08)
            ax.set_yticks([0.0, 1.0], ["skipped", "applied"])
        else:
            if drew_any and log_scale_for_norms:
                cache = [
                    (run_data, {metric_keys[0]: get_train_metric_series_any(run_data["epoch_events"], metric_keys)})
                    for run_data in dataset_runs
                ]
                robust_range = get_positive_robust_range(cache, metric_keys[0])
                if robust_range is not None:
                    ax.set_yscale("log")
                    ax.set_ylim(*robust_range)

    def plot_per_level_validation_accuracy(
        self,
        mode_specs: Optional[Sequence[Tuple[str, str, str, str]]] = None,
        show_best_errorbars: bool = False,
        decoder_view: Optional[str] = None,
    ) -> None:
        view = _normalize_decoder_view(
            decoder_view if decoder_view is not None else self.config.decoder_view
        )
        mode_specs = (
            _default_mode_specs(view)
            if mode_specs is None
            else _filter_mode_specs(mode_specs, view)
        )

        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

            all_level_ids = sorted(
                {
                    int(key.rsplit("_", 1)[-1])
                    for run_data in dataset_runs
                    for event in run_data["epoch_events"]
                    for key in event["val_metrics_norm"].keys()
                    if (
                        _include_metric_for_view(key, view)
                        and
                        (key.startswith("acc_level_topdown_") or key.startswith("acc_level_independent_"))
                        and key.rsplit("_", 1)[-1].isdigit()
                    )
                }
            )

            if not all_level_ids:
                print(f"[{dataset_key}] No per-level validation accuracy metrics found in selected runs.")
                continue

            legend_labels = [
                f"{run_data['label']} ({mode_label})"
                for run_data in dataset_runs
                for _mode_key, _style, mode_label, _marker in mode_specs
            ]
            ncols = _panel_columns(len(all_level_ids))
            pages = list(_grid_pages(len(all_level_ids), ncols, legend_labels))
            for page_idx, (fig, axes, start, stop) in enumerate(pages, start=1):
                page_levels = all_level_ids[start:stop]
                for ax, level_idx in zip(axes, page_levels):
                    level_label = get_level_label(level_idx, dataset_runs[0])
                    for run_data in dataset_runs:
                        for mode_key, line_style, mode_label, marker in mode_specs:
                            metric_key = f"acc_level_{mode_key}_{level_idx}"
                            epochs, values, stds, counts = metric_series_with_std(
                                run_data["epoch_events"], metric_key
                            )
                            ax.plot(
                                epochs,
                                values * 100.0,
                                label=f"{run_data['label']} ({mode_label})",
                                color=run_data["color"],
                                linestyle=line_style,
                                linewidth=CURVE_LINEWIDTH,
                            )
                            plot_values = values * 100.0
                            plot_stds = stds * 100.0
                            band_mask = np.isfinite(plot_values) & np.isfinite(plot_stds) & (counts > 1)
                            if np.any(band_mask):
                                ax.fill_between(
                                    epochs,
                                    plot_values - plot_stds,
                                    plot_values + plot_stds,
                                    where=band_mask,
                                    color=run_data["color"],
                                    alpha=0.14,
                                    linewidth=0,
                                )

                            best_events = run_data.get("best_epoch_events", {})
                            best_event = (
                                best_events.get(mode_key)
                                if isinstance(best_events, MappingABC)
                                else run_data.get("best_epoch_event")
                            )
                            if best_event is not None:
                                best_value = float(best_event["val_metrics_norm"].get(metric_key, np.nan))
                                if np.isfinite(best_value):
                                    ax.scatter(
                                        [best_event["epoch"]],
                                        [best_value * 100.0],
                                        color=run_data["color"],
                                        marker=marker,
                                        s=14,
                                        zorder=4,
                                    )
                                    best_epoch_std = float(best_event.get("epoch_std", np.nan))
                                    best_value_std = float(
                                        best_event.get("val_metrics_norm_std", {}).get(metric_key, np.nan)
                                    )
                                    if show_best_errorbars and int(best_event.get("epoch_count", 0)) > 1:
                                        ax.errorbar(
                                            [best_event["epoch"]],
                                            [best_value * 100.0],
                                            xerr=[[best_epoch_std], [best_epoch_std]]
                                            if np.isfinite(best_epoch_std)
                                            else None,
                                            yerr=[[best_value_std * 100.0], [best_value_std * 100.0]]
                                            if np.isfinite(best_value_std)
                                            else None,
                                            color=run_data["color"],
                                            linewidth=1.0,
                                            capsize=2,
                                            alpha=0.8,
                                            zorder=3,
                                        )

                    ax.set_title(
                        _wrap_title(
                            f"{level_label.capitalize()} level ({level_index_symbol(level_idx)})",
                            ncols,
                        )
                    )

                _close_unused_axes(axes, len(page_levels), "Epoch")
                _label_left_column(axes, len(page_levels), "Accuracy (%)")
                handles, labels = axes[0].get_legend_handles_labels()
                _add_bottom_legend(fig, handles, labels)
                suffix = f"part{page_idx}" if len(pages) > 1 else ""
                _finish(
                    fig,
                    self._stem(dataset_key, "per_level_accuracy", suffix),
                    f"{dataset_display_name(dataset_key)}: per-level validation accuracy",
                )

    def plot_final_level_test_accuracy(
        self,
        decoder_view: Optional[str] = "independent",
    ) -> None:
        """Plot selected-checkpoint test accuracy across levels for all datasets.

        The three dataset panels share a row and a 0--100 percent axis. Each
        curve is one configured run, points are seed means and error bars are
        one sample standard deviation when more than one seed contributed.
        Values are read strictly from the checkpoint selected for the requested
        decoder; no fallback between independent and top-down results is used.
        """
        view = _normalize_decoder_view(decoder_view)
        modes = decoder_modes(view)
        dataset_keys = [
            dataset_key
            for dataset_key in self.dataset_keys
            if self.run_data_by_dataset.get(dataset_key)
        ]
        if not dataset_keys:
            print("No datasets with readable runs for final per-level test accuracy.")
            return

        legend_labels: List[str] = []
        for dataset_key in dataset_keys:
            for run_data in self.run_data_by_dataset[dataset_key]:
                for mode in modes:
                    label = (
                        run_data["label"]
                        if len(modes) == 1
                        else f"{run_data['label']} ({_MODE_SPECS[mode][2]})"
                    )
                    if label not in legend_labels:
                        legend_labels.append(label)
        _legend_ncol, legend_rows = _legend_layout(legend_labels)
        figure_height = PANEL_HEIGHT_2COL_IN + legend_height_in(legend_rows)
        fig, axes = plt.subplots(
            1,
            len(dataset_keys),
            figsize=(TEXT_WIDTH_IN, figure_height),
            sharey=True,
            squeeze=False,
        )
        axes_flat = axes.reshape(-1)

        for panel_idx, dataset_key in enumerate(dataset_keys):
            ax = axes_flat[panel_idx]
            dataset_runs = self.run_data_by_dataset[dataset_key]
            level_ids = sorted(
                {
                    int(key.rsplit("_", 1)[-1])
                    for run_data in dataset_runs
                    for mode in modes
                    for key in (
                        run_data.get("test_results", {})
                        .get(mode, {})
                        .get("test_metrics", {})
                    )
                    if (
                        key.startswith(f"acc_level_{mode}_")
                        and key.rsplit("_", 1)[-1].isdigit()
                    )
                }
            )
            if not level_ids:
                ax.text(
                    0.5,
                    0.5,
                    "no selected test metrics",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(dataset_display_name(dataset_key))
                continue

            x_values = np.asarray(level_ids, dtype=float)
            for run_data in dataset_runs:
                for mode in modes:
                    means = []
                    stds = []
                    counts = []
                    for level_idx in level_ids:
                        metric_key = f"acc_level_{mode}_{level_idx}"
                        mean, std, count = _test_metric_stats(run_data, metric_key)
                        means.append(mean * 100.0)
                        stds.append(std * 100.0)
                        counts.append(count)

                    mean_values = np.asarray(means, dtype=float)
                    std_values = np.asarray(stds, dtype=float)
                    count_values = np.asarray(counts, dtype=int)
                    finite = np.isfinite(mean_values)
                    if not np.any(finite):
                        continue

                    mode_label = _MODE_SPECS[mode][2]
                    label = (
                        run_data["label"]
                        if len(modes) == 1
                        else f"{run_data['label']} ({mode_label})"
                    )
                    ax.plot(
                        x_values[finite],
                        mean_values[finite],
                        color=run_data["color"],
                        linestyle=_MODE_SPECS[mode][1],
                        marker=_MODE_SPECS[mode][3],
                        markersize=3.6,
                        linewidth=CURVE_LINEWIDTH,
                        label=label,
                    )
                    error_mask = finite & np.isfinite(std_values) & (count_values > 1)
                    if np.any(error_mask):
                        ax.errorbar(
                            x_values[error_mask],
                            mean_values[error_mask],
                            yerr=std_values[error_mask],
                            color=run_data["color"],
                            linestyle="none",
                            linewidth=0.8,
                            capsize=1.8,
                            zorder=3,
                        )

            level_labels = [
                get_level_label(level_idx, dataset_runs[0]).capitalize()
                for level_idx in level_ids
            ]
            ax.set_xticks(x_values, level_labels)
            ax.set_xlim(float(x_values.min()) - 0.15, float(x_values.max()) + 0.15)
            ax.set_ylim(0.0, 102.0)
            ax.set_title(dataset_display_name(dataset_key))

        axes_flat[0].set_ylabel("Test accuracy (%)")
        handles_by_label: Dict[str, Any] = {}
        for ax in axes_flat:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                handles_by_label.setdefault(label, handle)
        _add_bottom_legend(
            fig,
            list(handles_by_label.values()),
            list(handles_by_label.keys()),
        )
        decoders = decoder_view_phrase(view)
        _finish(
            fig,
            f"{self.config.family_slug}_final_level_accuracy",
            f"Final selected-checkpoint level accuracy ({decoders} decoding)",
        )

    def show_final_test_tables(
        self,
        decoder_view: Optional[str] = None,
        allow_single_run: bool = True,
    ) -> None:
        # A single-run dataset (typically baseline-only) still gets a table; it
        # reports absolute values without deltas.
        view = _normalize_decoder_view(
            decoder_view if decoder_view is not None else self.config.decoder_view
        )
        modes = decoder_modes(view)
        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                print(f"[{dataset_key}] Skipping final test table: no selected runs.")
                continue
            if len(dataset_runs) < 2 and not allow_single_run:
                print(f"[{dataset_key}] Skipping final test table: need at least two runs (base + comparison).")
                continue

            base = dataset_runs[0]

            level_ids_from_test = {
                int(key.rsplit("_", 1)[-1])
                for run_data in dataset_runs
                for mode in BEST_SELECTION_MODES
                for key in _test_metrics_for_mode(run_data, mode).keys()
                if (
                    _include_metric_for_view(key, view)
                    and
                    (key.startswith("acc_level_topdown_") or key.startswith("acc_level_independent_"))
                    and key.rsplit("_", 1)[-1].isdigit()
                )
            }
            all_level_ids = sorted(level_ids_from_test)
            if not all_level_ids:
                all_level_ids = sorted(
                    {
                        int(key.rsplit("_", 1)[-1])
                        for run_data in dataset_runs
                        for event in run_data["epoch_events"]
                        for key in event["val_metrics_norm"].keys()
                        if (
                            _include_metric_for_view(key, view)
                            and
                            (key.startswith("acc_level_topdown_") or key.startswith("acc_level_independent_"))
                            and key.rsplit("_", 1)[-1].isdigit()
                        )
                    }
                )

            metric_rows: List[Tuple[str, str]] = [
                ("fpa_independent", "FPA (independent)"),
                ("fpa_topdown", "FPA (top-down)"),
                ("weighted_ap_independent", "wAP (independent)"),
                ("weighted_ap_topdown", "wAP (top-down)"),
                ("tice_independent", "TICE (independent)"),
                ("tice_topdown", "TICE (top-down)"),
                ("ahd_independent", "AHD (independent)"),
                ("ahd_topdown", "AHD (top-down)"),
            ]
            metric_rows = [
                row for row in metric_rows
                if _include_metric_for_view(row[0], view)
            ]

            for level_idx in all_level_ids:
                level_label = get_level_label(level_idx, base).capitalize()
                for mode in modes:
                    mode_label = _MODE_SPECS[mode][2]
                    metric_rows.append(
                        (
                            f"acc_level_{mode}_{level_idx}",
                            f"{level_label} accuracy ({mode_label})",
                        )
                    )

            values_by_metric = {}
            best_by_metric = {}
            second_best_by_metric = {}
            for metric_key, _ in metric_rows:
                mode = _mode_from_metric_key(metric_key)
                metric_values = [
                    float(_test_metrics_for_mode(run_data, mode).get(metric_key, np.nan))
                    for run_data in dataset_runs
                ]
                values_by_metric[metric_key] = metric_values
                if len(dataset_runs) < 2:
                    # Ranking a single run would bold every cell without meaning.
                    best_indices, second_best_indices = set(), set()
                else:
                    best_indices, second_best_indices = _best_and_second_best_indices(metric_key, metric_values)
                best_by_metric[metric_key] = best_indices
                second_best_by_metric[metric_key] = second_best_indices

            best_epoch_cells: List[str] = []
            for run_data in dataset_runs:
                test_results = run_data.get("test_results", {})
                td_section: Mapping[str, Any] = {}
                ind_section: Mapping[str, Any] = {}
                if isinstance(test_results, MappingABC):
                    raw_td = test_results.get("topdown", {})
                    raw_ind = test_results.get("independent", {})
                    td_section = raw_td if isinstance(raw_td, MappingABC) else {}
                    ind_section = raw_ind if isinstance(raw_ind, MappingABC) else {}

                def epoch_text(section: Mapping[str, Any]) -> str:
                    mean = float(section.get("best_epoch", np.nan))
                    std = float(section.get("best_epoch_std", np.nan))
                    count = int(section.get("best_epoch_count", 0))
                    if not np.isfinite(mean):
                        return "n/a"
                    if count > 1 and np.isfinite(std):
                        return f"{mean:.1f} ± {std:.1f}"
                    return f"{mean:.0f}"

                sections = {"topdown": td_section, "independent": ind_section}
                best_epoch_cells.append(
                    "/".join(epoch_text(sections[mode]) for mode in BEST_SELECTION_MODES if mode in modes)
                )

            header_labels = ["Metric"] + [
                f"{run_data['label']} (n={run_data.get('num_seeds', 1)})"
                for run_data in dataset_runs
            ]
            table_lines = [
                f"### Dataset: `{dataset_key}`",
                f"Baseline run: **{base['label']}**",
                "",
                "| " + " | ".join(header_labels) + " |",
                "|---|" + "|".join(["---:"] * (len(header_labels) - 1)) + "|",
                "| Best epoch ("
                + "/".join(
                    {"topdown": "TD", "independent": "Ind"}[mode]
                    for mode in BEST_SELECTION_MODES
                    if mode in modes
                )
                + ") | "
                + " | ".join(best_epoch_cells) + " |",
            ]

            for metric_key, metric_name in metric_rows:
                metric_label = f"{metric_name} {_metric_goal_arrow(metric_key)}"
                row_cells = [metric_label]
                values = values_by_metric[metric_key]
                for run_idx, value in enumerate(values):
                    mean, std, count = _test_metric_stats(dataset_runs[run_idx], metric_key)
                    value_text = _fmt_value_stats(metric_key, mean, std, count)
                    if run_idx == 0 or not np.isfinite(values[0]) or not np.isfinite(value):
                        cell = value_text
                    else:
                        cell = f"{value_text} ({_fmt_delta(metric_key, value - values[0])})"
                    if run_idx in best_by_metric[metric_key] and cell != "n/a":
                        cell = f"**{cell}**"
                    elif run_idx in second_best_by_metric[metric_key] and cell != "n/a":
                        cell = f"<u>{cell}</u>"
                    row_cells.append(cell)
                table_lines.append("| " + " | ".join(row_cells) + " |")

            show_markdown("\n".join(table_lines))


# --------------------------------------------------------------------------- #
# Cross-model mechanism deltas
# --------------------------------------------------------------------------- #
#
# The sections above put runs on a shared axis and read the level off them. That
# works when the runs are arms of one family, and it stops working the moment the
# runs are different architectures: H-CAST and Hier-COS are four points of FPA
# apart on CIFAR-100 and eleven on Aircraft, so a mechanism worth well under one
# point is invisible next to the gap between the models carrying it.
#
# What survives the architecture gap is the paired difference: each mechanism arm
# against its OWN family's baseline, which is the only comparison in which the
# architecture is held fixed and the mechanism is the single thing that changed.
# Everything below is that difference.
#
# The pairing is per seed, not between two aggregates. Both runs of a pair are
# trained from the same seeds, so seed s of the arm and seed s of the baseline
# share their initialisation and their data order, and differencing them first
# cancels the seed-to-seed variance that would otherwise sit in both means. The
# spread reported here is therefore the spread of the EFFECT across seeds, which
# is the quantity a reader needs to decide whether an effect is there at all -
# and it is typically much smaller than the spread of either run on its own.


def _strict_test_metrics_for_mode(run_like: Mapping[str, Any], mode: str) -> Mapping[str, float]:
    """`test_results[mode]['test_metrics']`, with no fallback to the other mode.

    `_test_metrics_for_mode` falls back to the top-down section when a mode is
    missing, which is right for a table that would otherwise lose a column. It is
    wrong here: a delta that silently paired an independent number against a
    top-down one would break the mode-matched checkpoint rule that every
    comparison in these notebooks depends on, and would do it invisibly.
    """
    test_results = run_like.get("test_results", {})
    if isinstance(test_results, MappingABC):
        section = test_results.get(mode)
        if isinstance(section, MappingABC):
            metrics = section.get("test_metrics")
            if isinstance(metrics, MappingABC) and metrics:
                return metrics
    return {}


def _seed_runs_by_seed(run_data: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    """The per-seed parsed runs of one aggregated run, keyed by seed number."""
    out: Dict[int, Mapping[str, Any]] = {}
    for seed_run in run_data.get("seed_runs", []) or []:
        if not isinstance(seed_run, MappingABC):
            continue
        try:
            seed = seed_value_from_dir(Path(seed_run["run_dir"]))
        except (KeyError, ValueError):
            continue
        out[seed] = seed_run
    return out


def _seed_metric_by_epoch(seed_run: Mapping[str, Any], metric_key: str) -> Dict[int, float]:
    """One seed's validation curve for one metric, as ``{epoch: value}``."""
    out: Dict[int, float] = {}
    for event in seed_run.get("epoch_events", []) or []:
        try:
            epoch = int(event["epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        value = float(event.get("val_metrics_norm", {}).get(metric_key, np.nan))
        if np.isfinite(value):
            out[epoch] = value
    return out


@dataclass(frozen=True)
class MechanismPair:
    """One family's mechanism arm against that family's own baseline."""

    dataset_key: str
    family: str
    family_label: str
    mechanism: str
    mechanism_label: str
    color: str
    baseline: RunData
    arm: RunData
    seeds: Tuple[int, ...]

    @property
    def label(self) -> str:
        return self.family_label

    def paired_curve(self, metric_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(epochs, mean_delta, std_delta, n)`` over the seeds shared by both runs.

        An epoch is kept when at least one paired seed has a finite value in both
        runs; ``n`` is how many did, so a band can be drawn only where more than
        one seed contributed and a partly-trained arm simply ends earlier.
        """
        baseline_seeds = _seed_runs_by_seed(self.baseline)
        arm_seeds = _seed_runs_by_seed(self.arm)

        per_seed: List[Dict[int, float]] = []
        for seed in self.seeds:
            base_curve = _seed_metric_by_epoch(baseline_seeds[seed], metric_key)
            arm_curve = _seed_metric_by_epoch(arm_seeds[seed], metric_key)
            shared = set(base_curve) & set(arm_curve)
            if shared:
                per_seed.append({epoch: arm_curve[epoch] - base_curve[epoch] for epoch in shared})

        if not per_seed:
            empty = np.zeros(0)
            return empty.astype(np.int32), empty, empty, empty.astype(np.int32)

        epochs = sorted(set().union(*(set(curve) for curve in per_seed)))
        means, stds, counts = [], [], []
        for epoch in epochs:
            values = [curve[epoch] for curve in per_seed if epoch in curve]
            mean, std, count = sample_stats(values)
            means.append(mean)
            stds.append(std)
            counts.append(count)
        return (
            np.asarray(epochs, dtype=np.int32),
            np.asarray(means, dtype=np.float64),
            np.asarray(stds, dtype=np.float64),
            np.asarray(counts, dtype=np.int32),
        )

    def paired_test_delta(self, metric_key: str) -> Tuple[float, float, int]:
        """``(mean, std, n)`` of the test-metric difference over paired seeds.

        Each seed contributes its arm value minus its baseline value, both read at
        that seed's own checkpoint for this metric's decoder, so the mode-matched
        selection rule holds inside every term of the difference.
        """
        mode = _mode_from_metric_key(metric_key)
        baseline_seeds = _seed_runs_by_seed(self.baseline)
        arm_seeds = _seed_runs_by_seed(self.arm)

        deltas: List[float] = []
        for seed in self.seeds:
            base_value = float(
                _strict_test_metrics_for_mode(baseline_seeds[seed], mode).get(metric_key, np.nan)
            )
            arm_value = float(
                _strict_test_metrics_for_mode(arm_seeds[seed], mode).get(metric_key, np.nan)
            )
            if np.isfinite(base_value) and np.isfinite(arm_value):
                deltas.append(arm_value - base_value)
        return sample_stats(deltas)


class MechanismDeltaComparison:
    """Does a mechanism transfer across architectures?

    One page per mechanism per dataset, each curve a family, each value that
    family's arm minus that family's baseline. A mechanism that helps every
    architecture puts every curve on the same side of zero; one that is a fact
    about a single architecture does not.

    The class deliberately reports what it could not build. A family with no arm
    for a mechanism, an arm that is still training, and a pair whose two runs
    share no seed are three different situations, and collapsing them into an
    absent curve would let "this mechanism does not apply here" read as "this
    mechanism did nothing here".
    """

    def __init__(
        self,
        pairs: Sequence[MechanismPair],
        mechanisms: Sequence[str],
        notes: Sequence[str] = (),
    ) -> None:
        self.pairs = list(pairs)
        self.mechanisms = [str(mechanism) for mechanism in mechanisms]
        self.notes = list(notes)

        self.pairs_by_dataset: Dict[str, List[MechanismPair]] = {}
        for pair in self.pairs:
            self.pairs_by_dataset.setdefault(pair.dataset_key, []).append(pair)

        canonical = list(DATASET_SLUGS)
        self.dataset_keys = sorted(
            self.pairs_by_dataset,
            key=lambda key: (canonical.index(key) if key in canonical else len(canonical), str(key)),
        )

    @classmethod
    def for_mechanisms(
        cls,
        mechanisms: Sequence[str],
        families: Sequence[str] = COMPARISON_FAMILIES,
        outputs_root: Optional[Union[str, Path]] = None,
    ) -> "MechanismDeltaComparison":
        """Pair every family's arm for each mechanism against its own baseline.

        Families keep the colours of the baseline comparison, so a family is the
        same colour on every page of the notebook.
        """
        outputs_root = Path(outputs_root or os.environ.get("OUTPUTS_ROOT", "/scratch/g.saggini1/outputs"))
        family_names = [str(family) for family in families]
        mechanism_keys = [str(mechanism) for mechanism in mechanisms]

        unknown = [key for key in mechanism_keys if key not in MECHANISM_LABELS]
        if unknown:
            known = ", ".join(sorted(MECHANISM_LABELS))
            raise KeyError(f"Unknown mechanism: {', '.join(unknown)}. Known mechanisms: {known}.")

        colors = {family_names[0]: BASELINE_COLOR}
        for position, family in enumerate(family_names[1:]):
            colors[family] = RUN_COLOR_PALETTE[position % len(RUN_COLOR_PALETTE)]

        # Baselines are parsed once per family and shared by that family's arms.
        baselines: Dict[Tuple[str, str], RunData] = {}
        for family in family_names:
            profile = FAMILY_PROFILES[family]
            for dataset_key, slug in DATASET_SLUGS.items():
                run_dir = outputs_root / profile.baseline_template.format(dataset=slug)
                if not run_dir.is_dir() or not has_seed_runs(run_dir):
                    continue
                baselines[(family, dataset_key)] = parse_run(
                    run_dir, profile.excluded_seeds_by_run.get(run_dir.name)
                )

        pairs: List[MechanismPair] = []
        notes: List[str] = []
        for mechanism in mechanism_keys:
            for family in family_names:
                profile = FAMILY_PROFILES[family]
                template = profile.mechanism_arms.get(mechanism)
                if template is None:
                    notes.append(
                        f"{profile.display_name} has no {MECHANISM_LABELS[mechanism]} arm "
                        "registered, so it contributes no curve to that mechanism."
                    )
                    continue

                for dataset_key, slug in DATASET_SLUGS.items():
                    run_dir = outputs_root / template.format(dataset=slug)
                    dataset_name = dataset_display_name(dataset_key)
                    if not run_dir.is_dir() or not has_seed_runs(run_dir):
                        notes.append(
                            f"[{dataset_name}] {profile.display_name} "
                            f"{MECHANISM_LABELS[mechanism]}: no readable seed directory "
                            f"({run_dir.name})."
                        )
                        continue

                    baseline = baselines.get((family, dataset_key))
                    if baseline is None:
                        notes.append(
                            f"[{dataset_name}] {profile.display_name} "
                            f"{MECHANISM_LABELS[mechanism]}: the arm exists but its baseline "
                            "does not, so no delta can be formed."
                        )
                        continue

                    arm = parse_run(run_dir, profile.excluded_seeds_by_run.get(run_dir.name))
                    shared = sorted(set(_seed_runs_by_seed(baseline)) & set(_seed_runs_by_seed(arm)))
                    if not shared:
                        notes.append(
                            f"[{dataset_name}] {profile.display_name} "
                            f"{MECHANISM_LABELS[mechanism]}: arm seeds "
                            f"{sorted(_seed_runs_by_seed(arm))} and baseline seeds "
                            f"{sorted(_seed_runs_by_seed(baseline))} do not overlap, so the "
                            "runs cannot be paired."
                        )
                        continue

                    pairs.append(
                        MechanismPair(
                            dataset_key=dataset_key,
                            family=family,
                            family_label=profile.display_name,
                            mechanism=mechanism,
                            mechanism_label=MECHANISM_LABELS[mechanism],
                            color=colors[family],
                            baseline=baseline,
                            arm=arm,
                            seeds=tuple(shared),
                        )
                    )

        return cls(pairs=pairs, mechanisms=mechanism_keys, notes=notes)

    def pairs_for(self, dataset_key: str, mechanism: str) -> List[MechanismPair]:
        return [
            pair
            for pair in self.pairs_by_dataset.get(dataset_key, [])
            if pair.mechanism == mechanism
        ]

    def print_pair_summary(self) -> None:
        """Which pairs were formed, on how many seeds, and what was left out."""
        if not self.pairs:
            print("No mechanism pairs could be formed from the selected families.")
        for mechanism in self.mechanisms:
            label = MECHANISM_LABELS[mechanism]
            families = sorted({pair.family_label for pair in self.pairs if pair.mechanism == mechanism})
            print(f"{label}: {len(families)} family/families with a delta ({', '.join(families) or 'none'})")
            for dataset_key in self.dataset_keys:
                for pair in self.pairs_for(dataset_key, mechanism):
                    base_seeds = sorted(_seed_runs_by_seed(pair.baseline))
                    arm_seeds = sorted(_seed_runs_by_seed(pair.arm))
                    unpaired = sorted(set(base_seeds) ^ set(arm_seeds))
                    unpaired_txt = f", unpaired seeds {unpaired}" if unpaired else ""
                    print(
                        f"    [{dataset_display_name(dataset_key)}] {pair.family_label}: "
                        f"{len(pair.seeds)} paired seed(s) {list(pair.seeds)}"
                        f"{unpaired_txt}"
                    )

        if self.notes:
            print()
            print("Not compared:")
            for note in dict.fromkeys(self.notes):
                print(f"    {note}")

    def report_arm_config_differences(self, max_keys: int = 12) -> None:
        """Every resolved-config key that differs between an arm and its baseline.

        A delta is only about the mechanism if the mechanism is the only thing
        that changed, and that is a property of the two configs, not of the run
        names. Some families cannot isolate it even in principle - H-CAST's
        lexicographic mode requires `model.loss.globalkl: false`, so its lex arm
        necessarily also drops the global KL term its baseline trains with - and
        some arms carry a second change that is simply how they were launched.
        Both show up here as extra keys beside the mechanism's own switch.

        The comparison is between the resolved configs of the first paired seed,
        with the per-run fields (seed, output dir, resume) already dropped by
        `normalized_config_for_seed_comparison`.
        """

        def flatten(node: Any, prefix: str = "") -> Dict[str, Any]:
            if isinstance(node, MappingABC):
                flat: Dict[str, Any] = {}
                for key, value in node.items():
                    flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
                return flat
            return {prefix: node}

        for dataset_key in self.dataset_keys:
            for mechanism in self.mechanisms:
                for pair in self.pairs_for(dataset_key, mechanism):
                    seed = pair.seeds[0]
                    base_cfg = _seed_runs_by_seed(pair.baseline)[seed].get("config", {}) or {}
                    arm_cfg = _seed_runs_by_seed(pair.arm)[seed].get("config", {}) or {}
                    base_flat = flatten(normalized_config_for_seed_comparison(base_cfg))
                    arm_flat = flatten(normalized_config_for_seed_comparison(arm_cfg))

                    differing = sorted(
                        key
                        for key in set(base_flat) | set(arm_flat)
                        if base_flat.get(key) != arm_flat.get(key)
                    )
                    head = (
                        f"[{dataset_display_name(dataset_key)}] {pair.family_label} "
                        f"{pair.mechanism_label} vs its baseline:"
                    )
                    if not differing:
                        print(f"{head} configs identical - the mechanism is switched on elsewhere.")
                        continue
                    print(f"{head} {len(differing)} differing config key(s)")
                    for key in differing[:max_keys]:
                        print(f"    {key}: {base_flat.get(key)!r} -> {arm_flat.get(key)!r}")
                    if len(differing) > max_keys:
                        print(f"    ... and {len(differing) - max_keys} more")

    def plot_delta_curves(
        self,
        metric_families: Optional[Sequence[Tuple[str, str, bool]]] = None,
        mode_specs: Optional[Sequence[Tuple[str, str, str, str]]] = None,
        decoder_view: Optional[str] = None,
    ) -> None:
        """One figure per dataset per mechanism: every family's delta against zero.

        Colour is the family and line style is the decoder, the same convention as
        every other figure here, so only the quantity on the y axis has changed.
        The rule at zero is the baseline: a curve above it means the arm scored
        higher than its own baseline that epoch, which is an improvement for FPA
        and weighted AP and a regression for TICE and AHD - the axis label carries
        the direction so the sign is never read on its own.

        The band is one standard deviation of the per-seed differences, so a band
        that straddles zero is a mechanism whose sign is not settled across seeds.
        """
        metric_families = metric_families or [
            ("fpa", "$\\Delta$ Validation FPA (pp, higher better)", True),
            ("weighted_ap", "$\\Delta$ Validation wAP (pp, higher better)", True),
            ("tice", "$\\Delta$ Validation TICE (pp, lower better)", True),
            ("ahd", "$\\Delta$ Validation AHD (edges, lower better)", False),
        ]
        view = _normalize_decoder_view(decoder_view)
        mode_specs = (
            _default_mode_specs(view)
            if mode_specs is None
            else _filter_mode_specs(mode_specs, view)
        )

        for dataset_key in self.dataset_keys:
            for mechanism in self.mechanisms:
                pairs = self.pairs_for(dataset_key, mechanism)
                if not pairs:
                    continue

                legend_labels = [
                    f"{pair.family_label} ({mode_label})"
                    for pair in pairs
                    for _mode_key, _style, mode_label, _marker in mode_specs
                ]
                ncols = _panel_columns(len(metric_families))
                pages = list(_grid_pages(len(metric_families), ncols, legend_labels))
                for page_idx, (fig, axes, start, stop) in enumerate(pages, start=1):
                    page_families = metric_families[start:stop]
                    for ax, (metric_prefix, metric_title, is_percent) in zip(axes, page_families):
                        ax.axhline(0.0, color="#444444", linewidth=0.7, zorder=1)
                        for pair in pairs:
                            for mode_key, line_style, mode_label, _marker in mode_specs:
                                epochs, means, stds, counts = pair.paired_curve(
                                    f"{metric_prefix}_{mode_key}"
                                )
                                if epochs.size == 0:
                                    continue
                                scale = 100.0 if is_percent else 1.0
                                values = means * scale
                                ax.plot(
                                    epochs,
                                    values,
                                    label=f"{pair.family_label} ({mode_label})",
                                    color=pair.color,
                                    linestyle=line_style,
                                    linewidth=CURVE_LINEWIDTH,
                                    zorder=2,
                                )
                                band = stds * scale
                                band_mask = np.isfinite(values) & np.isfinite(band) & (counts > 1)
                                if np.any(band_mask):
                                    ax.fill_between(
                                        epochs,
                                        values - band,
                                        values + band,
                                        where=band_mask,
                                        color=pair.color,
                                        alpha=0.14,
                                        linewidth=0,
                                    )
                        ax.set_title(_wrap_title(metric_title, ncols))

                    _close_unused_axes(axes, len(page_families), "Epoch")
                    handles, labels = axes[0].get_legend_handles_labels()
                    _add_bottom_legend(fig, handles, labels)
                    suffix = f"part{page_idx}" if len(pages) > 1 else ""
                    stem = "_".join(
                        part
                        for part in ("comparison", dataset_slug(dataset_key), "delta", mechanism, suffix)
                        if part
                    )
                    _finish(
                        fig,
                        stem,
                        f"{dataset_display_name(dataset_key)}: {MECHANISM_LABELS[mechanism]} "
                        "against each family's own baseline",
                    )

    def show_delta_tables(self, decoder_view: Optional[str] = None) -> None:
        """One table per dataset: the paired test-metric delta of every arm.

        Columns are grouped by mechanism, so the question "did this mechanism do
        the same thing to two architectures" is read across a group. Nothing is
        ranked and nothing is bold: these are differences of different models, not
        competitors on one axis, and the useful reading is the sign and whether it
        is larger than the spread beside it.
        """
        view = _normalize_decoder_view(decoder_view)
        modes = decoder_modes(view)
        for dataset_key in self.dataset_keys:
            columns = [
                pair
                for mechanism in self.mechanisms
                for pair in self.pairs_for(dataset_key, mechanism)
            ]
            if not columns:
                print(f"[{dataset_display_name(dataset_key)}] no mechanism pairs to tabulate.")
                continue

            level_ids = sorted(
                {
                    int(key.rsplit("_", 1)[-1])
                    for pair in columns
                    for run in (pair.baseline, pair.arm)
                    for mode in BEST_SELECTION_MODES
                    for key in _strict_test_metrics_for_mode(run, mode).keys()
                    if key.startswith(("acc_level_topdown_", "acc_level_independent_"))
                    and key.rsplit("_", 1)[-1].isdigit()
                }
            )

            metric_rows: List[Tuple[str, str]] = [
                ("fpa_independent", "FPA (independent)"),
                ("fpa_topdown", "FPA (top-down)"),
                ("weighted_ap_independent", "wAP (independent)"),
                ("weighted_ap_topdown", "wAP (top-down)"),
                ("tice_independent", "TICE (independent)"),
                ("tice_topdown", "TICE (top-down)"),
                ("ahd_independent", "AHD (independent)"),
                ("ahd_topdown", "AHD (top-down)"),
            ]
            metric_rows = [row for row in metric_rows if _include_metric_for_view(row[0], view)]
            for level_idx in level_ids:
                level_label = get_level_label(level_idx, columns[0].baseline).capitalize()
                for mode in modes:
                    mode_label = _MODE_SPECS[mode][2]
                    metric_rows.append(
                        (
                            f"acc_level_{mode}_{level_idx}",
                            f"{level_label} accuracy ({mode_label})",
                        )
                    )

            header = ["Metric"] + [
                f"{pair.mechanism_label}<br>{pair.family_label} (n={len(pair.seeds)})"
                for pair in columns
            ]
            lines = [
                f"### Dataset: `{dataset_key}`",
                "Each cell is the arm minus that family's own baseline, paired by seed.",
                "",
                "| " + " | ".join(header) + " |",
                "|---|" + "|".join(["---:"] * (len(header) - 1)) + "|",
            ]

            dropped: List[str] = []
            for metric_key, metric_name in metric_rows:
                cells = [f"{metric_name} {_metric_goal_arrow(metric_key)}"]
                stats = [pair.paired_test_delta(metric_key) for pair in columns]

                # A row whose every delta is exactly zero carries no information
                # about the mechanism: TICE under top-down decoding is zero in both
                # runs by construction, so its difference can only be zero. The row
                # is named rather than dropped in silence, because a zero that is
                # measured and a zero that is forced look identical in a table.
                finite = [mean for mean, _std, _count in stats if np.isfinite(mean)]
                if finite and all(mean == 0.0 for mean in finite):
                    dropped.append(metric_name)
                    continue

                for mean, std, count in stats:
                    if not np.isfinite(mean):
                        cells.append("n/a")
                        continue
                    text = _fmt_delta(metric_key, mean)
                    if count > 1 and np.isfinite(std):
                        std_text = (
                            f"{std:.3f}" if _is_distance_metric(metric_key) else f"{100.0 * std:.2f}"
                        )
                        text = f"{text} ± {std_text}"
                    cells.append(text)
                lines.append("| " + " | ".join(cells) + " |")

            if dropped:
                lines.extend(
                    [
                        "",
                        "Identically zero in every column and therefore not tabulated: "
                        + "; ".join(dropped)
                        + ".",
                    ]
                )
            lines.extend(
                [
                    "",
                    "A delta is an improvement when its sign matches the arrow in its row: "
                    "positive for FPA, wAP and the per-level accuracies, negative for TICE "
                    "and AHD. The spread is the standard deviation of the per-seed "
                    "differences, so a delta smaller than the number beside it has not been "
                    "shown to differ from zero on these seeds. `n/a` is a metric one of the "
                    "two runs has not written yet.",
                ]
            )
            show_markdown("\n".join(lines))
