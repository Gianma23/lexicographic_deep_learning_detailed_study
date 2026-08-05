"""Shared aggregation and plotting helpers for the current-run notebooks.

Every notebook under ``analysis/current_runs`` follows the same shape: declare a
run matrix, discover the completed seed directories under ``OUTPUTS_ROOT``,
aggregate the selected test checkpoints, and draw the same three figures. This
module holds everything that is not model-specific.

All of it reads the **independently selected** checkpoint and the independent
metric family. Top-down decoding is deliberately not offered here: its predicted
path is consistent by construction, so ``tice_topdown`` is identically zero and
``fpa_topdown`` collapses onto top-down fine accuracy, which leaves nothing for a
trade-off view to show.

Figures are authored for the thesis at their final printed size, using the same
style contract as ``analysis/datasets_analysis.ipynb`` so that every figure in
the document reads as one system. See :func:`use_paper_style` and
:func:`save_figure`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox


LEVEL_LABELS = ("Coarse", "Middle", "Fine")

OFF_SCALE_GUTTER = 0.22
OFF_SCALE_GROUP_SEP = 0.095
OFF_SCALE_SEED_SEP = 0.030

DEFAULT_PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#6A3D9A")
DEFAULT_MARKERS = ("o", "^", "s", "D", "v", "P", "X")
DEFAULT_HATCHES = ("", "//", "\\\\", "xx", "..", "++", "oo")

MODEL_REFERENCE_SPECS = (
    {"key": "hcast", "label": "H-CAST", "run_name": "hcast_{dataset}", "color": "#6A3D9A"},
    {"key": "hrn", "label": "HRN", "run_name": "hrn_{dataset}", "color": "#33A02C"},
    {
        "key": "hiercos",
        "label": "Hier-COS",
        "run_name": "hiercos_{dataset}_global_softmax_ce_reg_baseline_kl_leaf",
        "color": "#E6AB02",
    },
    {"key": "lhdnn", "label": "LH-DNN", "run_name": "lhdnn_{dataset}", "color": "#E7298A"},
    {"key": "ht_capsnet", "label": "HT-CapsNet", "run_name": "capsnet_{dataset}", "color": "#8C564B"},
)


# --------------------------------------------------------------------------- #
# Print style
# --------------------------------------------------------------------------- #

# Figures are authored at their final printed width and must be included with
# width=\linewidth and no extra scaling: any additional resizing in LaTeX shrinks
# the font sizes set below along with the artwork. These values are copied from
# analysis/datasets_analysis.ipynb so the two notebook families produce figures
# that sit side by side in the thesis without a visible style break.
TEXT_WIDTH_IN = 6.3   # A4 text block with 2.5 cm margins
HALF_WIDTH_IN = 3.05

PAPER_RCPARAMS = {
    "figure.dpi": 140, "savefig.dpi": 400,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "legend.title_fontsize": 7,
    "axes.axisbelow": True, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
}

# Sizes tuned for a 6.3 in wide, roughly 2.3 in tall panel. Marker areas are in
# points squared, so they do not follow the figure size automatically.
POINT_LABEL_FONTSIZE = 6.4
BAR_LABEL_FONTSIZE = 5.8
SUPLABEL_FONTSIZE = 8.5
SEED_MARKER_SIZE = 11
MEAN_MARKER_SIZE = 38
EMPHASIS_MARKER_SIZE = 56
REFERENCE_MARKER_SIZE = 58


def use_paper_style():
    """Apply the thesis figure style shared with the dataset-analysis notebook."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PAPER_RCPARAMS)


def save_figure(fig, figure_dir, stem, save_figures=True, formats=("pdf", "png")):
    """Write a figure at its authored size, then show it.

    ``bbox_inches='tight'`` is deliberately not used: it re-crops the canvas and
    changes the physical width, which would break the one-to-one relationship
    between the authored figure size and the printed text width.
    """
    if save_figures:
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        for suffix in formats:
            fig.savefig(figure_dir / f"{stem}.{suffix}")
    plt.show()


def _legend_height_in(handle_count, ncol):
    """Vertical inches a bottom legend needs, so the panels can be sized around it."""
    if handle_count <= 0:
        return 0.0
    rows = math.ceil(handle_count / max(1, ncol))
    return 0.22 + 0.155 * rows


# --------------------------------------------------------------------------- #
# Discovery and aggregation
# --------------------------------------------------------------------------- #


def format_percent(value, std=np.nan):
    return f"{value:.2f}%" + (f" ± {std:.2f}" if np.isfinite(std) else "")


def format_ahd(value, std=np.nan):
    return f"{value:.3f}" + (f" ± {std:.3f}" if np.isfinite(std) else "")


def sample_stats(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    std = float(values.std(ddof=1)) if values.size > 1 else np.nan
    return float(values.mean()), std


def seed_dirs(run_dir):
    """Return completed ``(seed, directory)`` pairs, including legacy flat runs."""
    run_dir = Path(run_dir)
    candidates = []
    for child in run_dir.iterdir() if run_dir.is_dir() else []:
        if not child.is_dir() or not child.name.startswith("seed_"):
            continue
        try:
            seed = int(child.name.removeprefix("seed_"))
        except ValueError:
            continue
        if (child / "test_metrics.yaml").is_file():
            candidates.append((seed, child))
    if not candidates and (run_dir / "test_metrics.yaml").is_file():
        candidates.append((0, run_dir))
    return sorted(candidates)


def load_test_metrics(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def independent_selection(seed_dir):
    """Return the independently selected checkpoint and whether it is legacy.

    Runs written before the top-down/independent selection split store a single
    checkpoint; it is used as-is and flagged so notebooks can mark it.
    """
    payload = load_test_metrics(Path(seed_dir) / "test_metrics.yaml")
    if "independent" in payload:
        return payload["independent"], False
    return payload, True


def independent_values(metrics):
    """Extract the independent-decoding metrics from a selected checkpoint."""
    return {
        "fpa": 100.0 * float(metrics["fpa_independent"]),
        "tice": 100.0 * float(metrics["tice_independent"]),
        "ahd": float(metrics["ahd_independent"]),
        "weighted_ap": 100.0 * float(metrics["weighted_ap_independent"]),
        **{
            f"acc_level_{level}": 100.0 * float(metrics[f"acc_level_independent_{level}"])
            for level in range(3)
        },
    }


VALUE_KEYS = ("best_epoch", "fpa", "tice", "ahd", "weighted_ap", *(f"acc_level_{i}" for i in range(3)))


def _first_active_hcc_epoch(seed_dir):
    """Find the first logged epoch whose train/validation HCC alpha is positive."""
    log_path = Path(seed_dir) / "run_log.jsonl"
    if not log_path.is_file():
        return np.nan
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "epoch":
                continue
            alphas = [
                metrics.get("proj_constraint_alpha")
                for metrics in (event.get("train_metrics", {}), event.get("val_metrics", {}))
            ]
            if any(alpha is not None and float(alpha) > 0.0 for alpha in alphas):
                return float(event["epoch"])
    return np.nan


def _run_name(spec, dataset_key):
    run_name = spec["run_name"]
    return run_name(dataset_key) if callable(run_name) else run_name.format(dataset=dataset_key)


def discover_rows(outputs_root, datasets, run_specs, pareto=True):
    """Aggregate the independently selected test metrics of every completed seed.

    Returns ``(rows, missing)``. A spec may restrict itself to a subset of
    datasets through ``datasets`` and opts into HCC activation diagnostics
    through ``family='hcc'``.
    """
    outputs_root = Path(outputs_root)
    rows = []
    missing = []
    for dataset_key, dataset_label in datasets.items():
        for spec in run_specs:
            if spec.get("datasets") and dataset_key not in spec["datasets"]:
                continue
            run_name = _run_name(spec, dataset_key)
            completed = seed_dirs(outputs_root / run_name)
            if not completed:
                missing.append({"dataset": dataset_label, "label": spec["label"], "run_name": run_name})
                continue

            seed_values = {}
            single_selection = False
            for seed, seed_dir in completed:
                selected, is_legacy = independent_selection(seed_dir)
                single_selection = single_selection or is_legacy
                metrics = selected["test_metrics"]
                values = {
                    "best_epoch": float(selected.get("best_epoch", np.nan)),
                    **independent_values(metrics),
                }
                if spec.get("family") == "hcc":
                    values["selected_hcc_alpha"] = float(metrics.get("proj_constraint_alpha", np.nan))
                    values["first_active_hcc_epoch"] = _first_active_hcc_epoch(seed_dir)
                seed_values[seed] = values

            row = {
                **spec,
                "dataset": dataset_label,
                "dataset_key": dataset_key,
                "run_name": run_name,
                "seeds": sorted(seed_values),
                "seed_values": seed_values,
                "single_selection": single_selection,
                "point_label": spec.get("point_label") or spec.get("plot_label") or spec["label"],
            }
            stat_keys = list(VALUE_KEYS)
            if spec.get("family") == "hcc":
                stat_keys.extend(("selected_hcc_alpha", "first_active_hcc_epoch"))
            for key in stat_keys:
                mean, std = sample_stats(values.get(key, np.nan) for values in seed_values.values())
                row[key] = mean
                row[f"{key}_std"] = std
            rows.append(row)

    if pareto:
        mark_pareto(rows, datasets)
    return rows, missing


def mark_pareto(rows, datasets, label_key="label"):
    """Flag rows that no other run of the same dataset dominates on (FPA, TICE)."""
    for dataset in datasets.values():
        subset = [row for row in rows if row["dataset"] == dataset]
        for row in subset:
            dominators = [
                other[label_key]
                for other in subset
                if other is not row
                and other["fpa"] >= row["fpa"]
                and other["tice"] <= row["tice"]
                and (other["fpa"] > row["fpa"] or other["tice"] < row["tice"])
            ]
            row["pareto_optimal"] = not dominators
            row["dominated_by"] = dominators
    return rows


def assign_default_styles(rows, palette=DEFAULT_PALETTE, markers=DEFAULT_MARKERS, hatches=DEFAULT_HATCHES):
    """Fill in ``color``/``marker``/``hatch`` for specs that do not declare them.

    Styles are assigned per run key so that a run keeps the same appearance in
    every dataset panel.
    """
    keys = list(dict.fromkeys(row["key"] for row in rows))
    for row in rows:
        index = keys.index(row["key"])
        row.setdefault("color", palette[index % len(palette)])
        row.setdefault("marker", markers[index % len(markers)])
        row.setdefault("hatch", hatches[index % len(hatches)])
    return rows


def model_reference_specs(exclude=()):
    """Return consistently styled cross-model references, excluding the focal model."""
    excluded = set(exclude)
    return [
        {**spec, "family": "reference", "marker": "s"}
        for spec in MODEL_REFERENCE_SPECS
        if spec["key"] not in excluded
    ]


def print_availability(rows, missing, datasets, label_key="label", title="runs"):
    """Report completed seed counts per dataset."""
    for dataset in datasets.values():
        print(f"\n{dataset} {title}")
        for row in (row for row in rows if row["dataset"] == dataset):
            note = " [legacy single selection]" if row["single_selection"] else ""
            print(f"  {row[label_key]:<26} {len(row['seeds']):>2} seed(s): {row['run_name']}{note}")
        for item in (item for item in missing if item["dataset"] == dataset):
            print(f"  {item['label']:<26} missing: {item['run_name']}")


def print_reference_availability(rows, missing, datasets):
    """Report the cross-model references with their independent metrics."""
    for dataset in datasets.values():
        print(f"\n{dataset} reference models")
        for row in (row for row in rows if row["dataset"] == dataset):
            suffix = "*" if row["single_selection"] else ""
            fpa = format_percent(row["fpa"], row["fpa_std"])
            tice = format_percent(row["tice"], row["tice_std"])
            print(f"  {row['label'] + suffix:<14} {len(row['seeds']):>2} seed(s)  FPA {fpa:<18} TICE {tice}")
        for item in (item for item in missing if item["dataset"] == dataset):
            print(f"  {item['label']:<14} missing: {item['run_name']}")


# --------------------------------------------------------------------------- #
# Plot primitives
# --------------------------------------------------------------------------- #


def add_covariance_ellipse(ax, row, color=None):
    points = np.asarray(
        [[values["tice"], values["fpa"]] for values in row["seed_values"].values()],
        dtype=float,
    )
    if points.shape[0] < 3:
        return
    covariance = np.cov(points, rowvar=False, ddof=1)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    color = color or row["color"]
    ax.add_patch(
        Ellipse(
            (row["tice"], row["fpa"]),
            width=2.0 * np.sqrt(eigenvalues[0]),
            height=2.0 * np.sqrt(eigenvalues[1]),
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=0.12,
            linewidth=0.7,
            zorder=2,
        )
    )


def _bbox_overlap_area(first, second):
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


# Candidate label offsets are a polar grid rather than a hand-ordered list, so
# every direction is reachable at every distance and nothing biases the search
# toward one corner. Radii are in typographic points, matching ``offset points``.
LABEL_ANGLES = np.deg2rad(np.arange(0.0, 360.0, 18.0))
LABEL_RADII_PT = (7.0, 13.0, 21.0, 31.0, 43.0, 58.0)

# Score weights. Overlaps are areas in square pixels and act as hard constraints;
# the remaining terms only break ties between non-overlapping placements.
LABEL_OVERLAP_WEIGHT = 1000.0
POINT_OVERLAP_WEIGHT = 250.0
OUTSIDE_WEIGHT = 1000.0
DISTANCE_WEIGHT = 0.60
DIRECTION_WEIGHT = 17.0
CROWDING_WEIGHT = 9.0
CROWDING_RADIUS_PT = 58.0


def _outward_directions(point_pixels):
    """Unit vector per point pointing away from the other points, in display space.

    Labels default to fanning outward from the local cloud. Without this the
    scoring ties on every empty direction and the first candidate wins for every
    point, which is what packs them all into the same corner.
    """
    count = len(point_pixels)
    directions = np.zeros((count, 2), dtype=float)
    for index in range(count):
        delta = point_pixels[index] - point_pixels
        distance = np.linalg.norm(delta, axis=1)
        neighbours = distance > 1e-6
        if not neighbours.any():
            continue
        weights = 1.0 / np.clip(distance[neighbours], 10.0, None) ** 2
        directions[index] = (delta[neighbours] * weights[:, None]).sum(axis=0)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    fallback = np.array([np.sqrt(0.5), np.sqrt(0.5)])
    return np.where(norms > 1e-9, directions / np.maximum(norms, 1e-9), fallback)


def _alignment(dx, dy):
    return ("left" if dx >= 0 else "right", "bottom" if dy > 0 else "top" if dy < 0 else "center")


def _slide_to_edge(bbox, direction, axes_bbox, blockers):
    """Distance a placed label can still travel toward the axes edge it hugs.

    The chosen offset is measured from the marker and, for a right- or
    top-aligned label, cannot reach past it. A gutter-pinned marker sits in the
    middle of its gutter, so that leaves an unused strip between the label and
    the axes edge while the label's far end reaches into the data. This recovers
    the strip, stopping at the edge or at whatever is already placed there.
    """
    if direction is None:
        return 0.0, 0.0
    dx_dir, dy_dir = direction
    if dx_dir:
        limit = (axes_bbox.x1 - bbox.x1) if dx_dir > 0 else (bbox.x0 - axes_bbox.x0)
        for other in blockers:
            if other.y1 <= bbox.y0 or other.y0 >= bbox.y1:
                continue
            gap = (other.x0 - bbox.x1) if dx_dir > 0 else (bbox.x0 - other.x1)
            if gap >= 0.0:
                limit = min(limit, gap)
        return max(0.0, limit) * np.sign(dx_dir), 0.0
    if dy_dir:
        limit = (axes_bbox.y1 - bbox.y1) if dy_dir > 0 else (bbox.y0 - axes_bbox.y0)
        for other in blockers:
            if other.x1 <= bbox.x0 or other.x0 >= bbox.x1:
                continue
            gap = (other.y0 - bbox.y1) if dy_dir > 0 else (bbox.y0 - other.y1)
            if gap >= 0.0:
                limit = min(limit, gap)
        return 0.0, max(0.0, limit) * np.sign(dy_dir)
    return 0.0, 0.0


def _offset_bbox(px, py, dx, dy, width, height, pad):
    """Where the label box lands for a given pixel offset, mirroring the ha/va rules."""
    horizontal, vertical = _alignment(dx, dy)
    x0 = px + dx if horizontal == "left" else px + dx - width
    if vertical == "bottom":
        y0 = py + dy
    elif vertical == "top":
        y0 = py + dy - height
    else:
        y0 = py + dy - 0.5 * height
    return Bbox.from_extents(x0 - pad, y0 - pad, x0 + width + pad, y0 + height + pad)


def place_point_labels(ax, label_specs, fontsize=POINT_LABEL_FONTSIZE):
    """Greedily place direct labels in display space and draw leader lines.

    Placement is scored per candidate offset: overlapping another label, a data
    point, or the axes edge is effectively forbidden, and among the remaining
    offsets the label prefers to stay close to its point, to point away from its
    neighbours, and to keep its distance from labels already placed.

    A spec may override the neighbour-repulsion default with its own
    ``preferred`` direction. Off-scale markers use it: their repulsion vector
    points out of the axes, which is forbidden, so without a hint they drift
    diagonally into the middle of the panel instead of running along the gutter
    they are pinned to.
    """
    if not label_specs:
        return
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    # Offsets are given in points but every box is measured in display pixels.
    to_pixels = figure.dpi / 72.0
    pad = 1.6 * to_pixels
    marker_halfsize = 3.2 * to_pixels

    axes_bbox = ax.get_window_extent(renderer=renderer).padded(-2.0)
    point_pixels = np.asarray([ax.transData.transform((item["x"], item["y"])) for item in label_specs])
    point_boxes = [
        Bbox.from_extents(x - marker_halfsize, y - marker_halfsize, x + marker_halfsize, y + marker_halfsize)
        for x, y in point_pixels
    ]
    directions = _outward_directions(point_pixels)
    for index, spec in enumerate(label_specs):
        preferred = spec.get("preferred")
        if preferred is None:
            continue
        vector = np.asarray(preferred, dtype=float)
        norm = np.linalg.norm(vector)
        if norm > 1e-9:
            directions[index] = vector / norm

    distances = np.linalg.norm(point_pixels[:, None, :] - point_pixels[None, :, :], axis=2)
    density = ((distances < 55.0 * to_pixels) & (distances > 0)).sum(axis=1)
    order = sorted(
        range(len(label_specs)),
        key=lambda index: (density[index], len(label_specs[index]["text"])),
        reverse=True,
    )
    candidates = [
        (float(radius * np.cos(angle)), float(radius * np.sin(angle)), float(radius))
        for radius in LABEL_RADII_PT
        for angle in LABEL_ANGLES
    ]
    crowding_radius = CROWDING_RADIUS_PT * to_pixels

    annotations = []
    for spec in label_specs:
        annotations.append(
            ax.annotate(
                spec["text"],
                (spec["x"], spec["y"]),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=fontsize,
                color=spec["color"],
                linespacing=1.25,
                zorder=5,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.80},
                arrowprops={
                    "arrowstyle": "-", "color": spec["color"], "alpha": 0.55,
                    "linewidth": 0.5, "shrinkA": 1, "shrinkB": 3,
                },
            )
        )

    occupied = []
    for index in order:
        annotation = annotations[index]
        # The rendered size does not depend on the offset, so measure once and
        # translate the box analytically instead of re-laying out per candidate.
        annotation.xyann = (0, 0)
        annotation.set_horizontalalignment("left")
        annotation.set_verticalalignment("bottom")
        annotation.update_positions(renderer)
        extent = Text.get_window_extent(annotation, renderer=renderer)
        width, height = extent.width, extent.height
        px, py = point_pixels[index]
        preferred = directions[index]

        best = None
        for dx_pt, dy_pt, radius_pt in candidates:
            dx, dy = dx_pt * to_pixels, dy_pt * to_pixels
            bbox = _offset_bbox(px, py, dx, dy, width, height, pad)
            overlap = sum(_bbox_overlap_area(bbox, other) for other in occupied)
            point_overlap = sum(
                _bbox_overlap_area(bbox, point_box)
                for point_index, point_box in enumerate(point_boxes)
                if point_index != index
            )
            outside = (
                max(0.0, axes_bbox.x0 - bbox.x0) * bbox.height
                + max(0.0, bbox.x1 - axes_bbox.x1) * bbox.height
                + max(0.0, axes_bbox.y0 - bbox.y0) * bbox.width
                + max(0.0, bbox.y1 - axes_bbox.y1) * bbox.width
            )
            cos_theta = (dx_pt * preferred[0] + dy_pt * preferred[1]) / radius_pt
            centre = np.array([0.5 * (bbox.x0 + bbox.x1), 0.5 * (bbox.y0 + bbox.y1)])
            crowding = sum(
                max(0.0, 1.0 - np.linalg.norm(
                    centre - [0.5 * (other.x0 + other.x1), 0.5 * (other.y0 + other.y1)]
                ) / crowding_radius)
                for other in occupied
            )
            score = (
                LABEL_OVERLAP_WEIGHT * overlap
                + POINT_OVERLAP_WEIGHT * point_overlap
                + OUTSIDE_WEIGHT * outside
                + DISTANCE_WEIGHT * radius_pt
                + DIRECTION_WEIGHT * (1.0 - cos_theta)
                + CROWDING_WEIGHT * crowding
            )
            if best is None or score < best[0]:
                best = (score, dx_pt, dy_pt, bbox)

        _, dx_pt, dy_pt, bbox = best
        # Alignment follows the chosen offset, not the slid one: sliding must not
        # flip the anchor side and move the box back where it came from.
        horizontal, vertical = _alignment(dx_pt, dy_pt)
        slide_x, slide_y = _slide_to_edge(
            bbox, label_specs[index].get("slide"), axes_bbox, occupied + point_boxes
        )
        if slide_x or slide_y:
            bbox = Bbox.from_extents(
                bbox.x0 + slide_x, bbox.y0 + slide_y, bbox.x1 + slide_x, bbox.y1 + slide_y
            )
            dx_pt += slide_x / to_pixels
            dy_pt += slide_y / to_pixels
        annotation.xyann = (dx_pt, dy_pt)
        annotation.set_horizontalalignment(horizontal)
        annotation.set_verticalalignment(vertical)
        occupied.append(bbox)


def reference_display_label(row):
    return row["label"] + ("*" if row["single_selection"] else "")


def reference_point_label(row):
    """Identity, then the numbers the pinned position cannot show.

    FPA and TICE go on separate lines rather than one: a pinned label hugs the
    gutter it sits in, and a narrow three-line block stays at the edge where a
    single wide line would reach into the middle of the panel.
    """
    seed_count = len(row["seeds"])
    header = reference_display_label(row) + (f" ×{seed_count}" if seed_count > 1 else "")
    if seed_count > 1:
        fpa = format_percent(row["fpa"], row["fpa_std"])
        tice = format_percent(row["tice"], row["tice_std"])
    else:
        fpa = format_percent(row["fpa"])
        tice = format_percent(row["tice"])
    return f"{header}\nFPA {fpa}\nTICE {tice}"


def _off_scale_sign(value, inner_lo, inner_hi):
    if value < inner_lo:
        return -1
    if value > inner_hi:
        return 1
    return 0


def off_scale_layout(reference_subset, inner_xlim, inner_ylim):
    """Pin out-of-range references into gutters around the focal-family range."""
    x_span = inner_xlim[1] - inner_xlim[0]
    y_span = inner_ylim[1] - inner_ylim[0]
    entries = [
        {
            "row": row,
            "sx": _off_scale_sign(row["tice"], *inner_xlim),
            "sy": _off_scale_sign(row["fpa"], *inner_ylim),
        }
        for row in reference_subset
    ]
    used_sides = {
        "left": any(entry["sx"] < 0 for entry in entries),
        "right": any(entry["sx"] > 0 for entry in entries),
        "bottom": any(entry["sy"] < 0 for entry in entries),
        "top": any(entry["sy"] > 0 for entry in entries),
    }
    outer_xlim = (
        inner_xlim[0] - (OFF_SCALE_GUTTER * x_span if used_sides["left"] else 0.0),
        inner_xlim[1] + (OFF_SCALE_GUTTER * x_span if used_sides["right"] else 0.0),
    )
    outer_ylim = (
        inner_ylim[0] - (OFF_SCALE_GUTTER * y_span if used_sides["bottom"] else 0.0),
        inner_ylim[1] + (OFF_SCALE_GUTTER * y_span if used_sides["top"] else 0.0),
    )

    def pinned(sign, value, inner, outer):
        if sign < 0:
            return 0.5 * (outer[0] + inner[0])
        if sign > 0:
            return 0.5 * (inner[1] + outer[1])
        return value

    for entry in entries:
        entry["x"] = pinned(entry["sx"], entry["row"]["tice"], inner_xlim, outer_xlim)
        entry["y"] = pinned(entry["sy"], entry["row"]["fpa"], inner_ylim, outer_ylim)

    zones = {}
    for entry in entries:
        zones.setdefault((entry["sx"], entry["sy"]), []).append(entry)
    for (sx, sy), members in zones.items():
        if len(members) < 2 or (sx == 0 and sy == 0):
            continue
        norm = float(np.hypot(sx, sy))
        perp_x, perp_y = -sy / norm, sx / norm
        for index, entry in enumerate(sorted(members, key=lambda item: item["row"]["label"])):
            offset = (index - (len(members) - 1) / 2) * OFF_SCALE_GROUP_SEP
            if sx:
                entry["x"] += offset * perp_x * x_span
            if sy:
                entry["y"] += offset * perp_y * y_span

    for entry in entries:
        sx, sy = entry["sx"], entry["sy"]
        seed_points = list(entry["row"]["seed_values"].values())
        if sx and sy:
            norm = float(np.hypot(sx, sy))
            perp_x, perp_y = -sy / norm, sx / norm
            count = len(seed_points)
            entry["seed_xy"] = [
                (
                    entry["x"] + (index - (count - 1) / 2) * OFF_SCALE_SEED_SEP * perp_x * x_span,
                    entry["y"] + (index - (count - 1) / 2) * OFF_SCALE_SEED_SEP * perp_y * y_span,
                )
                for index in range(count)
            ]
        else:
            entry["seed_xy"] = [
                (entry["x"] if sx else point["tice"], entry["y"] if sy else point["fpa"])
                for point in seed_points
            ]
        entry["marker"] = (
            "s" if (sx == 0 and sy == 0)
            else (3, 0, float(np.degrees(np.arctan2(sy, sx)) - 90.0))
        )
        entry["off_scale"] = bool(sx or sy)

    return {
        "entries": entries,
        "used_sides": used_sides,
        "outer_xlim": outer_xlim,
        "outer_ylim": outer_ylim,
    }


def draw_off_scale_gutters(ax, inner_xlim, inner_ylim, layout):
    shade = {"facecolor": "#000000", "alpha": 0.05, "linewidth": 0, "zorder": 0}
    boundary = {"color": "#777777", "linestyle": (0, (4, 3)), "linewidth": 0.6, "zorder": 1}
    outer_xlim, outer_ylim = layout["outer_xlim"], layout["outer_ylim"]
    if layout["used_sides"]["left"]:
        ax.axvspan(outer_xlim[0], inner_xlim[0], **shade)
        ax.axvline(inner_xlim[0], **boundary)
    if layout["used_sides"]["right"]:
        ax.axvspan(inner_xlim[1], outer_xlim[1], **shade)
        ax.axvline(inner_xlim[1], **boundary)
    if layout["used_sides"]["bottom"]:
        ax.axhspan(outer_ylim[0], inner_ylim[0], **shade)
        ax.axhline(inner_ylim[0], **boundary)
    if layout["used_sides"]["top"]:
        ax.axhspan(inner_ylim[1], outer_ylim[1], **shade)
        ax.axhline(inner_ylim[1], **boundary)


def reference_legend_handles(reference_rows, reference_specs, off_scale=True):
    handles = [
        Line2D(
            [], [], color=spec["color"], marker="s", linestyle="None",
            markersize=5, label=spec["label"],
        )
        for spec in reference_specs
        if any(row["key"] == spec["key"] for row in reference_rows)
    ]
    if handles and off_scale:
        handles.extend(
            [
                Line2D(
                    [], [], color="#555555", marker=(3, 0, -90), linestyle="None",
                    markersize=6, label="Off-scale: tip → true value",
                ),
                Patch(
                    facecolor="#000000", alpha=0.05, edgecolor="#777777", linestyle="--",
                    label="Off-scale gutter",
                ),
            ]
        )
    return handles


def default_point_legend_handles(rows, label_key="label"):
    unique = {row["key"]: row for row in rows}
    return [
        Line2D(
            [], [], color=row["color"], marker=row["marker"], linestyle="None",
            markersize=5, markerfacecolor="none" if row.get("hollow") else row["color"],
            markeredgewidth=1.1 if row.get("hollow") else 0.5, label=row[label_key],
        )
        for row in unique.values()
    ]


def spread_legend_handles():
    return [
        Line2D([], [], color="#555555", marker=".", linestyle="None", markersize=4,
               alpha=0.4, label="Individual seed"),
        Ellipse((0, 0), 1.0, 0.5, facecolor="#777777", edgecolor="#777777",
                alpha=0.15, label="1-SD covariance ellipse"),
    ]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
#
# All three figures stack one panel per dataset vertically. At the 6.3 in thesis
# text width a 1x3 row leaves only about 2 in per panel, which is not enough for
# the direct labels or for grouped bars once a run matrix grows; a full-width
# panel per dataset keeps every label legible at 6-7 pt. Figure titles are
# omitted on purpose — they belong in the LaTeX caption.


def _draw_run_points(ax, subset, label_key):
    """Draw focal-family seeds, means, and ellipses; return their label specs."""
    labels = []
    for row in subset:
        color = row["color"]
        hollow = bool(row.get("hollow"))
        seed_points = list(row["seed_values"].values())
        ax.scatter(
            [point["tice"] for point in seed_points],
            [point["fpa"] for point in seed_points],
            marker=row["marker"], s=SEED_MARKER_SIZE,
            facecolor="none" if hollow else color,
            edgecolor=color if hollow else "none",
            linewidth=0.5 if hollow else 0, alpha=0.35, zorder=2,
        )
        add_covariance_ellipse(ax, row, color)
        ax.scatter(
            row["tice"], row["fpa"], marker=row["marker"],
            s=EMPHASIS_MARKER_SIZE if row.get("emphasis") else MEAN_MARKER_SIZE,
            facecolor="none" if hollow else color,
            edgecolor=color if hollow else "white",
            linewidth=1.1 if hollow else 0.6, zorder=3,
        )
        labels.append({"x": row["tice"], "y": row["fpa"], "text": row[label_key], "color": color})
    return labels


def _draw_reference_points(ax, reference_subset):
    """Draw references on the shared scale, without the off-scale machinery."""
    labels = []
    for row in reference_subset:
        seed_points = list(row["seed_values"].values())
        ax.scatter(
            [point["tice"] for point in seed_points],
            [point["fpa"] for point in seed_points],
            marker="s", s=SEED_MARKER_SIZE, facecolor="none", edgecolor=row["color"],
            linewidth=0.5, alpha=0.55, zorder=3,
        )
        ax.scatter(
            row["tice"], row["fpa"], marker="s", s=REFERENCE_MARKER_SIZE,
            facecolor=row["color"], edgecolor="white", linewidth=0.6, zorder=4,
        )
        labels.append(
            {"x": row["tice"], "y": row["fpa"], "text": reference_point_label(row), "color": row["color"]}
        )
    return labels


def plot_tradeoff(
    rows,
    datasets,
    figure_dir,
    save_figures=False,
    reference_rows=(),
    reference_specs=(),
    freeze_on_focal="auto",
    label_key="point_label",
    legend_handles=None,
    legend_ncol=3,
    panel_height=2.55,
    stem="fpa_tice_tradeoff",
):
    """Draw the independent FPA-TICE trade-off, one full-width panel per dataset.

    ``freeze_on_focal`` keeps the axis range on the focal model family and pins
    cross-model references into shaded gutters, which is what the multi-arm
    notebooks want. ``False`` puts every model on one shared scale instead, which
    is the readable choice when the focal family has a single arm. The default
    ``'auto'`` decides per panel: freeze once the panel holds at least two focal
    runs, since a single point carries no range to freeze on.
    """
    handles = list(legend_handles) if legend_handles is not None else default_point_legend_handles(rows)
    handles.extend(spread_legend_handles())
    # Reserve legend space up front so the panels keep their authored height.
    estimated = len(handles) + len({row["key"] for row in reference_rows}) + 2
    figure_height = len(datasets) * panel_height + _legend_height_in(estimated, legend_ncol)

    fig, axes = plt.subplots(len(datasets), 1, figsize=(TEXT_WIDTH_IN, figure_height))
    axes = np.atleast_1d(axes)
    labels_by_axis = []
    plotted_references = []
    used_gutters = False
    for ax, dataset in zip(axes, datasets.values()):
        subset = [row for row in rows if row["dataset"] == dataset]
        reference_subset = [row for row in reference_rows if row["dataset"] == dataset]
        labels = _draw_run_points(ax, subset, label_key)
        freeze = len(subset) >= 2 if freeze_on_focal == "auto" else bool(freeze_on_focal)

        if subset and freeze:
            # Freeze the view on the focal family before adding other models.
            ax.margins(x=0.16, y=0.26)
            ax.autoscale_view()
            inner_xlim, inner_ylim = ax.get_xlim(), ax.get_ylim()
            layout = off_scale_layout(reference_subset, inner_xlim, inner_ylim)
            ax.set_xlim(layout["outer_xlim"])
            ax.set_ylim(layout["outer_ylim"])
            draw_off_scale_gutters(ax, inner_xlim, inner_ylim, layout)
            used_gutters = used_gutters or any(layout["used_sides"].values())
            for entry in layout["entries"]:
                row = entry["row"]
                seed_xy = np.asarray(entry["seed_xy"], dtype=float)
                if seed_xy.size:
                    ax.scatter(
                        seed_xy[:, 0], seed_xy[:, 1], marker=entry["marker"], s=SEED_MARKER_SIZE,
                        facecolor="none", edgecolor=row["color"], linewidth=0.5, alpha=0.55, zorder=3,
                    )
                ax.scatter(
                    entry["x"], entry["y"], marker=entry["marker"], s=REFERENCE_MARKER_SIZE,
                    facecolor=row["color"], edgecolor="white", linewidth=0.6, zorder=4,
                )
                # Run the label back along the gutter it is pinned to, so it
                # hugs that edge instead of cutting across the panel. Panels are
                # much wider than they are tall, so a corner pin follows the
                # horizontal gutter.
                sx, sy = entry["sx"], entry["sy"]
                preferred = (-sx, 0.0) if sx else ((0.0, -sy) if sy else None)
                labels.append(
                    {
                        "x": entry["x"], "y": entry["y"], "text": reference_point_label(row),
                        "color": row["color"], "preferred": preferred,
                        # Once placed, push it back out to the axes edge so the
                        # strip beyond the pinned marker is not left empty.
                        "slide": (sx, 0.0) if sx else ((0.0, sy) if sy else None),
                    }
                )
                plotted_references.append(row)
        else:
            labels.extend(_draw_reference_points(ax, reference_subset))
            plotted_references.extend(reference_subset)
            ax.margins(x=0.14, y=0.22)

        ax.set_title(dataset)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, alpha=0.25)
        labels_by_axis.append(labels)

    # The x label goes on the bottom panel rather than through fig.supxlabel:
    # constrained_layout puts a supxlabel in the same outside-bottom slot as the
    # legend, and the two overlap. Every panel carries its own x scale anyway.
    axes[-1].set_xlabel("TICE, independent decoding (%) — lower is better")
    fig.supylabel("FPA, independent decoding (%) — higher is better", fontsize=SUPLABEL_FONTSIZE)

    handles.extend(reference_legend_handles(plotted_references, reference_specs, off_scale=used_gutters))
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=legend_ncol, frameon=True)

    # Resolve the constrained layout, then freeze it: the label placement below
    # measures in display space and must not be invalidated by a later relayout.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax, labels in zip(axes, labels_by_axis):
        place_point_labels(ax, labels)
    save_figure(fig, figure_dir, stem, save_figures)


def plot_level_accuracy(
    rows,
    datasets,
    figure_dir,
    save_figures=False,
    reference_rows=(),
    label_key="label",
    legend_ncol=3,
    panel_height=1.55,
    stem="level_accuracy",
):
    """Draw absolute coarse/middle/fine accuracy per run, with optional references.

    This is the view that stays informative when a model family has a single arm,
    where the delta figure below has nothing to compare against.
    """
    x = np.arange(3)
    # Prefer the bar-specific style: point colour may encode a shared property
    # (Hier-COS uses it for the softmax family) and would leave bars ambiguous.
    series = [
        {"key": row["key"], "label": row[label_key],
         "color": row.get("bar_color") or row["color"],
         "hatch": row.get("bar_hatch", row.get("hatch", "")), "reference": False}
        for row in {row["key"]: row for row in rows}.values()
    ]
    series += [
        {"key": row["key"], "label": row["label"] + " (reference)", "color": row["color"],
         "hatch": "..", "reference": True}
        for row in {row["key"]: row for row in reference_rows}.values()
    ]
    if not series:
        print("No runs available for the level-accuracy figure.")
        return
    width = min(0.22, 0.86 / len(series))
    # Per-bar value labels stop being readable once the bars get thin.
    annotate = len(series) <= 6

    figure_height = len(datasets) * panel_height + _legend_height_in(len(series), legend_ncol)
    fig, axes = plt.subplots(
        len(datasets), 1, figsize=(TEXT_WIDTH_IN, figure_height), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets.values()):
        for index, spec in enumerate(series):
            pool = reference_rows if spec["reference"] else rows
            row = next((r for r in pool if r["dataset"] == dataset and r["key"] == spec["key"]), None)
            if row is None:
                continue
            means = [row[f"acc_level_{level}"] for level in range(3)]
            stds = [row[f"acc_level_{level}_std"] for level in range(3)]
            positions = x + (index - (len(series) - 1) / 2) * width
            bars = ax.bar(
                positions, means, width=width, color=spec["color"], hatch=spec["hatch"],
                edgecolor="#333333", linewidth=0.35, zorder=2,
                alpha=0.75 if spec["reference"] else 1.0,
                yerr=stds if np.all(np.isfinite(stds)) else None, capsize=1.6,
                error_kw={"linewidth": 0.6},
            )
            if annotate:
                ax.bar_label(bars, fmt="%.1f", fontsize=BAR_LABEL_FONTSIZE, padding=1.5)
        ax.set_title(dataset)
        ax.set_xticks(x, LEVEL_LABELS)
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, axis="y", alpha=0.25)
    fig.supylabel("Independent accuracy (%)", fontsize=SUPLABEL_FONTSIZE)
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=spec["color"], hatch=spec["hatch"],
            edgecolor="#333333", linewidth=0.35,
            alpha=0.75 if spec["reference"] else 1.0, label=spec["label"],
        )
        for spec in series
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=legend_ncol, frameon=True)
    save_figure(fig, figure_dir, stem, save_figures)


def default_baseline_selector(row, subset):
    return next((other for other in subset if other.get("family") == "baseline"), None)


def plot_level_accuracy_deltas(
    rows,
    datasets,
    figure_dir,
    save_figures=False,
    baseline_selector=default_baseline_selector,
    baseline_name="native baseline",
    label_key="label",
    legend_ncol=3,
    panel_height=1.55,
    share_y=False,
    stem="level_accuracy_matched_deltas",
):
    """Plot matched-seed level-accuracy deltas from each run's reference baseline.

    Deltas are computed per seed and then averaged, so only seeds present in both
    the run and its baseline contribute. ``baseline_selector(row, subset)`` picks
    the baseline within the same dataset; runs without one are omitted.

    ``share_y`` is off by default: delta ranges differ by an order of magnitude
    across datasets, and one shared scale flattens the small-delta panels into
    invisible bars. Each panel therefore carries its own labelled y axis, and the
    printed values make the magnitudes explicit; state in the caption that the
    vertical scales are per dataset.
    """
    x = np.arange(3)
    pairs = []
    for dataset in datasets.values():
        subset = [row for row in rows if row["dataset"] == dataset]
        for row in subset:
            baseline = baseline_selector(row, subset)
            if baseline is not None and baseline is not row:
                pairs.append({"run": row, "baseline": baseline})
    if not pairs:
        print("No run/baseline pair available for the delta figure.")
        return

    comparison_keys = list(dict.fromkeys(pair["run"]["key"] for pair in pairs))
    comparison_specs = []
    for index, key in enumerate(comparison_keys):
        run = next(pair["run"] for pair in pairs if pair["run"]["key"] == key)
        comparison_specs.append(
            {
                "key": key,
                "label": run.get("bar_label") or run[label_key],
                "color": run.get("bar_color") or run["color"],
                "hatch": run.get("bar_hatch", run.get("hatch", DEFAULT_HATCHES[index % len(DEFAULT_HATCHES)])),
            }
        )
    width = min(0.20, 0.8 / len(comparison_specs))
    annotate = len(comparison_specs) <= 6

    figure_height = len(datasets) * panel_height + _legend_height_in(len(comparison_specs), legend_ncol)
    fig, axes = plt.subplots(
        len(datasets), 1, figsize=(TEXT_WIDTH_IN, figure_height), sharex=True, sharey=share_y
    )
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets.values()):
        dataset_pairs = [pair for pair in pairs if pair["run"]["dataset"] == dataset]
        for index, spec in enumerate(comparison_specs):
            pair = next((p for p in dataset_pairs if p["run"]["key"] == spec["key"]), None)
            if pair is None:
                continue
            run, baseline = pair["run"], pair["baseline"]
            common_seeds = sorted(set(run["seeds"]) & set(baseline["seeds"]))
            if not common_seeds:
                continue
            samples = np.asarray(
                [
                    [
                        run["seed_values"][seed][f"acc_level_{level}"]
                        - baseline["seed_values"][seed][f"acc_level_{level}"]
                        for level in range(3)
                    ]
                    for seed in common_seeds
                ],
                dtype=float,
            )
            means = samples.mean(axis=0)
            stds = samples.std(axis=0, ddof=1) if len(common_seeds) > 1 else np.full(3, np.nan)
            positions = x + (index - (len(comparison_specs) - 1) / 2) * width
            bars = ax.bar(
                positions, means, width=width, color=spec["color"], hatch=spec["hatch"],
                edgecolor="#333333", linewidth=0.35, zorder=2,
                yerr=stds if np.all(np.isfinite(stds)) else None,
                capsize=1.6, error_kw={"linewidth": 0.6},
            )
            if annotate:
                ax.bar_label(bars, fmt="%+.2f", fontsize=BAR_LABEL_FONTSIZE, padding=1.5)
        ax.axhline(0, color="#333333", linewidth=0.7)
        ax.set_title(dataset)
        ax.set_xticks(x, LEVEL_LABELS)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, axis="y", alpha=0.25)
    fig.supylabel(
        f"Independent accuracy change from {baseline_name} (pp)", fontsize=SUPLABEL_FONTSIZE
    )
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=spec["color"], hatch=spec["hatch"],
            edgecolor="#333333", linewidth=0.35, label=spec["label"],
        )
        for spec in comparison_specs
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=legend_ncol, frameon=True)
    save_figure(fig, figure_dir, stem, save_figures)


def print_summary(rows, datasets, label_key="label", label_width=24):
    """Print per-dataset run tables with Pareto status and HCC diagnostics."""
    show_hcc = any(row.get("family") == "hcc" for row in rows)
    header = (
        f"  {'run':<{label_width}} {'n':>2}  {'epoch':<15} {'FPA':<17} "
        f"{'TICE':<17} {'AHD':<17} {'weighted AP':<17} "
    )
    if show_hcc:
        header += f"{'HCC diagnostics':<24} "
    header += "status"
    for dataset in datasets.values():
        print(f"\n{dataset}")
        print(header)
        subset = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: (-row["fpa"], row["tice"]),
        )
        for row in subset:
            epoch = (
                f"{row['best_epoch']:.1f} ± {row['best_epoch_std']:.1f}"
                if np.isfinite(row["best_epoch_std"]) else f"{row['best_epoch']:.0f}"
            )
            status = (
                "Pareto-optimal" if row.get("pareto_optimal")
                else "dominated by " + ", ".join(row.get("dominated_by", []))
            )
            label = row[label_key] + ("*" if row["single_selection"] else "")
            line = (
                f"  {label:<{label_width}} {len(row['seeds']):>2}  {epoch:<15} "
                f"{format_percent(row['fpa'], row['fpa_std']):<17} "
                f"{format_percent(row['tice'], row['tice_std']):<17} "
                f"{format_ahd(row['ahd'], row['ahd_std']):<17} "
                f"{format_percent(row['weighted_ap'], row['weighted_ap_std']):<17} "
            )
            if show_hcc:
                if row.get("family") == "hcc":
                    alpha, first = row["selected_hcc_alpha"], row["first_active_hcc_epoch"]
                    diagnostics = (
                        f"selected α={alpha:.2f}; first active={first:.0f}"
                        if np.isfinite(first) else f"selected α={alpha:.2f}; no active epoch"
                    )
                else:
                    diagnostics = "—"
                line += f"{diagnostics:<24} "
            print(line + status)
