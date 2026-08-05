"""Shared aggregation and plotting helpers for compact current-run notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox


LEVEL_LABELS = ("Coarse", "Middle", "Fine")
OFF_SCALE_GUTTER = 0.22
OFF_SCALE_GROUP_SEP = 0.095
OFF_SCALE_SEED_SEP = 0.030

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


def seed_paths(run_dir):
    """Return only completed seed directories for notebook-specific discovery."""
    return [path for _, path in seed_dirs(run_dir)]


def load_test_metrics(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _independent_selection(seed_dir):
    payload = load_test_metrics(Path(seed_dir) / "test_metrics.yaml")
    single_selection = "independent" not in payload
    selected = payload if single_selection else payload["independent"]
    return selected, single_selection


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


def discover_rows(outputs_root, datasets, run_specs):
    """Aggregate independently selected test metrics by dataset and run family."""
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
                selected, seed_single_selection = _independent_selection(seed_dir)
                single_selection = single_selection or seed_single_selection
                metrics = selected["test_metrics"]
                values = {
                    "best_epoch": float(selected.get("best_epoch", np.nan)),
                    "fpa": 100.0 * float(metrics["fpa_independent"]),
                    "tice": 100.0 * float(metrics["tice_independent"]),
                    **{
                        f"acc_level_{level}": 100.0 * float(metrics[f"acc_level_independent_{level}"])
                        for level in range(3)
                    },
                }
                if spec["family"] == "hcc":
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
            }
            stat_keys = ["best_epoch", "fpa", "tice", *(f"acc_level_{level}" for level in range(3))]
            if spec["family"] == "hcc":
                stat_keys.extend(("selected_hcc_alpha", "first_active_hcc_epoch"))
            for key in stat_keys:
                mean, std = sample_stats(values.get(key, np.nan) for values in seed_values.values())
                row[key] = mean
                row[f"{key}_std"] = std
            rows.append(row)
    return mark_pareto(rows, datasets), missing


def mark_pareto(rows, datasets):
    for dataset in datasets.values():
        subset = [row for row in rows if row["dataset"] == dataset]
        for row in subset:
            dominators = [
                other["label"]
                for other in subset
                if other is not row
                and other["fpa"] >= row["fpa"]
                and other["tice"] <= row["tice"]
                and (other["fpa"] > row["fpa"] or other["tice"] < row["tice"])
            ]
            row["pareto_optimal"] = not dominators
            row["dominated_by"] = dominators
    return rows


def print_availability(rows, missing, datasets):
    for dataset in datasets.values():
        print(f"\n{dataset}")
        for row in (row for row in rows if row["dataset"] == dataset):
            selection_note = " [legacy single selection]" if row["single_selection"] else ""
            print(f"  {row['label']:<24} {len(row['seeds']):>2} seed(s): {row['run_name']}{selection_note}")
        for item in (item for item in missing if item["dataset"] == dataset):
            print(f"  {item['label']:<24} missing: {item['run_name']}")


def model_reference_specs(exclude=()):
    """Return consistently styled baseline references, excluding the focal model."""
    excluded = set(exclude)
    return [dict(spec) for spec in MODEL_REFERENCE_SPECS if spec["key"] not in excluded]


def discover_reference_rows(outputs_root, datasets, reference_specs):
    """Aggregate baseline models used only as contextual plot references."""
    outputs_root = Path(outputs_root)
    rows = []
    missing = []
    for dataset_key, dataset_label in datasets.items():
        for spec in reference_specs:
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
                selected, seed_single_selection = _independent_selection(seed_dir)
                single_selection = single_selection or seed_single_selection
                metrics = selected["test_metrics"]
                seed_values[seed] = {
                    "best_epoch": float(selected.get("best_epoch", np.nan)),
                    "fpa": 100.0 * float(metrics["fpa_independent"]),
                    "tice": 100.0 * float(metrics["tice_independent"]),
                    **{
                        f"acc_level_{level}": 100.0 * float(metrics[f"acc_level_independent_{level}"])
                        for level in range(3)
                    },
                }
            row = {
                **spec,
                "dataset": dataset_label,
                "dataset_key": dataset_key,
                "run_name": run_name,
                "seeds": sorted(seed_values),
                "seed_values": seed_values,
                "single_selection": single_selection,
            }
            for key in ("best_epoch", "fpa", "tice", *(f"acc_level_{level}" for level in range(3))):
                mean, std = sample_stats(values[key] for values in seed_values.values())
                row[key] = mean
                row[f"{key}_std"] = std
            rows.append(row)
    return rows, missing


def print_reference_availability(rows, missing, datasets):
    for dataset in datasets.values():
        print(f"\n{dataset} reference models")
        for row in (row for row in rows if row["dataset"] == dataset):
            suffix = "*" if row["single_selection"] else ""
            fpa = f"{row['fpa']:.2f}% ± {row['fpa_std']:.2f}" if np.isfinite(row["fpa_std"]) else f"{row['fpa']:.2f}%"
            tice = f"{row['tice']:.2f}% ± {row['tice_std']:.2f}" if np.isfinite(row["tice_std"]) else f"{row['tice']:.2f}%"
            print(f"  {row['label'] + suffix:<14} {len(row['seeds']):>2} seed(s)  {fpa:<17} {tice:<17}")
        for item in (item for item in missing if item["dataset"] == dataset):
            print(f"  {item['label']:<14} missing: {item['run_name']}")


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
            linewidth=1.0,
            zorder=2,
        )
    )


def _bbox_overlap_area(first, second):
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def place_point_labels(ax, label_specs, fontsize=8):
    """Greedily place direct labels in display space and draw leader lines."""
    if not label_specs:
        return
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer).padded(-3)
    point_pixels = np.asarray([ax.transData.transform((item["x"], item["y"])) for item in label_specs])
    point_boxes = [Bbox.from_extents(x - 6, y - 6, x + 6, y + 6) for x, y in point_pixels]
    distances = np.linalg.norm(point_pixels[:, None, :] - point_pixels[None, :, :], axis=2)
    density = ((distances < 80) & (distances > 0)).sum(axis=1)
    order = sorted(
        range(len(label_specs)),
        key=lambda index: (density[index], len(label_specs[index]["text"])),
        reverse=True,
    )
    candidates = [
        (8, 6), (8, -6), (-8, 6), (-8, -6), (18, 0), (-18, 0),
        (8, 20), (8, -20), (-8, 20), (-8, -20), (30, 10), (30, -10),
        (-30, 10), (-30, -10), (14, 34), (14, -34), (-14, 34), (-14, -34),
        (46, 18), (46, -18), (-46, 18), (-46, -18),
        (12, 48), (12, -48), (-12, 48), (-12, -48),
        (36, 34), (36, -34), (-36, 34), (-36, -34),
        (12, 62), (12, -62), (-12, 62), (-12, -62),
        (58, 30), (58, -30), (-58, 30), (-58, -30),
        (16, 78), (16, -78), (-16, 78), (-16, -78),
    ]
    occupied = []
    for index in order:
        spec = label_specs[index]
        annotation = ax.annotate(
            spec["text"],
            (spec["x"], spec["y"]),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=fontsize,
            color=spec["color"],
            zorder=5,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            arrowprops={
                "arrowstyle": "-", "color": spec["color"], "alpha": 0.55,
                "linewidth": 0.7, "shrinkA": 2, "shrinkB": 5,
            },
        )
        best = None
        for rank, (dx, dy) in enumerate(candidates):
            annotation.xyann = (dx, dy)
            annotation.set_horizontalalignment("left" if dx >= 0 else "right")
            annotation.set_verticalalignment("bottom" if dy > 0 else "top" if dy < 0 else "center")
            annotation.update_positions(renderer)
            bbox = Text.get_window_extent(annotation, renderer=renderer).padded(2)
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
            distance_penalty = 0.15 * np.hypot(dx, dy) + 0.02 * rank
            score = 1000.0 * overlap + 250.0 * point_overlap + 1000.0 * outside + distance_penalty
            if best is None or score < best[0]:
                best = (score, dx, dy, bbox)
        _, dx, dy, bbox = best
        annotation.xyann = (dx, dy)
        annotation.set_horizontalalignment("left" if dx >= 0 else "right")
        annotation.set_verticalalignment("bottom" if dy > 0 else "top" if dy < 0 else "center")
        occupied.append(bbox)


def finish_figure(fig, figure_dir, filename, save_figures):
    if save_figures:
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_dir / filename, bbox_inches="tight")
    plt.show()


def reference_display_label(row):
    return row["label"] + ("*" if row["single_selection"] else "")


def reference_point_label(row):
    seed_count = len(row["seeds"])
    header = reference_display_label(row) + (f" ×{seed_count}" if seed_count > 1 else "")
    if seed_count > 1 and np.isfinite(row["fpa_std"]) and np.isfinite(row["tice_std"]):
        return (
            f"{header}\nFPA {row['fpa']:.1f} ± {row['fpa_std']:.1f}"
            f"\nTICE {row['tice']:.2f} ± {row['tice_std']:.2f}"
        )
    return f"{header}\nFPA {row['fpa']:.1f}\nTICE {row['tice']:.2f}"


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
    boundary = {"color": "#777777", "linestyle": (0, (4, 3)), "linewidth": 0.9, "zorder": 1}
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


def reference_legend_handles(reference_rows, reference_specs):
    handles = [
        Line2D(
            [], [], color=spec["color"], marker="s", linestyle="None",
            markersize=8, label=spec["label"],
        )
        for spec in reference_specs
        if any(row["key"] == spec["key"] for row in reference_rows)
    ]
    if handles:
        handles.extend(
            [
                Line2D(
                    [], [], color="#555555", marker=(3, 0, -90), linestyle="None",
                    markersize=11, label="Off-scale: tip points to the true value",
                ),
                Patch(
                    facecolor="#000000", alpha=0.05, edgecolor="#777777", linestyle="--",
                    label="Off-scale gutter (beyond focal-model range)",
                ),
            ]
        )
    return handles


def plot_fpa_tice(
    rows,
    datasets,
    model_name,
    figure_dir,
    save_figures=False,
    reference_rows=(),
    reference_specs=(),
):
    """Match the Hier-COS notebook's independent FPA–TICE trade-off view."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = np.atleast_1d(axes)
    labels_by_axis = []
    plotted_references = []
    for ax, dataset in zip(axes, datasets.values()):
        subset = [row for row in rows if row["dataset"] == dataset]
        labels = []
        for row in subset:
            seed_points = list(row["seed_values"].values())
            ax.scatter(
                [point["tice"] for point in seed_points],
                [point["fpa"] for point in seed_points],
                marker=row["marker"],
                s=24,
                facecolor="none",
                edgecolor=row["color"],
                linewidth=0.8,
                alpha=0.35,
                zorder=2,
            )
            add_covariance_ellipse(ax, row)
            ax.scatter(
                row["tice"],
                row["fpa"],
                marker=row["marker"],
                s=145 if row["family"] == "baseline" else 105,
                facecolor=row["color"],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
            labels.append({"x": row["tice"], "y": row["fpa"], "text": row["label"], "color": row["color"]})

        # Freeze the view on the focal model family before adding contextual models.
        ax.margins(x=0.22, y=0.22)
        ax.autoscale_view()
        inner_xlim, inner_ylim = ax.get_xlim(), ax.get_ylim()
        reference_subset = [row for row in reference_rows if row["dataset"] == dataset] if subset else []
        layout = off_scale_layout(reference_subset, inner_xlim, inner_ylim)
        ax.set_xlim(layout["outer_xlim"])
        ax.set_ylim(layout["outer_ylim"])
        draw_off_scale_gutters(ax, inner_xlim, inner_ylim, layout)
        for entry in layout["entries"]:
            row = entry["row"]
            seed_xy = np.asarray(entry["seed_xy"], dtype=float)
            if seed_xy.size:
                ax.scatter(
                    seed_xy[:, 0], seed_xy[:, 1], marker=entry["marker"], s=26,
                    facecolor="none", edgecolor=row["color"], linewidth=0.9, alpha=0.55, zorder=3,
                )
            ax.scatter(
                entry["x"], entry["y"], marker=entry["marker"], s=170,
                facecolor=row["color"], edgecolor="white", linewidth=1.0, zorder=4,
            )
            labels.append(
                {
                    "x": entry["x"], "y": entry["y"],
                    "text": reference_point_label(row), "color": row["color"],
                }
            )
            plotted_references.append(row)
        ax.set_title(dataset, fontweight="bold")
        ax.set_xlabel("TICE independent (%) — lower is better")
        ax.set_ylabel("Independent FPA (%) — higher is better")
        ax.grid(True, alpha=0.25)
        labels_by_axis.append(labels)

    unique_specs = {row["key"]: row for row in rows}
    handles = [
        Line2D([], [], color=row["color"], marker=row["marker"], linestyle="None", markersize=9, label=row["label"])
        for row in unique_specs.values()
    ]
    handles.extend(
        [
            Line2D([], [], color="#555555", marker=".", linestyle="None", markersize=7, alpha=0.4, label="Individual seed"),
            Ellipse((0, 0), 1.0, 0.5, facecolor="#777777", edgecolor="#777777", alpha=0.15, label="1-SD covariance ellipse"),
        ]
    )
    handles.extend(reference_legend_handles(plotted_references, reference_specs))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=max(2, min(6, len(handles))), bbox_to_anchor=(0.5, -0.05), frameon=True)
    fig.suptitle(f"{model_name}: independent FPA–TICE trade-off", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    for ax, labels in zip(axes, labels_by_axis):
        place_point_labels(ax, labels)
    finish_figure(fig, figure_dir, "fpa_tice_tradeoff.svg", save_figures)


def plot_level_accuracy_deltas(rows, datasets, model_name, figure_dir, save_figures=False):
    """Plot matched-seed independent level-accuracy deltas from the native baseline."""
    comparison_keys = list(dict.fromkeys(row["key"] for row in rows if row["family"] != "baseline"))
    comparison_specs = {key: next(row for row in rows if row["key"] == key) for key in comparison_keys}
    x = np.arange(3)
    width = min(0.22, 0.8 / max(1, len(comparison_keys)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets.values()):
        baseline = next((row for row in rows if row["dataset"] == dataset and row["family"] == "baseline"), None)
        for index, key in enumerate(comparison_keys):
            run = next((row for row in rows if row["dataset"] == dataset and row["key"] == key), None)
            if run is None or baseline is None:
                continue
            common_seeds = sorted(set(run["seeds"]) & set(baseline["seeds"]))
            if not common_seeds:
                continue
            samples = np.asarray(
                [
                    [run["seed_values"][seed][f"acc_level_{level}"] - baseline["seed_values"][seed][f"acc_level_{level}"] for level in range(3)]
                    for seed in common_seeds
                ],
                dtype=float,
            )
            means = samples.mean(axis=0)
            stds = samples.std(axis=0, ddof=1) if len(common_seeds) > 1 else np.full(3, np.nan)
            positions = x + (index - (len(comparison_keys) - 1) / 2) * width
            bars = ax.bar(
                positions,
                means,
                width=width,
                color=run["color"],
                hatch=run.get("hatch", ""),
                edgecolor="#333333",
                linewidth=0.5,
                yerr=stds if np.all(np.isfinite(stds)) else None,
                capsize=3,
                zorder=2,
            )
            ax.bar_label(bars, fmt="%+.2f", fontsize=7.5, padding=2)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title(dataset, fontweight="bold")
        ax.set_xticks(x, LEVEL_LABELS)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Independent accuracy change from native baseline (percentage points)")
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=comparison_specs[key]["color"],
            hatch=comparison_specs[key].get("hatch", ""),
            edgecolor="#333333",
            linewidth=0.5,
            label=comparison_specs[key]["label"],
        )
        for key in comparison_keys
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)), bbox_to_anchor=(0.5, -0.04), frameon=True)
    fig.suptitle(f"{model_name}: selected-run changes versus native baseline", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    finish_figure(fig, figure_dir, "level_accuracy_matched_lex_deltas.svg", save_figures)


def print_summary(rows, datasets):
    for dataset in datasets.values():
        print(f"\n{dataset}")
        print("  run                      n   epoch             FPA               TICE              HCC diagnostics          status")
        subset = sorted((row for row in rows if row["dataset"] == dataset), key=lambda row: (-row["fpa"], row["tice"]))
        for row in subset:
            epoch = f"{row['best_epoch']:.1f} ± {row['best_epoch_std']:.1f}" if np.isfinite(row["best_epoch_std"]) else f"{row['best_epoch']:.0f}"
            fpa = f"{row['fpa']:.2f}% ± {row['fpa_std']:.2f}" if np.isfinite(row["fpa_std"]) else f"{row['fpa']:.2f}%"
            tice = f"{row['tice']:.2f}% ± {row['tice_std']:.2f}" if np.isfinite(row["tice_std"]) else f"{row['tice']:.2f}%"
            if row["family"] == "hcc":
                alpha = row["selected_hcc_alpha"]
                first = row["first_active_hcc_epoch"]
                diagnostics = f"selected α={alpha:.2f}; first active={first:.0f}" if np.isfinite(first) else f"selected α={alpha:.2f}; no active epoch"
            else:
                diagnostics = "—"
            status = "Pareto-optimal" if row["pareto_optimal"] else "dominated by " + ", ".join(row["dominated_by"])
            print(f"  {row['label']:<24} {len(row['seeds']):>2}  {epoch:<17} {fpa:<17} {tice:<17} {diagnostics:<24} {status}")
