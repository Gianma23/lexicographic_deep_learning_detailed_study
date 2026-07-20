from __future__ import annotations

import html
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from datasets.aircraft import load_official_aircraft_hierarchy
from datasets.cifar100 import (
    B_CNN_COARSE_TO_SUPER,
    load_official_cifar100_fine_to_coarse,
)
from datasets.cub_tree import TREES as CUB_TREES


load_dotenv(
    Path(os.environ.get("PROJECT_ENV_FILE", REPO_ROOT / ".env")).expanduser(),
    override=False,
)


OUTPUTS_ROOT = Path(os.environ.get("OUTPUTS_ROOT", "/scratch/g.saggini1/outputs"))
FIG_ROOT = REPO_ROOT / "docs" / "figures" / "hcc_internal_insights_slides"

SWITCH_EPOCH = 100
SWITCH_LINE_EPOCH = 101
WINDOW_START = 95
WINDOW_END = 115

RUNS = {
    "cifar100": {
        "label": "CIFAR-100",
        "baseline": OUTPUTS_ROOT / "hcast_cifar100",
        "step100": OUTPUTS_ROOT / "hcast_hcc_cifar100_step_100epochs",
    },
    "cub200": {
        "label": "CUB-200",
        "baseline": OUTPUTS_ROOT / "hcast_cub200",
        "step100": OUTPUTS_ROOT / "hcast_hcc_cub200_step_100epochs",
    },
    "aircraft": {
        "label": "Aircraft",
        "baseline": OUTPUTS_ROOT / "hcast_aircraft",
        "step100": OUTPUTS_ROOT / "hcast_hcc_aircraft_step_100epochs",
    },
}

COLORS = {
    "fpa_independent": "#c44e52",
    "fpa_topdown": "#4c72b0",
    "acc_level_independent_2": "#55a868",
    "tice_independent": "#8172b2",
    "proj_flip_rate_level_2": "#dd8452",
    "gt_parent_mass_pre_l2": "#64b5cd",
    "gt_parent_mass_post_l2": "#cc6677",
    "epoch100": "#bdbdbd",
    "epoch115": "#4c72b0",
}

FONT_FAMILY = "Arial, Helvetica, sans-serif"


def _load_run_metrics(run_dir: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with (run_dir / "run_log.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") != "epoch":
                continue
            metrics = event.get("val_metrics") or {}
            row: Dict[str, float] = {"epoch": int(event["epoch"])}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key] = float(value)
            rows.append(row)
    return rows


def _metric_series(rows: Iterable[Dict[str, float]], key: str, start: int, end: int) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for row in rows:
        epoch = int(row["epoch"])
        if epoch < start or epoch > end or key not in row:
            continue
        xs.append(epoch)
        ys.append(float(row[key]))
    return xs, ys


def _row_for_epoch(rows: Iterable[Dict[str, float]], epoch: int) -> Dict[str, float]:
    for row in rows:
        if int(row["epoch"]) == epoch:
            return row
    raise KeyError(f"Epoch {epoch} not found.")


def _last_row(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("Expected non-empty run rows.")
    return rows[-1]


def _required_dataset_root(env_name: str) -> Path:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        raise RuntimeError(
            f"{env_name} must point to the official dataset download when generating taxonomy figures."
        )
    return Path(raw).expanduser()


def _fanout_summary(
    dataset_label: str,
    middle_to_coarse: Dict[int, int],
    fine_to_middle: Dict[int, int],
) -> Dict[str, object]:
    fine_per_middle = Counter(fine_to_middle.values())
    middle_per_coarse = Counter(middle_to_coarse.values())
    size_hist = Counter(fine_per_middle.values())
    singleton_middle = sum(1 for size in fine_per_middle.values() if size == 1)
    return {
        "label": dataset_label,
        "num_coarse": len(set(middle_to_coarse.values())),
        "num_middle": len(middle_to_coarse),
        "num_fine": len(fine_to_middle),
        "fine_per_middle_hist": dict(sorted(size_hist.items())),
        "fine_per_middle_min": min(fine_per_middle.values()),
        "fine_per_middle_mean": sum(fine_per_middle.values()) / len(fine_per_middle),
        "fine_per_middle_max": max(fine_per_middle.values()),
        "middle_per_coarse_min": min(middle_per_coarse.values()),
        "middle_per_coarse_mean": sum(middle_per_coarse.values()) / len(middle_per_coarse),
        "middle_per_coarse_max": max(middle_per_coarse.values()),
        "singleton_middle_count": singleton_middle,
    }


def load_fanout_summaries() -> Dict[str, Dict[str, object]]:
    cifar_fine_to_middle = load_official_cifar100_fine_to_coarse(
        _required_dataset_root("CIFAR100_ROOT")
    )
    aircraft_hierarchy = load_official_aircraft_hierarchy(
        _required_dataset_root("AIRCRAFT_ROOT")
    )

    return {
        "cifar100": _fanout_summary(
            dataset_label="CIFAR-100",
            middle_to_coarse={
                idx: int(parent)
                for idx, parent in enumerate(B_CNN_COARSE_TO_SUPER)
            },
            fine_to_middle={idx: int(parent) for idx, parent in enumerate(cifar_fine_to_middle)},
        ),
        "cub200": _fanout_summary(
            dataset_label="CUB-200",
            middle_to_coarse={int(tree[2]) - 1: int(tree[1]) - 1 for tree in CUB_TREES},
            fine_to_middle={idx: int(tree[2]) - 1 for idx, tree in enumerate(CUB_TREES)},
        ),
        "aircraft": _fanout_summary(
            dataset_label="Aircraft",
            middle_to_coarse=dict(aircraft_hierarchy.family_to_manufacturer),
            fine_to_middle=dict(aircraft_hierarchy.variant_to_family),
        ),
    }


class SvgCanvas:
    def __init__(self, width: int, height: int, background: str = "white") -> None:
        self.width = width
        self.height = height
        self.background = background
        self.items: List[str] = []

    def add(self, item: str) -> None:
        self.items.append(item)

    def save(self, path: Path) -> None:
        path.write_text(self.render(), encoding="utf-8")

    def render(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">'
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="{self.background}" />'
            + "".join(self.items)
            + "</svg>"
        )


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 12,
    anchor: str = "start",
    weight: str = "normal",
    fill: str = "#111111",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT_FAMILY}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}"{transform}>{_escape(value)}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = "#333333", width: float = 1.0,
          dash: str | None = None, opacity: float = 1.0) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash is not None else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
        f'stroke-width="{width:.1f}" opacity="{opacity:.3f}"{dash_attr} />'
    )


def _rect(x: float, y: float, width: float, height: float, *, fill: str = "none", stroke: str = "none",
          stroke_width: float = 0.0, opacity: float = 1.0, rx: float = 0.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}" opacity="{opacity:.3f}" rx="{rx:.1f}" />'
    )


def _circle(cx: float, cy: float, r: float, *, fill: str, stroke: str = "white", stroke_width: float = 1.0) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}" />'
    )


def _polyline(points: Sequence[Tuple[float, float]], *, stroke: str, width: float = 2.0, fill: str = "none") -> str:
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{encoded}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.1f}" />'


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if abs(src_max - src_min) < 1e-12:
        return 0.5 * (dst_min + dst_max)
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + (ratio * (dst_max - dst_min))


def _pct_label(value: float) -> str:
    return f"{100.0 * value:.0f}%"


def _decimal_label(value: float) -> str:
    return f"{value:.2f}"


def _draw_title(canvas: SvgCanvas, title: str, subtitle: str | None = None) -> None:
    canvas.add(_text(canvas.width / 2, 34, title, size=22, anchor="middle", weight="bold"))
    if subtitle:
        canvas.add(_text(canvas.width / 2, 58, subtitle, size=12, anchor="middle", fill="#444444"))


def _draw_legend(canvas: SvgCanvas, entries: Sequence[Tuple[str, str]], *, x: float, y: float, spacing: float = 190.0) -> None:
    for idx, (label, color) in enumerate(entries):
        lx = x + (idx * spacing)
        canvas.add(_line(lx, y, lx + 24, y, stroke=color, width=3.0))
        canvas.add(_circle(lx + 12, y, 3.2, fill=color, stroke=color, stroke_width=0.5))
        canvas.add(_text(lx + 32, y + 4, label, size=12))


def _draw_note_box(canvas: SvgCanvas, x: float, y: float, lines: Sequence[str], *, width: float = 160.0) -> None:
    height = 18 + (16 * len(lines))
    canvas.add(_rect(x, y, width, height, fill="white", stroke="#cccccc", stroke_width=1.0, opacity=0.9, rx=6.0))
    for idx, line in enumerate(lines):
        canvas.add(_text(x + 8, y + 18 + (16 * idx), line, size=11, fill="#333333"))


def _draw_line_panel(
    canvas: SvgCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    rows: Sequence[Dict[str, float]],
    series: Sequence[Tuple[str, str, str]],
    y_min: float,
    y_max: float,
    y_ticks: Sequence[float],
    y_formatter,
    note_lines: Sequence[str] | None = None,
) -> None:
    canvas.add(_rect(x, y, width, height, fill="white", stroke="#d6d6d6", stroke_width=1.0, rx=8.0))
    canvas.add(_text(x + (width / 2), y + 28, title, size=16, anchor="middle", weight="bold"))

    plot_left = x + 58
    plot_right = x + width - 18
    plot_top = y + 48
    plot_bottom = y + height - 46
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    for tick in y_ticks:
        ty = _scale(tick, y_min, y_max, plot_bottom, plot_top)
        canvas.add(_line(plot_left, ty, plot_right, ty, stroke="#e8e8e8", width=1.0))
        canvas.add(_text(plot_left - 8, ty + 4, y_formatter(tick), size=11, anchor="end", fill="#555555"))

    x_ticks = [95, 100, 101, 105, 110, 115]
    for tick in x_ticks:
        tx = _scale(tick, WINDOW_START, WINDOW_END, plot_left, plot_right)
        canvas.add(_line(tx, plot_top, tx, plot_bottom, stroke="#f1f1f1", width=1.0))
        canvas.add(_text(tx, plot_bottom + 18, str(tick), size=11, anchor="middle", fill="#555555"))

    switch_x = _scale(SWITCH_LINE_EPOCH, WINDOW_START, WINDOW_END, plot_left, plot_right)
    canvas.add(_line(switch_x, plot_top, switch_x, plot_bottom, stroke="#111111", width=1.4, dash="5,4", opacity=0.8))

    canvas.add(_line(plot_left, plot_bottom, plot_right, plot_bottom, stroke="#444444", width=1.3))
    canvas.add(_line(plot_left, plot_top, plot_left, plot_bottom, stroke="#444444", width=1.3))

    for key, _, color in series:
        xs, ys = _metric_series(rows, key, WINDOW_START, WINDOW_END)
        if not xs:
            continue
        points = [
            (
                _scale(epoch, WINDOW_START, WINDOW_END, plot_left, plot_right),
                _scale(value, y_min, y_max, plot_bottom, plot_top),
            )
            for epoch, value in zip(xs, ys)
        ]
        canvas.add(_polyline(points, stroke=color, width=2.4))
        for px, py in points:
            canvas.add(_circle(px, py, 2.8, fill=color, stroke=color, stroke_width=0.6))

    if note_lines:
        _draw_note_box(canvas, plot_right - 166, plot_top + 8, note_lines)


def _draw_grouped_bar_panel(
    canvas: SvgCanvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    categories: Sequence[str],
    groups: Sequence[Tuple[str, str, Sequence[float]]],
    y_min: float,
    y_max: float,
    y_ticks: Sequence[float],
    y_formatter,
) -> None:
    canvas.add(_rect(x, y, width, height, fill="white", stroke="#d6d6d6", stroke_width=1.0, rx=8.0))
    canvas.add(_text(x + (width / 2), y + 28, title, size=16, anchor="middle", weight="bold"))

    plot_left = x + 58
    plot_right = x + width - 18
    plot_top = y + 48
    plot_bottom = y + height - 64
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    for tick in y_ticks:
        ty = _scale(tick, y_min, y_max, plot_bottom, plot_top)
        canvas.add(_line(plot_left, ty, plot_right, ty, stroke="#e8e8e8", width=1.0))
        canvas.add(_text(plot_left - 8, ty + 4, y_formatter(tick), size=11, anchor="end", fill="#555555"))

    canvas.add(_line(plot_left, plot_bottom, plot_right, plot_bottom, stroke="#444444", width=1.3))
    canvas.add(_line(plot_left, plot_top, plot_left, plot_bottom, stroke="#444444", width=1.3))

    num_categories = len(categories)
    num_groups = len(groups)
    slot_width = plot_width / max(num_categories, 1)
    bar_width = min(34.0, (slot_width * 0.72) / max(num_groups, 1))

    for cat_idx, category in enumerate(categories):
        center = plot_left + ((cat_idx + 0.5) * slot_width)
        canvas.add(_text(center, plot_bottom + 20, category, size=11, anchor="middle", fill="#555555"))
        for group_idx, (_, color, values) in enumerate(groups):
            value = values[cat_idx]
            bar_x = center - ((num_groups * bar_width) / 2.0) + (group_idx * bar_width)
            bar_y = _scale(value, y_min, y_max, plot_bottom, plot_top)
            canvas.add(
                _rect(bar_x, bar_y, bar_width - 2.0, plot_bottom - bar_y, fill=color, stroke="none", opacity=0.92, rx=2.0)
            )


def create_switch_window_accuracy(step100_runs: Dict[str, List[Dict[str, float]]]) -> None:
    canvas = SvgCanvas(1800, 540)
    _draw_title(
        canvas,
        "Step-100 Hard HCC Switch: Accuracy and Path Metrics",
        "Epochs 95-115 from validation logs. Dashed line marks the first hard-projection epoch.",
    )
    _draw_legend(
        canvas,
        [
            ("FPA independent", COLORS["fpa_independent"]),
            ("FPA top-down", COLORS["fpa_topdown"]),
            ("L2 independent acc", COLORS["acc_level_independent_2"]),
        ],
        x=430,
        y=86,
    )

    panel_width = 560
    panel_gap = 20
    base_x = 30
    for idx, dataset_key in enumerate(["cifar100", "cub200", "aircraft"]):
        panel_x = base_x + idx * (panel_width + panel_gap)
        note = []
        if dataset_key == "cifar100":
            note = ["small shift", "limited rewrite"]
        elif dataset_key == "cub200":
            note = ["severe collapse", "parent mostly right"]
        else:
            note = ["partial collapse", "slow recovery"]
        _draw_line_panel(
            canvas,
            panel_x,
            110,
            panel_width,
            390,
            title=RUNS[dataset_key]["label"],
            rows=step100_runs[dataset_key],
            series=[
                ("fpa_independent", "FPA independent", COLORS["fpa_independent"]),
                ("fpa_topdown", "FPA top-down", COLORS["fpa_topdown"]),
                ("acc_level_independent_2", "L2 independent acc", COLORS["acc_level_independent_2"]),
            ],
            y_min=0.1,
            y_max=0.86,
            y_ticks=[0.2, 0.4, 0.6, 0.8],
            y_formatter=_pct_label,
            note_lines=note,
        )
    canvas.save(FIG_ROOT / "switch_window_accuracy.svg")


def create_switch_window_consistency(step100_runs: Dict[str, List[Dict[str, float]]]) -> None:
    canvas = SvgCanvas(1800, 540)
    _draw_title(
        canvas,
        "Hard HCC Rewrite Intensity and Inconsistency",
        "High flip rate means the projection often changes the fine argmax. High TICE means predictions are more inconsistent.",
    )
    _draw_legend(
        canvas,
        [
            ("TICE independent", COLORS["tice_independent"]),
            ("Fine argmax flip rate", COLORS["proj_flip_rate_level_2"]),
        ],
        x=560,
        y=86,
    )

    panel_width = 560
    panel_gap = 20
    base_x = 30
    for idx, dataset_key in enumerate(["cifar100", "cub200", "aircraft"]):
        panel_x = base_x + idx * (panel_width + panel_gap)
        _draw_line_panel(
            canvas,
            panel_x,
            110,
            panel_width,
            390,
            title=RUNS[dataset_key]["label"],
            rows=step100_runs[dataset_key],
            series=[
                ("tice_independent", "TICE independent", COLORS["tice_independent"]),
                ("proj_flip_rate_level_2", "Fine argmax flip rate", COLORS["proj_flip_rate_level_2"]),
            ],
            y_min=0.0,
            y_max=0.85,
            y_ticks=[0.0, 0.2, 0.4, 0.6, 0.8],
            y_formatter=_pct_label,
        )
    canvas.save(FIG_ROOT / "switch_window_consistency.svg")


def create_parent_mass_reallocation(step100_runs: Dict[str, List[Dict[str, float]]]) -> None:
    canvas = SvgCanvas(1800, 540)
    _draw_title(
        canvas,
        "Raw Fine Mass Keeps Rising, HCC Post-Mass Gets Clamped to Parent Budget",
        "The gap between pre and post mass shows disagreement between the fine head and the middle-level budget enforced by HCC.",
    )
    _draw_legend(
        canvas,
        [
            ("GT parent mass pre", COLORS["gt_parent_mass_pre_l2"]),
            ("GT parent mass post", COLORS["gt_parent_mass_post_l2"]),
        ],
        x=560,
        y=86,
    )

    panel_width = 560
    panel_gap = 20
    base_x = 30
    for idx, dataset_key in enumerate(["cifar100", "cub200", "aircraft"]):
        panel_x = base_x + idx * (panel_width + panel_gap)
        _draw_line_panel(
            canvas,
            panel_x,
            110,
            panel_width,
            390,
            title=RUNS[dataset_key]["label"],
            rows=step100_runs[dataset_key],
            series=[
                ("gt_parent_mass_pre_l2", "GT parent mass pre", COLORS["gt_parent_mass_pre_l2"]),
                ("gt_parent_mass_post_l2", "GT parent mass post", COLORS["gt_parent_mass_post_l2"]),
            ],
            y_min=0.15,
            y_max=0.9,
            y_ticks=[0.2, 0.4, 0.6, 0.8],
            y_formatter=_pct_label,
        )
    canvas.save(FIG_ROOT / "parent_mass_reallocation.svg")


def create_conditioned_leaf_accuracy(step100_runs: Dict[str, List[Dict[str, float]]]) -> None:
    canvas = SvgCanvas(1800, 540)
    _draw_title(
        canvas,
        "Correct Parent Does Not Guarantee Correct Independent Fine Argmax",
        "Compare the last pre-switch epoch (100) to the end of the run (115).",
    )
    _draw_legend(
        canvas,
        [
            ("Epoch 100", COLORS["epoch100"]),
            ("Epoch 115", COLORS["epoch115"]),
        ],
        x=700,
        y=86,
        spacing=150,
    )

    panel_width = 560
    panel_gap = 20
    base_x = 30
    categories = ["L1 support\n(ind)", "L2 acc\n| L1 ok (ind)", "L1 support\n(td)", "L2 acc\n| L1 ok (td)"]
    for idx, dataset_key in enumerate(["cifar100", "cub200", "aircraft"]):
        before = _row_for_epoch(step100_runs[dataset_key], SWITCH_EPOCH)
        after = _row_for_epoch(step100_runs[dataset_key], WINDOW_END)
        groups = [
            (
                "Epoch 100",
                COLORS["epoch100"],
                [
                    before["support_l1_ind_correct"],
                    before["acc_l2_ind_given_l1_correct"],
                    before["support_l1_td_correct"],
                    before["acc_l2_td_given_l1_correct"],
                ],
            ),
            (
                "Epoch 115",
                COLORS["epoch115"],
                [
                    after["support_l1_ind_correct"],
                    after["acc_l2_ind_given_l1_correct"],
                    after["support_l1_td_correct"],
                    after["acc_l2_td_given_l1_correct"],
                ],
            ),
        ]
        panel_x = base_x + idx * (panel_width + panel_gap)
        _draw_grouped_bar_panel(
            canvas,
            panel_x,
            110,
            panel_width,
            390,
            title=RUNS[dataset_key]["label"],
            categories=categories,
            groups=groups,
            y_min=0.0,
            y_max=1.0,
            y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
            y_formatter=_pct_label,
        )
    canvas.save(FIG_ROOT / "conditioned_leaf_accuracy.svg")


def create_taxonomy_fanout(fanout: Dict[str, Dict[str, object]]) -> None:
    canvas = SvgCanvas(1800, 540)
    _draw_title(
        canvas,
        "Taxonomy Fan-Out Uniformity Explains Why CIFAR Tolerates HCC Better",
        "CIFAR has uniform five-way middle nodes. CUB and Aircraft are highly uneven, so parent-budget clamping affects global leaf competition much more.",
    )

    panel_width = 560
    panel_gap = 20
    base_x = 30
    for idx, dataset_key in enumerate(["cifar100", "cub200", "aircraft"]):
        summary = fanout[dataset_key]
        hist = summary["fine_per_middle_hist"]
        categories = [str(x) for x in hist.keys()]
        groups = [("Middle parents", "#4c72b0", list(hist.values()))]
        panel_x = base_x + idx * (panel_width + panel_gap)
        _draw_grouped_bar_panel(
            canvas,
            panel_x,
            110,
            panel_width,
            390,
            title=str(summary["label"]),
            categories=categories,
            groups=groups,
            y_min=0.0,
            y_max=max(hist.values()) + 2.0,
            y_ticks=[0.0, max(hist.values()) / 3.0, 2.0 * max(hist.values()) / 3.0, float(max(hist.values()))],
            y_formatter=_decimal_label,
        )
        note_lines = [
            f"mean={summary['fine_per_middle_mean']:.2f}",
            f"min={summary['fine_per_middle_min']}",
            f"max={summary['fine_per_middle_max']}",
        ]
        if int(summary["singleton_middle_count"]) > 0:
            note_lines.append(f"singletons={summary['singleton_middle_count']}")
        _draw_note_box(canvas, panel_x + panel_width - 165, 168, note_lines)
    canvas.save(FIG_ROOT / "taxonomy_fanout.svg")


def write_summary_text(
    fanout: Dict[str, Dict[str, object]],
    baseline_runs: Dict[str, List[Dict[str, float]]],
    step100_runs: Dict[str, List[Dict[str, float]]],
) -> None:
    summary_path = FIG_ROOT / "README.txt"
    lines: List[str] = []
    lines.append("HCC internal insights slide figures")
    lines.append("")
    lines.append("Files:")
    lines.append("  switch_window_accuracy.svg")
    lines.append("  switch_window_consistency.svg")
    lines.append("  parent_mass_reallocation.svg")
    lines.append("  conditioned_leaf_accuracy.svg")
    lines.append("  taxonomy_fanout.svg")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - Hard HCC is off through epoch 100 and turns on at epoch 101.")
    lines.append("  - Use val logs for switch analysis. test_metrics.yaml stores validation-best checkpoint metrics, which can predate the switch.")
    lines.append("")
    for dataset_key in ["cifar100", "cub200", "aircraft"]:
        label = RUNS[dataset_key]["label"]
        baseline_row = _last_row(baseline_runs[dataset_key])
        before = _row_for_epoch(step100_runs[dataset_key], SWITCH_EPOCH)
        after = _row_for_epoch(step100_runs[dataset_key], WINDOW_END)
        lines.append(label)
        lines.append(
            f"  fan-out fine/middle min-mean-max: "
            f"{fanout[dataset_key]['fine_per_middle_min']}-"
            f"{fanout[dataset_key]['fine_per_middle_mean']:.2f}-"
            f"{fanout[dataset_key]['fine_per_middle_max']}"
        )
        lines.append(
            f"  FPA_ind baseline={100.0 * baseline_row['fpa_independent']:.2f}%, "
            f"epoch100={100.0 * before['fpa_independent']:.2f}%, "
            f"epoch115={100.0 * after['fpa_independent']:.2f}%"
        )
        lines.append(
            f"  TICE_ind epoch100={100.0 * before['tice_independent']:.2f}%, "
            f"epoch115={100.0 * after['tice_independent']:.2f}%, "
            f"flip_rate_epoch115={100.0 * after['proj_flip_rate_level_2']:.2f}%"
        )
        lines.append(
            f"  acc_l2_ind|l1_ok epoch100={100.0 * before['acc_l2_ind_given_l1_correct']:.2f}%, "
            f"epoch115={100.0 * after['acc_l2_ind_given_l1_correct']:.2f}%, "
            f"td_epoch115={100.0 * after['acc_l2_td_given_l1_correct']:.2f}%"
        )
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    baseline_runs = {dataset_key: _load_run_metrics(cfg["baseline"]) for dataset_key, cfg in RUNS.items()}
    step100_runs = {dataset_key: _load_run_metrics(cfg["step100"]) for dataset_key, cfg in RUNS.items()}
    fanout = load_fanout_summaries()

    create_switch_window_accuracy(step100_runs)
    create_switch_window_consistency(step100_runs)
    create_parent_mass_reallocation(step100_runs)
    create_conditioned_leaf_accuracy(step100_runs)
    create_taxonomy_fanout(fanout)
    write_summary_text(fanout, baseline_runs, step100_runs)

    print(f"Wrote slide figures to {FIG_ROOT}")


if __name__ == "__main__":
    main()
