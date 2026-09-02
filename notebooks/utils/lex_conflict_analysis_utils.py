"""Utilities for the multi-model lexicographic conflict notebook.

Four substrates are covered: H-CAST, Hier-COS (global and level softmax), HRN
under the level-conditional objective, and HT-CapsNet.

The analysis deliberately separates two questions:

1. Predictive: do baseline cosines predict the paired performance change
   under lexicographic training?
2. Mechanistic: within a lexicographic run, does the post-projection cosine
   verify that the configured projection actually removed the component?

The second question validates the operator but has no baseline counterfactual.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

# The decoder/readout contract is shared with the trade-off notebooks so that a
# view selected here means exactly what it means there.
from current_run_plot_utils import (  # noqa: E402
    DECODERS,
    READOUTS,
    SUBSPACE_SCORE_SPACES,
    selected_readout,
    validate_analysis_view,
)


COSINE_STAGES = {
    "mid / coarse": {
        "pre": ("cos_p123_mid_coarse", "cos_t1_mid_coarse"),
        "post": ("post_cos_p123_mid_coarse", "post_cos_t1_mid_proj_coarse"),
        "applied": (
            "post_projection_applied_p123_mid_coarse",
            "post_projection_applied_t1_mid_coarse",
        ),
    },
    "fine / higher": {
        # The compatibility key is needed because the raw fine/higher
        # resultant is not a pairwise canonical p123 key.
        "pre": ("cos_p123_fine_higher", "cos_t1_fine_higher"),
        "post": ("post_cos_p123_fine_higher", "post_cos_t1_fine_proj_higher"),
        "applied": (
            "post_projection_applied_p123_fine_higher",
            "post_projection_applied_t1_fine_higher",
        ),
    },
}


# The p12 block is reached by the coarse and middle objectives but not by the
# fine one. It is architecture-specific: only H-CAST has a non-empty p12 block.
# On Hier-COS, HRN and HT-CapsNet the block is empty and the logger writes a
# constant zero rather than a measured cosine, so the stage is emitted for
# H-CAST alone.
P12_COSINE_STAGE = {
    "p12 mid / coarse": {
        "pre": ("cos_p12_mid_coarse", "cos_t2_mid_coarse"),
        "post": ("post_cos_p12_mid_coarse", "post_cos_t2_mid_proj_coarse"),
        "applied": (
            "post_projection_applied_p12_mid_coarse",
            "post_projection_applied_t2_mid_coarse",
        ),
    },
}

PERFORMANCE_SPECS = {
    "TICE reduction (pp)": {
        "key": "tice_{decoder}",
        "scale": 100.0,
        "orientation": -1.0,
    },
    "FPA gain (pp)": {
        "key": "fpa_{decoder}",
        "scale": 100.0,
        "orientation": 1.0,
    },
    "fine-accuracy gain (pp)": {
        "key": "acc_level_{decoder}_2",
        "scale": 100.0,
        "orientation": 1.0,
    },
    "AHD gain": {
        "key": "ahd_{decoder}",
        "scale": 1.0,
        "orientation": -1.0,
    },
}

# The two responses carried into the signed joint margin.
JOINT_RESPONSE_LABELS = ("TICE reduction (pp)", "FPA gain (pp)")

# Top-down decoding walks the taxonomy, so its predicted path is consistent by
# construction and tice_topdown is identically zero. Under that decoder TICE
# carries no signal and is dropped rather than reported as a column of zeros.
DEGENERATE_TOPDOWN_RESPONSES = ("TICE reduction (pp)",)


def performance_specs(decoder: str = "independent") -> dict[str, dict]:
    """Return the performance metric specs resolved for one decoder."""

    validate_analysis_view(decoder=decoder)
    specs: dict[str, dict] = {}
    for label, spec in PERFORMANCE_SPECS.items():
        if decoder == "topdown" and label in DEGENERATE_TOPDOWN_RESPONSES:
            continue
        specs[label] = {**spec, "key": spec["key"].format(decoder=decoder)}
    return specs


def joint_response_labels(decoder: str = "independent") -> list[str]:
    """Return the joint-margin responses available under one decoder."""

    available = performance_specs(decoder)
    return [label for label in JOINT_RESPONSE_LABELS if label in available]



# Plotting order and identity of the substrates. Hier-COS is split by softmax
# variant because the two are separate training objectives, not two readouts of
# one run. Colours are Okabe-Ito so the five series stay separable.
SUBSTRATE_ORDER = (
    "H-CAST",
    "Hier-COS global",
    "Hier-COS level",
    "HRN",
    "HT-CapsNet",
)

SUBSTRATE_STYLES = {
    "H-CAST": {"color": "#009E73", "marker": "s"},
    "Hier-COS global": {"color": "#0072B2", "marker": "o"},
    "Hier-COS level": {"color": "#D55E00", "marker": "^"},
    "HRN": {"color": "#CC79A7", "marker": "D"},
    "HT-CapsNet": {"color": "#E69F00", "marker": "v"},
}


def substrate_order(frame: pd.DataFrame | None = None) -> list[str]:
    """Return the canonical substrate order, restricted to what a frame holds."""

    if frame is None or "substrate" not in frame:
        return list(SUBSTRATE_ORDER)
    present = set(frame["substrate"].dropna().unique())
    return [substrate for substrate in SUBSTRATE_ORDER if substrate in present]


def default_run_specs(prefer_matched_hcast: bool = True) -> list[dict]:
    """Return the current coarse-first comparison matrix over all substrates."""

    datasets = {
        "cifar100": "CIFAR-100",
        "cub200": "CUB-200",
        "aircraft": "Aircraft",
    }
    dataset_codes = {"cifar100": "CIF", "cub200": "CUB", "aircraft": "AIR"}
    specs: list[dict] = []

    for dataset_key, dataset_label in datasets.items():
        hcast_candidates = [f"hcast_{dataset_key}_nokl", f"hcast_{dataset_key}"]
        if not prefer_matched_hcast:
            hcast_candidates.reverse()
        specs.append(
            {
                "cell_id": f"HC-{dataset_key}",
                "short_label": dataset_codes[dataset_key],
                "family": "H-CAST",
                "substrate": "H-CAST",
                "dataset_key": dataset_key,
                "dataset": dataset_label,
                "baseline_candidates": hcast_candidates,
                "lex_run": f"hcast_{dataset_key}_lex_coarse_first",
            }
        )

        specs.append(
            {
                "cell_id": f"HG-{dataset_key}",
                "short_label": dataset_codes[dataset_key],
                "family": "Hier-COS",
                "substrate": "Hier-COS global",
                "dataset_key": dataset_key,
                "dataset": dataset_label,
                "baseline_candidates": [
                    f"hiercos_{dataset_key}_global_softmax_ce_reg_baseline_kl_leaf"
                ],
                "lex_run": (
                    f"hiercos_{dataset_key}_global_softmax_ce_reg_"
                    "lex_coarse_first_kl_leaf"
                ),
            }
        )

        level_frame = "block" if dataset_key == "cifar100" else "identity"
        specs.append(
            {
                "cell_id": f"HL-{dataset_key}",
                "short_label": dataset_codes[dataset_key],
                "family": "Hier-COS",
                "substrate": "Hier-COS level",
                "dataset_key": dataset_key,
                "dataset": dataset_label,
                "baseline_candidates": [
                    f"hiercos_{dataset_key}_level_softmax_ce_reg_"
                    f"baseline_kl_leaf_{level_frame}"
                ],
                "lex_run": (
                    f"hiercos_{dataset_key}_level_softmax_ce_reg_"
                    f"lex_coarse_first_kl_leaf_{level_frame}"
                ),
            }
        )

        # HRN is paired against the level-conditional control, which is the
        # objective the lex arm actually projects. The plain HRN runs are a
        # different objective and also differ in batch size, so they are not
        # offered as a fallback control here.
        specs.append(
            {
                "cell_id": f"HR-{dataset_key}",
                "short_label": dataset_codes[dataset_key],
                "family": "HRN",
                "substrate": "HRN",
                "dataset_key": dataset_key,
                "dataset": dataset_label,
                "baseline_candidates": [f"hrn_{dataset_key}_level_conditional"],
                "lex_run": f"hrn_{dataset_key}_level_conditional_lex_coarse_first",
            }
        )

        specs.append(
            {
                "cell_id": f"HT-{dataset_key}",
                "short_label": dataset_codes[dataset_key],
                "family": "HT-CapsNet",
                "substrate": "HT-CapsNet",
                "dataset_key": dataset_key,
                "dataset": dataset_label,
                "baseline_candidates": [f"capsnet_{dataset_key}"],
                "lex_run": f"ht_capsnet_{dataset_key}_lex_coarse_first",
            }
        )
    return specs


def _completed_seeds(outputs_root: Path, run_name: str) -> list[int]:
    run_dir = outputs_root / run_name
    if not run_dir.is_dir():
        return []
    seeds = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        required = ("config_resolved.yaml", "run_log.jsonl", "test_metrics.yaml")
        if all((seed_dir / name).is_file() for name in required):
            seeds.append(seed)
    return seeds


def _resolve_baseline(outputs_root: Path, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if _completed_seeds(outputs_root, candidate):
            return candidate
    return candidates[0]


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_epoch_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "epoch":
                rows.append(row)
    return rows


def _first_metric(metrics: Mapping, aliases: Sequence[str]) -> float | None:
    for key in aliases:
        if key in metrics:
            return float(metrics[key])
    return None


def _epoch_metric_summary(
    path: Path,
    aliases: Sequence[str],
    epoch_window: tuple[int, int],
) -> tuple[float, int]:
    start, end = epoch_window
    values = []
    for row in _read_epoch_rows(path):
        epoch = int(row["epoch"])
        if start <= epoch <= end:
            value = _first_metric(row.get("train_metrics", {}), aliases)
            if value is not None:
                values.append(value)
    if not values:
        return np.nan, 0
    return float(np.mean(values)), len(values)


def _epoch_metric_series(
    path: Path,
    aliases: Sequence[str],
    epoch_range: tuple[int, int],
) -> dict[int, float]:
    """Return the per-epoch values of a logged metric, keyed by epoch."""

    start, end = epoch_range
    series: dict[int, float] = {}
    for row in _read_epoch_rows(path):
        epoch = int(row["epoch"])
        if start <= epoch <= end:
            value = _first_metric(row.get("train_metrics", {}), aliases)
            if value is not None:
                series[epoch] = value
    return series


def _drop_path(mapping: dict, *path: str) -> None:
    current = mapping
    for key in path[:-1]:
        value = current.get(key)
        if not isinstance(value, dict):
            return
        current = value
    current.pop(path[-1], None)


def normalized_control_config(config: Mapping) -> dict:
    """Remove only operational and lex-switch fields before config comparison."""

    normalized = deepcopy(dict(config))
    for key in ("resume", "output_dir"):
        _drop_path(normalized, "train", key)
    _drop_path(normalized, "train", "lexicographic")
    # The selected support blocks are part of the lex mechanism. On a no-lex
    # control they affect diagnostics only, not the optimizer update, and older
    # configs may omit an effective p123-only selection.
    _drop_path(normalized, "train", "gradient_blocks")

    # Older baseline serialization wrote the disabled projection as
    # projection.feature_dim=0 while lex configs omitted the no-op section.
    projection = normalized.get("model", {}).get("projection")
    if projection == {"feature_dim": 0}:
        normalized["model"].pop("projection", None)
    return normalized


def _global_kl_value(config: Mapping) -> object:
    loss = config.get("model", {}).get("loss")
    return loss.get("globalkl") if isinstance(loss, dict) else None


def collect_analysis(
    outputs_root: Path,
    run_specs: Sequence[Mapping],
    epoch_window: tuple[int, int] = (81, 100),
    decoder: str = "independent",
    readout: str = "native",
    subspace_score_space: str = "probability",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load paired performance changes and baseline/lex cosine summaries.

    Performance is read under one decoder and one readout, matched across both
    arms, using the same resolution as the trade-off notebooks. Cosines are
    training-step diagnostics and do not depend on either choice.
    """

    decoder, readout, subspace_score_space = validate_analysis_view(
        decoder, readout, subspace_score_space
    )
    metric_specs = performance_specs(decoder)
    outputs_root = Path(outputs_root)
    seed_rows: list[dict] = []
    cosine_rows: list[dict] = []
    audit_rows: list[dict] = []

    for requested in run_specs:
        spec = dict(requested)
        baseline_run = _resolve_baseline(outputs_root, spec["baseline_candidates"])
        lex_run = spec["lex_run"]
        baseline_seeds = set(_completed_seeds(outputs_root, baseline_run))
        lex_seeds = set(_completed_seeds(outputs_root, lex_run))
        seeds = sorted(baseline_seeds & lex_seeds)
        if not seeds:
            audit_rows.append(
                {
                    **{key: spec[key] for key in ("cell_id", "family", "substrate", "dataset")},
                    "baseline_run": baseline_run,
                    "lex_run": lex_run,
                    "n": 0,
                    "config_matched": False,
                    "comparison_scope": "missing completed paired seeds",
                    "decoder": decoder,
                    "readout": readout,
                    "baseline_global_kl": None,
                    "lex_global_kl": None,
                }
            )
            continue

        base_config = _read_yaml(
            outputs_root / baseline_run / f"seed_{seeds[0]}" / "config_resolved.yaml"
        )
        lex_config = _read_yaml(
            outputs_root / lex_run / f"seed_{seeds[0]}" / "config_resolved.yaml"
        )
        config_matched = normalized_control_config(base_config) == normalized_control_config(
            lex_config
        )
        comparison_scope = (
            "projection-only matched comparison"
            if config_matched
            else "descriptive lex package; inspect config difference"
        )
        common = {
            **{
                key: spec[key]
                for key in (
                    "cell_id",
                    "short_label",
                    "family",
                    "substrate",
                    "dataset_key",
                    "dataset",
                )
            },
            "baseline_run": baseline_run,
            "lex_run": lex_run,
            "config_matched": config_matched,
            "comparison_scope": comparison_scope,
            "decoder": decoder,
            "readout": readout,
        }
        audit_rows.append(
            {
                **common,
                "n": len(seeds),
                "baseline_global_kl": _global_kl_value(base_config),
                "lex_global_kl": _global_kl_value(lex_config),
            }
        )

        for seed in seeds:
            base_dir = outputs_root / baseline_run / f"seed_{seed}"
            lex_dir = outputs_root / lex_run / f"seed_{seed}"
            base_view = selected_readout(
                base_dir, decoder, readout, subspace_score_space
            )
            lex_view = selected_readout(lex_dir, decoder, readout, subspace_score_space)
            base_metrics = base_view["metrics"]
            lex_metrics = lex_view["metrics"]

            seed_row = {
                **common,
                "seed": seed,
                "baseline_best_epoch": base_view["best_epoch"],
                "lex_best_epoch": lex_view["best_epoch"],
            }
            for label, metric_spec in metric_specs.items():
                baseline_value = float(base_metrics[metric_spec["key"]])
                lex_value = float(lex_metrics[metric_spec["key"]])
                seed_row[label] = (
                    metric_spec["orientation"]
                    * metric_spec["scale"]
                    * (lex_value - baseline_value)
                )
            seed_rows.append(seed_row)

            for stage, keys in COSINE_STAGES.items():
                baseline_pre, baseline_n = _epoch_metric_summary(
                    base_dir / "run_log.jsonl", keys["pre"], epoch_window
                )
                lex_pre, lex_n = _epoch_metric_summary(
                    lex_dir / "run_log.jsonl", keys["pre"], epoch_window
                )
                lex_post, post_n = _epoch_metric_summary(
                    lex_dir / "run_log.jsonl", keys["post"], epoch_window
                )
                applied, applied_n = _epoch_metric_summary(
                    lex_dir / "run_log.jsonl", keys["applied"], epoch_window
                )
                cosine_rows.append(
                    {
                        **common,
                        "seed": seed,
                        "stage": stage,
                        "baseline_pre_cos": baseline_pre,
                        "lex_pre_cos": lex_pre,
                        "lex_post_cos": lex_post,
                        "projection_applied_fraction": applied,
                        "baseline_logged_epochs": baseline_n,
                        "lex_pre_logged_epochs": lex_n,
                        "lex_post_logged_epochs": post_n,
                        "applied_logged_epochs": applied_n,
                    }
                )

    return pd.DataFrame(seed_rows), pd.DataFrame(cosine_rows), pd.DataFrame(audit_rows)


def summarize_cells(seed_df: pd.DataFrame, cosine_df: pd.DataFrame) -> pd.DataFrame:
    """Return cell means and paired-seed sample standard deviations."""

    keys = [
        "cell_id",
        "short_label",
        "family",
        "substrate",
        "dataset",
        "baseline_run",
        "lex_run",
        "config_matched",
        "comparison_scope",
        "decoder",
        "readout",
    ]
    performance_columns = [
        label for label in PERFORMANCE_SPECS if label in seed_df.columns
    ]
    performance = seed_df.groupby(keys, dropna=False)[performance_columns].agg(
        ["mean", "std", "count"]
    )
    performance.columns = [f"{column}__{stat}" for column, stat in performance.columns]
    performance = performance.reset_index()

    cosine_columns = [
        "baseline_pre_cos",
        "lex_pre_cos",
        "lex_post_cos",
        "projection_applied_fraction",
    ]
    cosine = cosine_df.groupby([*keys, "stage"], dropna=False)[cosine_columns].agg(
        ["mean", "std", "count"]
    )
    cosine.columns = [f"{column}__{stat}" for column, stat in cosine.columns]
    cosine = cosine.reset_index()
    return cosine.merge(performance, on=keys, validate="many_to_one")


def hcast_p12_diagnostics(
    outputs_root: Path,
    audit_df: pd.DataFrame,
    epoch_window: tuple[int, int] = (81, 100),
) -> pd.DataFrame:
    """Return H-CAST's architecture-specific p12 mid/coarse projection audit."""

    outputs_root = Path(outputs_root)
    rows = []
    aliases = {
        "pre": ("cos_p12_mid_coarse", "cos_t2_mid_coarse"),
        "post": ("post_cos_p12_mid_coarse", "post_cos_t2_mid_proj_coarse"),
        "applied": (
            "post_projection_applied_p12_mid_coarse",
            "post_projection_applied_t2_mid_coarse",
        ),
    }
    for _, audit in audit_df[audit_df["family"] == "H-CAST"].iterrows():
        for seed in _completed_seeds(outputs_root, audit["lex_run"]):
            log_path = outputs_root / audit["lex_run"] / f"seed_{seed}" / "run_log.jsonl"
            pre, _ = _epoch_metric_summary(log_path, aliases["pre"], epoch_window)
            post, _ = _epoch_metric_summary(log_path, aliases["post"], epoch_window)
            applied, _ = _epoch_metric_summary(log_path, aliases["applied"], epoch_window)
            rows.append(
                {
                    "dataset": audit["dataset"],
                    "seed": seed,
                    "p12 pre mid/coarse cosine": pre,
                    "p12 post mid/coarse cosine": post,
                    "projection applied fraction": applied,
                }
            )
    return pd.DataFrame(rows)


def cosine_trajectories(
    outputs_root: Path,
    run_specs: Sequence[Mapping],
    epoch_range: tuple[int, int] = (1, 100),
) -> pd.DataFrame:
    """Return per-epoch baseline, lex pre- and lex post-projection cosines.

    One row per paired seed, stage and epoch. The baseline column comes from the
    no-lex arm and the two lex columns from the lex arm of the same pair, so the
    three are directly comparable at equal step count. The p12 stage is emitted
    for H-CAST only, because on every other substrate here that block is empty.
    """

    outputs_root = Path(outputs_root)
    rows = []
    for requested in run_specs:
        spec = dict(requested)
        baseline_run = _resolve_baseline(outputs_root, spec["baseline_candidates"])
        lex_run = spec["lex_run"]
        seeds = sorted(
            set(_completed_seeds(outputs_root, baseline_run))
            & set(_completed_seeds(outputs_root, lex_run))
        )
        stages = dict(COSINE_STAGES)
        if spec["family"] == "H-CAST":
            stages.update(P12_COSINE_STAGE)

        for seed in seeds:
            baseline_log = outputs_root / baseline_run / f"seed_{seed}" / "run_log.jsonl"
            lex_log = outputs_root / lex_run / f"seed_{seed}" / "run_log.jsonl"
            for stage, keys in stages.items():
                baseline_series = _epoch_metric_series(baseline_log, keys["pre"], epoch_range)
                lex_pre_series = _epoch_metric_series(lex_log, keys["pre"], epoch_range)
                lex_post_series = _epoch_metric_series(lex_log, keys["post"], epoch_range)
                applied_series = _epoch_metric_series(lex_log, keys["applied"], epoch_range)
                for epoch in sorted(set(baseline_series) & set(lex_pre_series)):
                    rows.append(
                        {
                            **{
                                key: spec[key]
                                for key in (
                                    "cell_id",
                                    "short_label",
                                    "family",
                                    "substrate",
                                    "dataset",
                                )
                            },
                            "baseline_run": baseline_run,
                            "lex_run": lex_run,
                            "seed": seed,
                            "stage": stage,
                            "epoch": epoch,
                            "baseline_cos": baseline_series[epoch],
                            "lex_pre_cos": lex_pre_series[epoch],
                            "lex_post_cos": lex_post_series.get(epoch, np.nan),
                            "projection_applied": applied_series.get(epoch, np.nan),
                        }
                    )
    return pd.DataFrame(rows)


def window_joint_margins(
    outputs_root: Path,
    run_specs: Sequence[Mapping],
    windows: Iterable[tuple[str, tuple[int, int]]],
    decoder: str = "independent",
    readout: str = "native",
    subspace_score_space: str = "probability",
) -> pd.DataFrame:
    """Return dataset-level -cosine * improvement margins by epoch window."""

    rows = []
    responses = joint_response_labels(decoder)
    for window_label, epoch_window in windows:
        seed_df, cosine_df, _ = collect_analysis(
            outputs_root,
            run_specs,
            epoch_window,
            decoder=decoder,
            readout=readout,
            subspace_score_space=subspace_score_space,
        )
        cells = summarize_cells(seed_df, cosine_df)
        for substrate in substrate_order(cells):
            substrate_cells = cells[cells["substrate"] == substrate]
            for stage, stage_cells in substrate_cells.groupby("stage", sort=False):
                for _, cell in stage_cells.iterrows():
                    cosine = float(cell["baseline_pre_cos__mean"])
                    for response in responses:
                        improvement = float(cell[f"{response}__mean"])
                        if not np.isfinite(cosine) or not np.isfinite(improvement):
                            continue
                        rows.append(
                            {
                                "window": window_label,
                                "substrate": substrate,
                                "comparison_scope": cell["comparison_scope"],
                                "stage": stage,
                                "response": response,
                                "dataset": cell["dataset"],
                                "baseline_cosine": cosine,
                                "improvement": improvement,
                                "joint_margin": -cosine * improvement,
                            }
                        )
    return pd.DataFrame(rows)
