from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import yaml


SEED_DIR_PATTERN = re.compile(r"^seed_(-?\d+)$")
_CONFIG_IGNORED_PATHS = (
    ("train", "seed"),
    ("train", "output_dir"),
    ("train", "resume"),
)
# Aggregate loss terms that are the weighted sum of a per-level term the log
# also carries. Rebuilding them from the unweighted parts keeps the aggregate
# panel and the per-level panel telling the same story; `total` is deliberately
# absent because it is the objective, weights and all.
_UNWEIGHTABLE_AGGREGATES = {
    "ce": "ce_level_",
    "subspace_soft_cross_entropy": "subspace_soft_cross_entropy_level_",
    "subspace_target_kl": "subspace_target_kl_level_",
}
_EPOCH_MAPPING_KEYS = (
    "train_losses",
    "val_losses",
    "train_metrics",
    "val_metrics",
    "train_metrics_norm",
    "val_metrics_norm",
)


def sample_stats(values: Sequence[Any]) -> Tuple[float, float, int]:
    finite: List[float] = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            finite.append(parsed)

    if not finite:
        return float("nan"), float("nan"), 0

    arr = np.asarray(finite, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    return mean, std, int(arr.size)


def aggregate_numeric_mappings(
    mappings: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    keys = sorted({str(key) for mapping in mappings for key in mapping.keys()})
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for key in keys:
        mean, std, count = sample_stats([mapping.get(key) for mapping in mappings])
        if count <= 0:
            continue
        means[key] = mean
        stds[key] = std
        counts[key] = count
    return means, stds, counts


def seed_value_from_dir(seed_dir: Path) -> int:
    match = SEED_DIR_PATTERN.fullmatch(Path(seed_dir).name)
    if match is None:
        raise ValueError(f"Invalid seed directory name: {seed_dir}")
    return int(match.group(1))


def discover_seed_dirs(
    experiment_dir: Path,
    *,
    require_log: bool = True,
    require_config: bool = True,
) -> List[Path]:
    parent = Path(experiment_dir)
    if not parent.is_dir():
        return []

    seed_dirs: List[Tuple[int, Path]] = []
    for child in parent.iterdir():
        if not child.is_dir() or SEED_DIR_PATTERN.fullmatch(child.name) is None:
            continue
        if require_log and not (child / "run_log.jsonl").is_file():
            continue
        if require_config and not (child / "config_resolved.yaml").is_file():
            continue
        seed_dirs.append((seed_value_from_dir(child), child))

    seed_dirs.sort(key=lambda item: item[0])
    return [path for _, path in seed_dirs]


def has_seed_runs(experiment_dir: Path) -> bool:
    return bool(discover_seed_dirs(experiment_dir))


def _drop_nested_key(mapping: MutableMapping[str, Any], path: Sequence[str]) -> None:
    current: Any = mapping
    for key in path[:-1]:
        if not isinstance(current, MutableMapping) or key not in current:
            return
        current = current[key]
    if isinstance(current, MutableMapping):
        current.pop(path[-1], None)


def normalized_config_for_seed_comparison(config: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    for path in _CONFIG_IGNORED_PATHS:
        _drop_nested_key(normalized, path)
    return normalized


def validate_seed_group(experiment_dir: Path, seed_runs: Sequence[Mapping[str, Any]]) -> List[int]:
    if not seed_runs:
        raise ValueError(f"No completed seed runs found under {experiment_dir}")

    seeds: List[int] = []
    reference_config: Optional[Dict[str, Any]] = None
    reference_seed_dir: Optional[Path] = None

    for seed_run in seed_runs:
        seed_dir = Path(seed_run["run_dir"])
        folder_seed = seed_value_from_dir(seed_dir)
        config = seed_run.get("config")
        if not isinstance(config, Mapping):
            cfg_path = seed_dir / "config_resolved.yaml"
            with cfg_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}

        configured_seed = config.get("train", {}).get("seed")
        try:
            configured_seed_int = int(configured_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid train.seed in {seed_dir}") from exc
        if configured_seed_int != folder_seed:
            raise ValueError(
                f"Seed directory/config mismatch in {seed_dir}: "
                f"folder={folder_seed}, train.seed={configured_seed_int}"
            )
        if folder_seed in seeds:
            raise ValueError(f"Duplicate seed {folder_seed} under {experiment_dir}")
        seeds.append(folder_seed)

        comparable_config = normalized_config_for_seed_comparison(config)
        if reference_config is None:
            reference_config = comparable_config
            reference_seed_dir = seed_dir
        elif comparable_config != reference_config:
            raise ValueError(
                f"Seed configurations differ beyond allowed run fields: "
                f"{reference_seed_dir} vs {seed_dir}"
            )

    return seeds


def _level_ids_for_prefix(losses: Mapping[str, Any], prefix: str) -> List[int]:
    ids: List[int] = []
    for key in losses:
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        tail = key[len(prefix) :]
        if tail.isdigit():
            ids.append(int(tail))
    return sorted(set(ids))


def _finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _hiercos_weights_from_depth(weight_mode: str, depth: int) -> Dict[int, float]:
    """Rebuild Hier-COS level weights for runs logged before the weights were.

    `loss_weight_level_*` was added to the Hier-COS metrics after roughly half
    the campaign had already been trained, so the weights have to be recovered
    for the older runs. They can be, exactly: `node_prob_weights` in
    `models/hiercos/topology.py` is a function of hierarchy depth alone and not
    of the taxonomy - `exp(1/[L..1])` normalized to unit L2 norm and squared,
    which then sums to one - and `equal` is `1/L`. The branching modes need
    per-level class counts that the resolved config does not carry, so they
    return nothing and leave their runs unconverted.
    """
    if depth <= 0:
        return {}
    if weight_mode == "equal":
        return {level: 1.0 / float(depth) for level in range(depth)}
    if weight_mode in {"kl_leaf", "kl_coarse"}:
        values = np.exp(1.0 / np.arange(depth, 0, -1, dtype=np.float64))
        values = values / np.linalg.norm(values)
        values = np.square(values)
        if weight_mode == "kl_coarse":
            values = values[::-1]
        return {level: float(values[level]) for level in range(depth)}
    return {}


def _normalized_level_weights(
    losses: Mapping[str, Any],
    weight_mode: str,
    level_ids: Sequence[int],
) -> Dict[int, float]:
    """Level weights rescaled to sum to one, or {} when they cannot be resolved.

    Hier-COS logs `loss_weight_level_*` before normalization while its CE terms
    apply `w / sum(w)`, so the sum is taken here rather than assumed. Runs that
    never logged the weights fall back to rebuilding them from the depth.
    """
    weights: Dict[int, float] = {}
    for level in _level_ids_for_prefix(losses, "loss_weight_level_"):
        weight = _finite_float(losses.get(f"loss_weight_level_{level}"))
        if weight is None or weight <= 0.0:
            return {}
        weights[level] = weight
    if not weights:
        weights = _hiercos_weights_from_depth(weight_mode, len(level_ids))
    total = sum(weights.values())
    if not weights or total <= 0.0:
        return {}
    return {level: weight / total for level, weight in weights.items()}


def unweight_level_losses(
    epoch_events: Sequence[MutableMapping[str, Any]],
    config: Mapping[str, Any],
) -> bool:
    """Rewrite weighted per-level training losses into their unweighted form.

    Families disagree on what `loss_level_*` means. H-CAST, HT-CapsNet, LH-DNN
    and HRN log the raw level loss and keep the multiplier beside it in
    `loss_weight_level_*`; Hier-COS folds the weight into the CE term because
    its level decomposition is the native KL objective expanded rather than a
    sum of independently weighted parts, and direct subspace supervision
    follows Hier-COS. Overlaying those in one panel puts a fit signal against a
    contribution to the objective, and the mismatch is worst exactly where the
    panels matter: the lexicographic presets run `equal` weights against
    `kl_leaf` for the native baselines, so part of any gap would be the weight.

    Per-level panels therefore read the unweighted loss, converted once here at
    parse time, and the aggregates that weighted their per-level parts are
    re-summed to match (see `_unweight_aggregate_losses`); `total` keeps its
    weights because it *is* the objective. Every weighted spelling is preserved
    under `weighted_<key>`, so the contribution view stays one lookup away. The
    conversion is exact: `reg_level_*` and
    `subspace_soft_cross_entropy_level_*` are logged unweighted next to their
    weighted totals, the epoch value is a batch-weighted mean, and the weights
    of both weighted families are constant within a run.

    Returns whether any event was rewritten.
    """
    model_cfg = config.get("model", {}) if isinstance(config, Mapping) else {}
    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    # `alpha` scales the Hier-COS regularizer; 1.0 is the default in
    # `models/hiercos/losses.py`, and the shipped configs set 0.05-0.15.
    alpha = _finite_float(model_cfg.get("alpha"))
    if alpha is None:
        alpha = 1.0
    raw_weight_mode = model_cfg.get("weight_mode")
    weight_mode = raw_weight_mode if isinstance(raw_weight_mode, str) else ""

    converted = False
    for event in epoch_events:
        losses = event.get("train_losses")
        if not isinstance(losses, MutableMapping):
            continue
        level_ids = _level_ids_for_prefix(losses, "loss_level_")
        if not level_ids:
            continue
        weights = _normalized_level_weights(losses, weight_mode, level_ids)
        event_converted = False

        for level in level_ids:
            if f"weighted_loss_level_{level}" in losses:
                continue
            weighted_level_loss = _finite_float(losses.get(f"loss_level_{level}"))
            if weighted_level_loss is None:
                continue

            # Subspace supervision logs the unweighted term outright, so its
            # conversion is a lookup and needs no weight at all.
            raw_loss = _finite_float(
                losses.get(f"subspace_soft_cross_entropy_level_{level}")
            )
            if raw_loss is not None:
                losses[f"weighted_loss_level_{level}"] = weighted_level_loss
                losses[f"loss_level_{level}"] = raw_loss
                event_converted = True
                continue

            # Hier-COS: `loss_level_l` is `w_l * CE_l + alpha * reg_l`, with the
            # regularizer never level-weighted, so only the CE half is divided
            # out and the two summands are recombined.
            weighted_ce = _finite_float(losses.get(f"ce_level_{level}"))
            level_reg = _finite_float(losses.get(f"reg_level_{level}"))
            weight = weights.get(level)
            if weighted_ce is None or level_reg is None or not weight:
                continue
            raw_ce = weighted_ce / weight
            losses[f"weighted_ce_level_{level}"] = weighted_ce
            losses[f"ce_level_{level}"] = raw_ce
            losses[f"weighted_loss_level_{level}"] = weighted_level_loss
            losses[f"loss_level_{level}"] = raw_ce + alpha * level_reg
            event_converted = True

        if event_converted:
            converted = True
            _unweight_aggregate_losses(losses, level_ids)

    return converted


def _unweight_aggregate_losses(
    losses: MutableMapping[str, Any],
    level_ids: Sequence[int],
) -> None:
    """Re-sum the aggregates that weighted their per-level parts.

    Hier-COS `ce`, and the subspace `subspace_soft_cross_entropy` and
    `subspace_target_kl`, are logged as the weighted sum of terms the log also
    carries per level, so a weighted aggregate would sit next to unweighted
    level curves. They are rebuilt as the plain sum of those parts. `total`
    keeps its weights because it is the objective, and `reg` and
    `subspace_score_l2` never weighted theirs.
    """
    for aggregate_key, level_prefix in _UNWEIGHTABLE_AGGREGATES.items():
        if aggregate_key not in losses or f"weighted_{aggregate_key}" in losses:
            continue
        weighted_aggregate = _finite_float(losses.get(aggregate_key))
        if weighted_aggregate is None:
            continue
        parts = [_finite_float(losses.get(f"{level_prefix}{level}")) for level in level_ids]
        if not parts or any(part is None for part in parts):
            continue
        losses[f"weighted_{aggregate_key}"] = weighted_aggregate
        losses[aggregate_key] = float(sum(parts))


def unweighted_level_note(dataset_runs: Sequence[Mapping[str, Any]], dataset_name: str) -> None:
    """Say so when a family's per-level curves were converted at parse time.

    Only Hier-COS and direct subspace supervision log weighted level losses, so
    this line appears in some notebooks and not others; printing it keeps the
    reason visible instead of leaving a silent difference between families.
    """
    if not any(bool(run_data.get("level_losses_unweighted")) for run_data in dataset_runs):
        return
    print(
        f"[{dataset_name}] per-level curves are shown unweighted: this family folds the "
        "level weight into the logged term, so it is divided out here, and the aggregate "
        "terms built from those parts are re-summed to match. The weights are the next "
        "section, and `total` still carries them because it is the objective."
    )


def aggregate_epoch_events(seed_runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    event_maps: List[Dict[int, Mapping[str, Any]]] = []
    for seed_run in seed_runs:
        events = {
            int(event["epoch"]): event
            for event in seed_run.get("epoch_events", [])
            if isinstance(event, Mapping) and "epoch" in event
        }
        if events:
            event_maps.append(events)

    if not event_maps:
        return []

    available_epochs = sorted(set.union(*(set(events.keys()) for events in event_maps)))
    aggregated: List[Dict[str, Any]] = []
    for epoch in available_epochs:
        source_events = [events[epoch] for events in event_maps if epoch in events]
        event: Dict[str, Any] = {"event": "epoch", "epoch": epoch}
        for mapping_key in _EPOCH_MAPPING_KEYS:
            source_mappings = [
                source.get(mapping_key, {})
                for source in source_events
                if isinstance(source.get(mapping_key, {}), Mapping)
            ]
            means, stds, counts = aggregate_numeric_mappings(source_mappings)
            event[mapping_key] = means
            event[f"{mapping_key}_std"] = stds
            event[f"{mapping_key}_count"] = counts
        aggregated.append(event)
    return aggregated


def aggregate_test_results(seed_runs: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    modes = sorted(
        {
            str(mode)
            for seed_run in seed_runs
            for mode in seed_run.get("test_results", {}).keys()
        }
    )
    aggregated: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        sections = [
            seed_run.get("test_results", {}).get(mode)
            for seed_run in seed_runs
        ]
        sections = [section for section in sections if isinstance(section, Mapping)]
        if not sections:
            continue

        metric_mappings = [
            section.get("test_metrics", {})
            for section in sections
            if isinstance(section.get("test_metrics", {}), Mapping)
        ]
        metric_means, metric_stds, metric_counts = aggregate_numeric_mappings(metric_mappings)
        epoch_mean, epoch_std, epoch_count = sample_stats(
            [section.get("best_epoch") for section in sections]
        )
        metric_mean, metric_std, metric_count = sample_stats(
            [section.get("best_metric") for section in sections]
        )
        aggregated[mode] = {
            "best_checkpoint": "",
            "best_epoch": epoch_mean,
            "best_epoch_std": epoch_std,
            "best_epoch_count": epoch_count,
            "best_metric": metric_mean,
            "best_metric_std": metric_std,
            "best_metric_count": metric_count,
            "test_metrics": metric_means,
            "test_metrics_std": metric_stds,
            "test_metrics_count": metric_counts,
        }
    return aggregated


def aggregate_best_epoch_events(
    seed_runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    modes = sorted(
        {
            str(mode)
            for seed_run in seed_runs
            for mode in seed_run.get("best_epoch_events", {}).keys()
        }
    )
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for mode in modes:
        events = [
            seed_run.get("best_epoch_events", {}).get(mode)
            for seed_run in seed_runs
        ]
        events = [event for event in events if isinstance(event, Mapping)]
        if not events:
            out[mode] = None
            continue
        epoch_mean, epoch_std, epoch_count = sample_stats([event.get("epoch") for event in events])
        metric_maps = [
            event.get("val_metrics_norm", {})
            for event in events
            if isinstance(event.get("val_metrics_norm", {}), Mapping)
        ]
        means, stds, counts = aggregate_numeric_mappings(metric_maps)
        out[mode] = {
            "event": "epoch",
            "epoch": epoch_mean,
            "epoch_std": epoch_std,
            "epoch_count": epoch_count,
            "val_metrics_norm": means,
            "val_metrics_norm_std": stds,
            "val_metrics_norm_count": counts,
        }
    return out


def aggregate_parsed_seed_runs(
    experiment_dir: Path,
    seed_runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    parent = Path(experiment_dir)
    seeds = validate_seed_group(parent, seed_runs)
    aggregated = copy.deepcopy(dict(seed_runs[0]))
    aggregated["run_dir"] = parent
    aggregated["run_name"] = parent.name
    aggregated["seed_runs"] = list(seed_runs)
    aggregated["seeds"] = seeds
    aggregated["num_seeds"] = len(seeds)
    aggregated["epoch_events"] = aggregate_epoch_events(seed_runs)
    aggregated["events"] = aggregated["epoch_events"]
    aggregated["test_results"] = aggregate_test_results(seed_runs)
    aggregated["test_metrics"] = dict(
        aggregated["test_results"].get("topdown", {}).get("test_metrics", {})
    )
    if not aggregated["test_metrics"]:
        aggregated["test_metrics"] = dict(
            aggregated["test_results"].get("independent", {}).get("test_metrics", {})
        )
    aggregated["best_epoch_events"] = aggregate_best_epoch_events(seed_runs)
    aggregated["best_epoch_event"] = aggregated["best_epoch_events"].get("topdown")

    best_metric_means, best_metric_stds, best_metric_counts = aggregate_numeric_mappings(
        [
            seed_run.get("best_metrics", {})
            for seed_run in seed_runs
            if isinstance(seed_run.get("best_metrics", {}), Mapping)
        ]
    )
    aggregated["best_metrics"] = best_metric_means
    aggregated["best_metrics_std"] = best_metric_stds
    aggregated["best_metrics_count"] = best_metric_counts
    return aggregated


def metric_series_with_std(
    epoch_events: Sequence[Mapping[str, Any]],
    metric_key: str,
    *,
    source: str = "val_metrics_norm",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epochs = np.asarray([int(event["epoch"]) for event in epoch_events], dtype=np.int32)
    means = np.asarray(
        [float(event.get(source, {}).get(metric_key, np.nan)) for event in epoch_events],
        dtype=np.float64,
    )
    stds = np.asarray(
        [float(event.get(f"{source}_std", {}).get(metric_key, np.nan)) for event in epoch_events],
        dtype=np.float64,
    )
    counts = np.asarray(
        [int(event.get(f"{source}_count", {}).get(metric_key, 0)) for event in epoch_events],
        dtype=np.int32,
    )
    return epochs, means, stds, counts


def format_mean_std(mean: Any, std: Any, count: int, formatter) -> str:
    try:
        mean_value = float(mean)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(mean_value):
        return "n/a"
    mean_text = formatter(mean_value)
    try:
        std_value = float(std)
    except (TypeError, ValueError):
        std_value = float("nan")
    if int(count) > 1 and np.isfinite(std_value):
        return f"{mean_text} ± {formatter(std_value)}"
    return mean_text
