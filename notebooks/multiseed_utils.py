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
