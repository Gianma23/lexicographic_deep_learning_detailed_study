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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from train.runtime.selection import SelectionKey, selection_key

try:
    from notebooks.utils.multiseed_utils import (
        aggregate_parsed_seed_runs,
        discover_seed_dirs,
        has_seed_runs,
        metric_series_with_std,
    )
except ModuleNotFoundError:
    from multiseed_utils import (
        aggregate_parsed_seed_runs,
        discover_seed_dirs,
        has_seed_runs,
        metric_series_with_std,
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


plt.style.use("seaborn-v0_8-whitegrid")
np.set_printoptions(precision=4, suppress=True)


RunSpec = Union[str, Path, Mapping[str, Any]]
RunData = MutableMapping[str, Any]
BEST_SELECTION_MODES = ("topdown", "independent")

_ACC_LEVEL_PATTERN = re.compile(r"^acc_level_(\d+)$")
_ACC_LEVEL_TOPDOWN_PATTERN = re.compile(r"^acc_level_topdown_(\d+)$")
_ACC_LEVEL_INDEPENDENT_PATTERN = re.compile(r"^acc_level_independent_(\d+)$")
_LOSS_LEVEL_PATTERN = re.compile(r"^loss_level_(\d+)$")

_DATASET_IDS = {
    "cifar-100",
    "cub-200-2011",
    "fgvc-aircraft",
}

_DATASET_CANONICAL_ALIASES = {
    "cifar-100": "cifar-100",
    "cifar100": "cifar-100",
    "cub-200-2011": "cub-200-2011",
    "cub-200": "cub-200-2011",
    "cub200": "cub-200-2011",
    "cub2002011": "cub-200-2011",
    "fgvc-aircraft": "fgvc-aircraft",
    "fgvcaircraft": "fgvc-aircraft",
    "aircraft": "fgvc-aircraft",
}

_DATASET_DISPLAY = {
    "cifar-100": "CIFAR-100",
    "cub-200-2011": "CUB-200-2011",
    "fgvc-aircraft": "FGVC-Aircraft",
}

_MODEL_IDS = {
    "hcast",
    "hrn",
    "ht_capsnet",
    "lhdnn",
    "hiercos",
}

_MODEL_CANONICAL_ALIASES = {
    "hcast": "hcast",
    "hrn": "hrn",
    "ht-capsnet": "ht_capsnet",
    "capsnet": "ht_capsnet",
    "lhdnn": "lhdnn",
    "hiercos": "hiercos",
    "hier-cos": "hiercos",
}

_MODEL_DISPLAY = {
    "hcast": "H-CAST",
    "hrn": "HRN",
    "ht_capsnet": "HT-CapsNet",
    "lhdnn": "LH-DNN",
    "hiercos": "HierCoS",
}

_MODEL_RUN_NAME_TOKENS = {
    "hcast": ("hcast",),
    "hrn": ("hrn",),
    "ht_capsnet": ("capsnet", "ht_capsnet"),
    "lhdnn": ("lhdnn",),
    "hiercos": ("hiercos",),
}

_DEFAULT_MODEL_COLORS = {
    "hcast": "#1f77b4",
    "hrn": "#ff7f0e",
    "ht_capsnet": "#2ca02c",
    "lhdnn": "#d62728",
    "hiercos": "#cad627",
}

_DATASET_RUN_NAME_TOKENS = {
    "cifar-100": ("cifar100", "cifar-100"),
    "cub-200-2011": ("cub200", "cub-200-2011", "cub2002011"),
    "fgvc-aircraft": ("aircraft", "fgvc-aircraft", "fgvcaircraft"),
}


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


def _normalize_lookup_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def canonical_dataset_name(name: Optional[str]) -> str:
    if not isinstance(name, str):
        raise ValueError("dataset_name must be a string.")
    if name in _DATASET_IDS:
        return name
    alias = _DATASET_CANONICAL_ALIASES.get(_normalize_lookup_token(name))
    if alias is not None:
        return alias
    raise ValueError(
        f"Unsupported dataset_name '{name}'. "
        "Expected one of ['cifar-100', 'cub-200-2011', 'fgvc-aircraft']."
    )


def dataset_display_name(name: Optional[str]) -> str:
    canonical = canonical_dataset_name(name)
    return _DATASET_DISPLAY.get(canonical, canonical)


def canonical_model_name(name: Optional[str]) -> str:
    if not isinstance(name, str):
        raise ValueError("model_name must be a string.")
    if name in _MODEL_IDS:
        return name
    alias = _MODEL_CANONICAL_ALIASES.get(_normalize_lookup_token(name))
    if alias is not None:
        return alias
    raise ValueError(
        f"Unsupported model_name '{name}'. "
        "Expected one of ['hcast', 'hrn', 'ht_capsnet', 'lhdnn', 'hiercos']."
    )


def model_display_name(name: Optional[str]) -> str:
    canonical = canonical_model_name(name)
    return _MODEL_DISPLAY.get(canonical, canonical)


def _model_run_name_tokens(model_name: str) -> List[str]:
    canonical = canonical_model_name(model_name)
    explicit = _MODEL_RUN_NAME_TOKENS.get(canonical, ())
    if explicit:
        return [token for token in explicit if isinstance(token, str) and token]
    return [canonical]


def _dataset_run_name_tokens(dataset_name: str) -> List[str]:
    canonical = canonical_dataset_name(dataset_name)
    explicit = _DATASET_RUN_NAME_TOKENS.get(canonical, ())
    if explicit:
        return [token for token in explicit if isinstance(token, str) and token]
    return [canonical]


def matches_model_dataset_run_name(run_data: Mapping[str, Any]) -> bool:
    run_name = run_data.get("run_name", "")
    model_name = run_data.get("model_name", "")
    dataset_name = run_data.get("dataset_name", "")

    if not isinstance(run_name, str) or not isinstance(model_name, str) or not isinstance(dataset_name, str):
        return False
    if not run_name or not model_name or not dataset_name:
        return False

    model_tokens = _model_run_name_tokens(model_name)
    dataset_tokens = _dataset_run_name_tokens(dataset_name)

    for model_token in model_tokens:
        for dataset_token in dataset_tokens:
            if run_name == f"{model_token}_{dataset_token}":
                return True
    return False


def normalize_metrics(metrics: Any) -> Dict[str, float]:
    out = _as_float_dict(metrics)

    if "weighted_ap_topdown" not in out:
        if "weighted_ap" in out:
            out["weighted_ap_topdown"] = out["weighted_ap"]
        elif "weighted_ap_independent" in out:
            out["weighted_ap_topdown"] = out["weighted_ap_independent"]
    if "weighted_ap_independent" not in out and "weighted_ap_topdown" in out:
        out["weighted_ap_independent"] = out["weighted_ap_topdown"]

    if "fpa_topdown" not in out:
        if "fpa" in out:
            out["fpa_topdown"] = out["fpa"]
        elif "fpa_independent" in out:
            out["fpa_topdown"] = out["fpa_independent"]
    if "fpa_independent" not in out and "fpa_topdown" in out:
        out["fpa_independent"] = out["fpa_topdown"]

    if "tice_topdown" not in out:
        if "tice" in out:
            out["tice_topdown"] = out["tice"]
        elif "tice_independent" in out:
            out["tice_topdown"] = out["tice_independent"]
    if "tice_independent" not in out and "tice_topdown" in out:
        out["tice_independent"] = out["tice_topdown"]

    if "ahd_topdown" not in out:
        if "ahd" in out:
            out["ahd_topdown"] = out["ahd"]
        elif "ahd_independent" in out:
            out["ahd_topdown"] = out["ahd_independent"]
    if "ahd_independent" not in out and "ahd_topdown" in out:
        out["ahd_independent"] = out["ahd_topdown"]

    for key, value in list(out.items()):
        match = _ACC_LEVEL_PATTERN.match(key)
        if match:
            level_idx = int(match.group(1))
            out.setdefault(f"acc_level_independent_{level_idx}", value)
            out.setdefault(f"acc_level_topdown_{level_idx}", value)

    level_ids = set()
    for key in list(out.keys()):
        match = _ACC_LEVEL_TOPDOWN_PATTERN.match(key)
        if match:
            level_ids.add(int(match.group(1)))
            continue

        match = _ACC_LEVEL_INDEPENDENT_PATTERN.match(key)
        if match:
            level_ids.add(int(match.group(1)))

    for level_idx in level_ids:
        topdown_key = f"acc_level_topdown_{level_idx}"
        independent_key = f"acc_level_independent_{level_idx}"
        if topdown_key not in out and independent_key in out:
            out[topdown_key] = out[independent_key]
        if independent_key not in out and topdown_key in out:
            out[independent_key] = out[topdown_key]

    return out


def load_jsonl_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_best_metrics(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, MappingABC):
        return {}

    out: Dict[str, float] = {}
    for mode in BEST_SELECTION_MODES:
        value = _coerce_float(raw.get(mode))
        if value is not None:
            out[mode] = value
    return out


def _normalize_single_test_result(section: Mapping[str, Any]) -> Dict[str, Any]:
    metadata_keys = {
        "best_checkpoint",
        "best_epoch",
        "best_metric",
        "best_selection_key",
        "test_metrics",
    }
    test_metrics = normalize_metrics(section.get("test_metrics", {}))
    if not test_metrics:
        metric_like = {key: value for key, value in section.items() if key not in metadata_keys}
        test_metrics = normalize_metrics(metric_like)

    best_epoch = _coerce_int(section.get("best_epoch"))
    best_metric = _coerce_float(section.get("best_metric"))
    return {
        "best_checkpoint": str(section.get("best_checkpoint", "")),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_selection_key": section.get("best_selection_key"),
        "test_metrics": test_metrics,
    }


def _normalize_test_results(raw: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, MappingABC):
        return {}

    results: Dict[str, Dict[str, Any]] = {}
    for mode in BEST_SELECTION_MODES:
        section = raw.get(mode)
        if not isinstance(section, MappingABC):
            continue
        results[mode] = _normalize_single_test_result(section)
    if results:
        return results

    legacy_result = _normalize_single_test_result(raw)
    if (
        legacy_result["test_metrics"]
        or legacy_result["best_checkpoint"]
        or legacy_result["best_metric"] is not None
    ):
        results["topdown"] = legacy_result
    return results


def selection_key_for_best(
    metrics: Mapping[str, float],
    mode: str = "topdown",
) -> SelectionKey:
    """Use the training runtime's exact checkpoint ordering in analysis."""
    m = normalize_metrics(metrics)
    return selection_key(m, mode)


def metric_for_best(metrics: Mapping[str, float], mode: str = "topdown") -> float:
    """Return only the primary component for legacy notebook displays."""
    return float(selection_key_for_best(metrics, mode)[0])


def _find_epoch_event(epoch_events: Sequence[Mapping[str, Any]], epoch: Optional[int]) -> Optional[Mapping[str, Any]]:
    if epoch is None:
        return None
    for event in epoch_events:
        if _coerce_int(event.get("epoch")) == epoch:
            return event
    return None


def _event_score(event: Mapping[str, Any], mode: str) -> float:
    best_metrics = event.get("best_metrics_norm", {})
    if isinstance(best_metrics, MappingABC):
        value = _coerce_float(best_metrics.get(mode))
        if value is not None:
            return value

    rank_scores = event.get("rank_scores", {})
    if isinstance(rank_scores, MappingABC):
        value = _coerce_float(rank_scores.get(mode))
        if value is not None:
            return value

    return float(event.get("rank_score", np.nan))


def _best_epoch_events_by_mode(
    epoch_events: Sequence[MutableMapping[str, Any]],
    test_results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Optional[Mapping[str, Any]]]:
    out: Dict[str, Optional[Mapping[str, Any]]] = {}
    for mode in BEST_SELECTION_MODES:
        result = test_results.get(mode, {})
        best_event = _find_epoch_event(epoch_events, _coerce_int(result.get("best_epoch")))
        if best_event is not None:
            out[mode] = best_event
            continue

        final_best = _coerce_float(result.get("best_metric"))
        if final_best is None:
            for event in reversed(epoch_events):
                best_metrics = event.get("best_metrics_norm", {})
                if not isinstance(best_metrics, MappingABC):
                    continue
                final_best = _coerce_float(best_metrics.get(mode))
                if final_best is not None:
                    break

        if final_best is not None:
            for event in epoch_events:
                best_metrics = event.get("best_metrics_norm", {})
                if not isinstance(best_metrics, MappingABC):
                    continue
                event_best = _coerce_float(best_metrics.get(mode))
                if event_best is not None and np.isclose(event_best, final_best, rtol=1e-9, atol=1e-12):
                    best_event = event
                    break
            if best_event is not None:
                out[mode] = best_event
                continue

        finite_pairs = [
            (idx, event.get("rank_keys", {}).get(mode))
            for idx, event in enumerate(epoch_events)
            if isinstance(event.get("rank_keys", {}), MappingABC)
        ]
        finite_pairs = [
            (idx, tuple(float(component) for component in value))
            for idx, value in finite_pairs
            if isinstance(value, (list, tuple)) and len(value) == 3
        ]
        if finite_pairs:
            best_idx = max(finite_pairs, key=lambda item: item[1])[0]
            out[mode] = epoch_events[best_idx]
        else:
            out[mode] = None
    return out


def _final_best_metrics(
    epoch_events: Sequence[Mapping[str, Any]],
    test_results: Mapping[str, Mapping[str, Any]],
    best_epoch_events: Mapping[str, Optional[Mapping[str, Any]]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for mode in BEST_SELECTION_MODES:
        result = test_results.get(mode, {})
        value = _coerce_float(result.get("best_metric"))
        if value is None:
            for event in reversed(epoch_events):
                best_metrics = event.get("best_metrics_norm", {})
                if not isinstance(best_metrics, MappingABC):
                    continue
                value = _coerce_float(best_metrics.get(mode))
                if value is not None:
                    break
        if value is None:
            event = best_epoch_events.get(mode)
            if event is not None:
                value = _coerce_float(event.get("rank_scores", {}).get(mode))
        if value is not None:
            out[mode] = value
    return out


def _mode_from_metric_key(metric_key: str) -> str:
    key = str(metric_key)
    if "_independent" in key or key.startswith("acc_level_independent_"):
        return "independent"
    return "topdown"


def _iter_test_metric_maps(run_data: Mapping[str, Any]) -> Iterable[Mapping[str, float]]:
    test_results = run_data.get("test_results", {})
    if isinstance(test_results, MappingABC):
        for mode in BEST_SELECTION_MODES:
            section = test_results.get(mode)
            if isinstance(section, MappingABC) and isinstance(section.get("test_metrics"), MappingABC):
                yield section["test_metrics"]
    test_metrics = run_data.get("test_metrics", {})
    if isinstance(test_metrics, MappingABC):
        yield test_metrics


def _test_metrics_for_mode(run_data: Mapping[str, Any], mode: str) -> Mapping[str, float]:
    test_results = run_data.get("test_results", {})
    if isinstance(test_results, MappingABC):
        section = test_results.get(mode)
        if isinstance(section, MappingABC) and isinstance(section.get("test_metrics"), MappingABC):
            metrics = section["test_metrics"]
            if metrics:
                return metrics
        if test_results:
            # Structured results are checkpoint-specific. Never substitute a
            # different decoding mode when the requested one is absent.
            return {}

    test_metrics = run_data.get("test_metrics", {})
    if isinstance(test_metrics, MappingABC):
        return test_metrics
    return {}


def _test_metric_value(run_data: Mapping[str, Any], metric_key: str) -> float:
    metrics = _test_metrics_for_mode(run_data, _mode_from_metric_key(metric_key))
    return float(metrics.get(metric_key, np.nan))


def _test_metric_stats(run_data: Mapping[str, Any], metric_key: str) -> Tuple[float, float, int]:
    mode = _mode_from_metric_key(metric_key)
    section = run_data.get("test_results", {}).get(mode, {})
    if not isinstance(section, MappingABC):
        return float("nan"), float("nan"), 0
    mean = float(section.get("test_metrics", {}).get(metric_key, np.nan))
    std = float(section.get("test_metrics_std", {}).get(metric_key, np.nan))
    count = int(section.get("test_metrics_count", {}).get(metric_key, 0))
    return mean, std, count


def _parse_single_run(run_dir: Path) -> RunData:
    run_path = Path(run_dir)

    cfg: Dict[str, Any] = {}
    cfg_path = run_path / "config_resolved.yaml"
    if cfg_path.exists():
        cfg = load_yaml(cfg_path) or {}

    events: List[Dict[str, Any]] = []
    log_path = run_path / "run_log.jsonl"
    if log_path.exists():
        events = load_jsonl_events(log_path)

    epoch_events = [event for event in events if event.get("event") == "epoch"]
    for event in epoch_events:
        event["train_metrics_norm"] = normalize_metrics(event.get("train_metrics", {}))
        event["val_metrics_norm"] = normalize_metrics(event.get("val_metrics", {}))
        event["rank_keys"] = {
            mode: selection_key_for_best(event["val_metrics_norm"], mode=mode)
            for mode in BEST_SELECTION_MODES
        }
        event["rank_scores"] = {
            mode: float(event["rank_keys"][mode][0])
            for mode in BEST_SELECTION_MODES
        }
        event["rank_score"] = event["rank_scores"]["topdown"]
        event["best_metrics_norm"] = _normalize_best_metrics(event.get("best_metrics", {}))
        if not event["best_metrics_norm"]:
            legacy_best_metric = _coerce_float(event.get("best_metric"))
            if legacy_best_metric is not None:
                event["best_metrics_norm"] = {"topdown": legacy_best_metric}

    test_event: Dict[str, Any] = {}
    test_events = [event for event in events if event.get("event") == "test"]
    if test_events:
        test_event = test_events[-1]

    test_results = _normalize_test_results(test_event.get("test_results", {}))
    if not test_results:
        test_results = _normalize_test_results(test_event)
    test_yaml_path = run_path / "test_metrics.yaml"
    if test_yaml_path.exists():
        test_metrics_yaml = load_yaml(test_yaml_path) or {}
        yaml_test_results = _normalize_test_results(test_metrics_yaml)
        if yaml_test_results:
            test_results = {**test_results, **yaml_test_results}
    test_metrics = dict(test_results.get("topdown", {}).get("test_metrics", {}))

    best_epoch_events = _best_epoch_events_by_mode(epoch_events, test_results)
    best_epoch_event = best_epoch_events.get("topdown")
    best_metrics = _final_best_metrics(epoch_events, test_results, best_epoch_events)

    model_raw = cfg.get("model", {}).get("name")
    model_name = canonical_model_name(model_raw)

    dataset_raw = cfg.get("dataset", {}).get("name")

    return {
        "run_dir": run_path,
        "run_name": run_path.name,
        "config": cfg,
        "events": events,
        "epoch_events": epoch_events,
        "test_event": test_event,
        "test_metrics": test_metrics,
        "test_results": test_results,
        "best_metrics": best_metrics,
        "best_epoch_events": best_epoch_events,
        "best_epoch_event": best_epoch_event,
        "level_names": list(cfg.get("dataset", {}).get("levels", [])),
        "dataset_name": canonical_dataset_name(dataset_raw),
        "dataset_label": dataset_display_name(dataset_raw),
        "model_name": model_name,
        "model_label": model_display_name(model_name),
    }


def parse_run(run_dir: Path) -> RunData:
    experiment_dir = Path(run_dir)
    seed_dirs = discover_seed_dirs(experiment_dir)
    if not seed_dirs:
        raise ValueError(f"No completed seed directories found under {experiment_dir}")
    seed_runs = [_parse_single_run(seed_dir) for seed_dir in seed_dirs]
    return aggregate_parsed_seed_runs(experiment_dir, seed_runs)


def _resolve_run_dir(path_like: Union[str, Path], output_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path

    root = Path(output_root)
    direct_candidate = root / path
    if direct_candidate.exists():
        return direct_candidate

    if root.exists():
        matches = sorted(
            candidate
            for candidate in root.rglob(path.name)
            if candidate.is_dir() and candidate.name == path.name and has_seed_runs(candidate)
        )
        if matches:
            return matches[0]

    return direct_candidate


def build_model_dataset_run_specs(
    model_names: Sequence[str],
    dataset_names: Sequence[str],
    *,
    model_run_names: Optional[Mapping[str, str]] = None,
    dataset_run_names: Optional[Mapping[str, str]] = None,
    model_labels: Optional[Mapping[str, str]] = None,
    dataset_labels: Optional[Mapping[str, str]] = None,
    run_name_overrides: Optional[Mapping[Tuple[str, str], str]] = None,
    run_label_overrides: Optional[Mapping[Tuple[str, str], str]] = None,
    run_color_overrides: Optional[Mapping[Tuple[str, str], str]] = None,
) -> List[RunSpec]:
    specs: List[RunSpec] = []
    model_run_names = dict(model_run_names or {})
    dataset_run_names = dict(dataset_run_names or {})
    model_labels = dict(model_labels or {})
    dataset_labels = dict(dataset_labels or {})
    run_name_overrides = dict(run_name_overrides or {})
    run_label_overrides = dict(run_label_overrides or {})
    run_color_overrides = dict(run_color_overrides or {})

    for dataset_name_raw in dataset_names:
        dataset_key = str(dataset_name_raw)
        dataset_name = canonical_dataset_name(dataset_key)
        dataset_token = str(
            dataset_run_names.get(
                dataset_key,
                dataset_run_names.get(dataset_name, _dataset_run_name_tokens(dataset_name)[0]),
            )
        )
        dataset_label = dataset_labels.get(dataset_key, dataset_display_name(dataset_name))

        for model_name_raw in model_names:
            model_key = str(model_name_raw)
            model_name = canonical_model_name(model_key)
            model_token = str(
                model_run_names.get(
                    model_key,
                    model_run_names.get(model_name, _model_run_name_tokens(model_name)[0]),
                )
            )
            model_label = model_labels.get(model_key, model_labels.get(model_name, model_display_name(model_name)))
            run_dir = run_name_overrides.get(
                (model_key, dataset_key),
                run_name_overrides.get((model_name, dataset_name), f"{model_token}_{dataset_token}"),
            )
            run_label = run_label_overrides.get(
                (model_key, dataset_key),
                run_label_overrides.get((model_name, dataset_name), model_label),
            )
            run_color = run_color_overrides.get(
                (model_key, dataset_key),
                run_color_overrides.get((model_name, dataset_name), None),
            )

            spec = {
                "run_dir": run_dir,
                "label": run_label,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "dataset_label": dataset_label,
            }
            if run_color is not None:
                spec["color"] = run_color
            specs.append(spec)

    return specs


def _normalize_manual_run_spec(spec: RunSpec, output_root: Path, manual_order: int) -> Dict[str, Any]:
    if isinstance(spec, (str, Path)):
        normalized: Dict[str, Any] = {"run_dir": spec}
    elif isinstance(spec, MappingABC):
        normalized = dict(spec)
    else:
        raise TypeError(f"Unsupported manual run spec type: {type(spec)!r}")

    if "run_dir" not in normalized:
        raise KeyError("Manual run specs must define a `run_dir` entry.")

    normalized["run_dir"] = _resolve_run_dir(normalized["run_dir"], output_root)
    normalized["_manual_order"] = int(manual_order)
    return normalized


def _manual_specs_by_dir(
    output_root: Path,
    manual_runs: Optional[Sequence[RunSpec]],
    manual_run_dirs: Optional[Sequence[Union[str, Path]]],
) -> Dict[Path, Dict[str, Any]]:
    specs_by_dir: Dict[Path, Dict[str, Any]] = {}
    raw_specs: List[RunSpec] = list(manual_runs or [])
    raw_specs.extend({"run_dir": run_dir} for run_dir in (manual_run_dirs or []))

    for idx, raw_spec in enumerate(raw_specs):
        spec = _normalize_manual_run_spec(raw_spec, output_root, idx)
        key = Path(spec["run_dir"]).resolve()
        if key not in specs_by_dir:
            specs_by_dir[key] = spec

    return specs_by_dir


def _apply_manual_run_spec(run_data: RunData, spec: Mapping[str, Any]) -> None:
    if "dataset_name" in spec and spec.get("dataset_name") is not None:
        dataset_name = canonical_dataset_name(spec["dataset_name"])
        run_data["dataset_name"] = dataset_name
        dataset_label = spec.get("dataset_label")
        run_data["dataset_label"] = str(dataset_label) if dataset_label is not None else dataset_display_name(dataset_name)
    elif "dataset_label" in spec and spec.get("dataset_label") is not None:
        run_data["dataset_label"] = str(spec["dataset_label"])

    if "model_name" in spec and spec.get("model_name") is not None:
        model_name = canonical_model_name(spec["model_name"])
        run_data["model_name"] = model_name
        model_label = spec.get("model_label")
        run_data["model_label"] = str(model_label) if model_label is not None else model_display_name(model_name)

    label = spec.get("label", spec.get("model_label"))
    if label is not None:
        run_data["model_label"] = str(label)

    if spec.get("color") is not None:
        run_data["_manual_color"] = str(spec["color"])

    run_data["_manual_order"] = int(spec.get("_manual_order", 999_999))


def discover_run_dirs(
    output_root: Path,
    manual_run_dirs: Optional[Sequence[Union[str, Path]]] = None,
    auto_discover: bool = True,
) -> List[Path]:
    manual_run_dirs = manual_run_dirs or []
    out: List[Path] = []
    seen: set[Path] = set()

    def append_unique(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        out.append(resolved)

    for item in manual_run_dirs:
        path = _resolve_run_dir(item, output_root)
        if path.exists() and path.is_dir() and has_seed_runs(path):
            append_unique(path)
        else:
            print(f"Skipping missing/manual-invalid run: {path}")

    if auto_discover:
        root = Path(output_root)
        if root.exists():
            for path in sorted(
                {
                    log_path.parent.parent
                    for log_path in root.rglob("seed_*/run_log.jsonl")
                    if has_seed_runs(log_path.parent.parent)
                }
            ):
                append_unique(path)

    return out


def get_metric_series(
    epoch_events: Sequence[Mapping[str, Any]], metric_key: str, source: str = "val_metrics_norm"
) -> Tuple[np.ndarray, np.ndarray]:
    epochs = [int(event["epoch"]) for event in epoch_events]
    values = [float(event.get(source, {}).get(metric_key, np.nan)) for event in epoch_events]
    return np.array(epochs, dtype=np.int32), np.array(values, dtype=np.float64)


def get_train_loss_series(epoch_events: Sequence[Mapping[str, Any]], loss_key: str) -> Tuple[np.ndarray, np.ndarray]:
    epochs = [int(event["epoch"]) for event in epoch_events]
    values = [float(event.get("train_losses", {}).get(loss_key, np.nan)) for event in epoch_events]
    return np.array(epochs, dtype=np.int32), np.array(values, dtype=np.float64)


def get_level_label(level_idx: int, run_data: Mapping[str, Any]) -> str:
    names = run_data.get("level_names") or []
    if level_idx < len(names):
        return str(names[level_idx])
    return f"L{level_idx}"


def ordered(items: Iterable[str], preferred_order: Sequence[str]) -> List[str]:
    item_set = set(items)
    ordered_pref = [item for item in preferred_order if item in item_set]
    rest = sorted([item for item in item_set if item not in ordered_pref])
    return ordered_pref + rest


def _fmt_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def _fmt_delta_pp(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{100.0 * value:.2f} pp"


def _fmt_edges(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def _fmt_delta_edges(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.3f}"


def _fmt_score(value: Any) -> str:
    score = _coerce_float(value)
    if score is None:
        return "n/a"
    return f"{score:.4f}"


def _fmt_epoch_score(event: Optional[Mapping[str, Any]], score: Any) -> str:
    if event is None:
        return f"n/a/{_fmt_score(score)}"
    try:
        epoch = float(event.get("epoch"))
    except (TypeError, ValueError):
        epoch = float("nan")
    try:
        epoch_std = float(event.get("epoch_std"))
    except (TypeError, ValueError):
        epoch_std = float("nan")
    epoch_count = int(event.get("epoch_count", 0))
    if not np.isfinite(epoch):
        epoch_text = "n/a"
    elif epoch_count > 1 and np.isfinite(epoch_std):
        epoch_text = f"{epoch:.1f}±{epoch_std:.1f}"
    else:
        epoch_text = f"{epoch:.0f}"
    return f"{epoch_text}/{_fmt_score(score)}"


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


def _safe_delta(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a - b)


def _pct_or_na(value: float) -> str:
    return _fmt_pct(value)


def _edges_or_na(value: float) -> str:
    return _fmt_edges(value)


def _default_mode_specs(include_topdown: bool = True) -> List[Tuple[str, str, str, str]]:
    mode_specs = [("independent", "--", "independent", "x")]
    if include_topdown:
        mode_specs.append(("topdown", "-", "top-down", "o"))
    return mode_specs


def _filter_mode_specs(
    mode_specs: Sequence[Tuple[str, str, str, str]], include_topdown: bool = True
) -> List[Tuple[str, str, str, str]]:
    if include_topdown:
        return list(mode_specs)
    return [spec for spec in mode_specs if spec[0] != "topdown"]


def _include_topdown_metric(metric_key: str, include_topdown: bool = True) -> bool:
    if include_topdown:
        return True
    return "_topdown" not in metric_key and not metric_key.startswith("acc_level_topdown_")


@dataclass
class ModelComparisonConfig:
    output_root: Path = Path(os.environ.get("OUTPUTS_ROOT", "/scratch/g.saggini1/outputs"))
    auto_discover: bool = True
    manual_runs: List[RunSpec] = field(default_factory=list)
    manual_run_dirs: List[Union[str, Path]] = field(default_factory=list)

    include_models: List[str] = field(default_factory=list)
    include_datasets: List[str] = field(default_factory=list)
    include_run_name_substrings: List[str] = field(default_factory=list)
    exclude_run_name_substrings: List[str] = field(default_factory=lambda: ["design", "warmup"])
    require_model_dataset_run_name_format: bool = False
    include_topdown_metrics: bool = True

    preferred_dataset_order: List[str] = field(
        default_factory=lambda: ["cifar-100", "cub-200-2011", "fgvc-aircraft"]
    )
    preferred_model_order: List[str] = field(
        default_factory=lambda: ["hcast", "lhdnn", "hrn", "ht_capsnet", "hiercos"]
    )

    model_colors: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_MODEL_COLORS))


class ModelComparisonAnalysis:
    def __init__(
        self,
        config: ModelComparisonConfig,
        candidate_run_dirs: List[Path],
        parsed_runs: List[RunData],
        parse_errors: List[Tuple[str, str]],
        filtered_runs: List[RunData],
        runs_for_analysis: List[RunData],
    ) -> None:
        self.config = config
        self.candidate_run_dirs = candidate_run_dirs
        self.parsed_runs = parsed_runs
        self.parse_errors = parse_errors
        self.filtered_runs = filtered_runs
        self.runs_for_analysis = runs_for_analysis

        self.dataset_keys = ordered(
            [str(run["dataset_name"]) for run in runs_for_analysis],
            config.preferred_dataset_order,
        )
        self.model_keys = ordered(
            [str(run["model_name"]) for run in runs_for_analysis],
            config.preferred_model_order,
        )

        self.runs_by_dataset: Dict[str, List[RunData]] = {dataset: [] for dataset in self.dataset_keys}
        for run in runs_for_analysis:
            self.runs_by_dataset[str(run["dataset_name"])].append(run)

        for dataset in self.dataset_keys:
            self.runs_by_dataset[dataset] = sorted(
                self.runs_by_dataset[dataset],
                key=lambda run: (
                    config.preferred_model_order.index(str(run["model_name"]))
                    if str(run["model_name"]) in config.preferred_model_order
                    else 999,
                    str(run["model_name"]),
                    int(run.get("_manual_order", 999_999)),
                    str(run["run_name"]),
                ),
            )

        dataset_rank = {name: idx for idx, name in enumerate(self.dataset_keys)}
        self.runs_by_model: Dict[str, List[RunData]] = {model: [] for model in self.model_keys}
        for run in runs_for_analysis:
            self.runs_by_model.setdefault(str(run["model_name"]), []).append(run)

        for model_name, model_runs in list(self.runs_by_model.items()):
            self.runs_by_model[model_name] = sorted(
                model_runs,
                key=lambda run: (
                    dataset_rank.get(str(run["dataset_name"]), 999),
                    str(run["dataset_name"]),
                    int(run.get("_manual_order", 999_999)),
                    str(run["run_name"]),
                ),
            )

    @classmethod
    def from_config(cls, config: ModelComparisonConfig) -> "ModelComparisonAnalysis":
        manual_specs = _manual_specs_by_dir(config.output_root, config.manual_runs, config.manual_run_dirs)
        manual_run_dirs = [Path(spec["run_dir"]) for spec in manual_specs.values()]
        candidate_run_dirs = discover_run_dirs(config.output_root, manual_run_dirs, config.auto_discover)

        parsed_runs: List[RunData] = []
        parse_errors: List[Tuple[str, str]] = []
        for run_dir in candidate_run_dirs:
            try:
                run_data = parse_run(run_dir)
                manual_spec = manual_specs.get(Path(run_data["run_dir"]).resolve())
                if manual_spec is not None:
                    _apply_manual_run_spec(run_data, manual_spec)
                parsed_runs.append(run_data)
            except Exception as exc:
                parse_errors.append((str(run_dir), str(exc)))

        include_models_norm = {canonical_model_name(name) for name in config.include_models}
        include_datasets_norm = {canonical_dataset_name(name) for name in config.include_datasets}
        include_tokens = [token.lower() for token in config.include_run_name_substrings]
        exclude_tokens = [token.lower() for token in config.exclude_run_name_substrings]

        filtered_runs: List[RunData] = []
        for run in parsed_runs:
            run_name_l = str(run["run_name"]).lower()

            if include_models_norm and str(run["model_name"]) not in include_models_norm:
                continue
            if include_datasets_norm and str(run["dataset_name"]) not in include_datasets_norm:
                continue
            if config.require_model_dataset_run_name_format and not matches_model_dataset_run_name(run):
                continue
            if include_tokens and not any(token in run_name_l for token in include_tokens):
                continue
            if exclude_tokens and any(token in run_name_l for token in exclude_tokens):
                continue

            filtered_runs.append(run)

        runs_for_analysis = list(filtered_runs)

        if not runs_for_analysis:
            raise ValueError("No runs after filtering. Update configuration and rerun.")

        model_colors = dict(config.model_colors)
        unknown_models = sorted(
            {
                str(run["model_name"])
                for run in runs_for_analysis
                if str(run["model_name"]) not in model_colors
            }
        )
        if unknown_models:
            cmap = plt.get_cmap("tab10")
            for idx, model_name in enumerate(unknown_models):
                model_colors[model_name] = cmap(idx % 10)

        for run in runs_for_analysis:
            model_name = str(run["model_name"])
            run["color"] = run.get("_manual_color", model_colors[model_name])

        config.model_colors = model_colors
        return cls(
            config=config,
            candidate_run_dirs=candidate_run_dirs,
            parsed_runs=parsed_runs,
            parse_errors=parse_errors,
            filtered_runs=filtered_runs,
            runs_for_analysis=runs_for_analysis,
        )

    def print_run_summary(self) -> None:
        print(f"Candidate run directories: {len(self.candidate_run_dirs)}")
        print(f"Parsed runs: {len(self.parsed_runs)}")
        if self.config.require_model_dataset_run_name_format:
            print("Run-name filter: strict `model_dataset` format only.")

        if self.parse_errors:
            print("Parse errors:")
            for run_dir, err in self.parse_errors:
                print(f"  - {run_dir}: {err}")

        print(f"Filtered runs: {len(self.filtered_runs)}")

        print("\nSelected runs")
        print("-------------")
        for dataset_name in self.dataset_keys:
            print(dataset_display_name(dataset_name))
            for run in self.runs_by_dataset[dataset_name]:
                best_events = run.get("best_epoch_events", {})
                best_metrics = run.get("best_metrics", {})
                topdown_event = best_events.get("topdown") if isinstance(best_events, MappingABC) else None
                independent_event = best_events.get("independent") if isinstance(best_events, MappingABC) else None
                topdown_score = (
                    best_metrics.get("topdown")
                    if isinstance(best_metrics, MappingABC) and "topdown" in best_metrics
                    else _event_score(topdown_event, "topdown") if topdown_event is not None else None
                )
                independent_score = (
                    best_metrics.get("independent")
                    if isinstance(best_metrics, MappingABC) and "independent" in best_metrics
                    else _event_score(independent_event, "independent") if independent_event is not None else None
                )
                test_fpa_topdown = float(_test_metrics_for_mode(run, "topdown").get("fpa_topdown", np.nan))
                test_fpa_independent = float(_test_metrics_for_mode(run, "independent").get("fpa_independent", np.nan))
                summary_bits = [
                    f"  - {run['model_label']:<12} | run={run['run_name']:<35}",
                    f"seeds={run.get('seeds', [])}",
                ]
                if self.config.include_topdown_metrics:
                    summary_bits.extend(
                        [
                            f"best_td={_fmt_epoch_score(topdown_event, topdown_score):<12}",
                            f"best_ind={_fmt_epoch_score(independent_event, independent_score):<12}",
                            f"test_FPA_td={_fmt_pct(test_fpa_topdown)}",
                            f"test_FPA_ind={_fmt_pct(test_fpa_independent)}",
                        ]
                    )
                else:
                    summary_bits.extend(
                        [
                            f"best_ind={_fmt_epoch_score(independent_event, independent_score):<12}",
                            f"test_FPA_ind={_fmt_pct(test_fpa_independent)}",
                        ]
                    )
                print(" | ".join(summary_bits))

    def plot_validation_curves(
        self,
        metric_families: Optional[Sequence[Tuple[str, str, bool]]] = None,
        mode_specs: Optional[Sequence[Tuple[str, str, str, str]]] = None,
        show_best_errorbars: bool = False,
        include_topdown: Optional[bool] = None,
    ) -> None:
        include_topdown = self.config.include_topdown_metrics if include_topdown is None else bool(include_topdown)
        metric_families = metric_families or [
            ("fpa", "Validation FPA (%)", True),
            ("weighted_ap", "Validation wAP (%)", True),
            ("tice", "Validation TICE (%)", True),
            ("ahd", "Validation AHD (edges)", False),
        ]
        mode_specs = (
            _default_mode_specs(include_topdown)
            if mode_specs is None
            else _filter_mode_specs(mode_specs, include_topdown)
        )

        for dataset_name in self.dataset_keys:
            dataset_runs = self.runs_by_dataset[dataset_name]
            if not dataset_runs:
                continue

            fig, axes = plt.subplots(len(metric_families), 1, figsize=(10, 5 * len(metric_families)), sharex=True)
            if len(metric_families) == 1:
                axes = [axes]

            for ax, (metric_prefix, metric_title, is_percent) in zip(axes, metric_families):
                for run_data in dataset_runs:
                    for mode_key, line_style, mode_label, marker in mode_specs:
                        metric_key = f"{metric_prefix}_{mode_key}"
                        epochs, values, stds, counts = metric_series_with_std(
                            run_data["epoch_events"], metric_key
                        )
                        if np.all(~np.isfinite(values)):
                            continue

                        plot_values = values * 100.0 if is_percent else values
                        ax.plot(
                            epochs,
                            plot_values,
                            label=f"{run_data['model_label']} ({mode_label})",
                            color=run_data["color"],
                            linestyle=line_style,
                            linewidth=2.0,
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
                                    s=50,
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

                ax.set_title(metric_title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Score (%)" if is_percent else "Distance (edges)")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)

            plt.suptitle(
                f"{dataset_display_name(dataset_name)}: Validation Metrics ({'Top-Down + Independent' if include_topdown else 'Independent'})",
                y=1.03,
                fontsize=13,
            )
            plt.tight_layout()
            plt.show()

    def plot_training_losses_per_dataset(self, aggregate_loss_keys: Optional[Sequence[str]] = None) -> None:
        aggregate_loss_keys = list(aggregate_loss_keys or ["total", "ce", "reg", "kl", "level_ce", "gk_loss"])

        for dataset_name in self.dataset_keys:
            dataset_runs = self.runs_by_dataset[dataset_name]
            if not dataset_runs:
                continue

            all_loss_keys = sorted(
                {
                    key
                    for run_data in dataset_runs
                    for event in run_data["epoch_events"]
                    for key in event.get("train_losses", {}).keys()
                }
            )

            aggregate_present = [key for key in aggregate_loss_keys if key in all_loss_keys]
            level_loss_ids = sorted(
                {
                    int(match.group(1))
                    for key in all_loss_keys
                    for match in [_LOSS_LEVEL_PATTERN.match(key)]
                    if match is not None
                }
            )

            metric_specs = [(key, key) for key in aggregate_present]

            base_for_labels = dataset_runs[0]
            for level_idx in level_loss_ids:
                level_name = get_level_label(level_idx, base_for_labels)
                metric_specs.append((f"loss_level_{level_idx}", f"loss_{level_name} (L{level_idx})"))

            if not metric_specs:
                print(f"No train loss keys found for dataset {dataset_display_name(dataset_name)}")
                continue

            n_metrics = len(metric_specs)
            ncols = 2
            nrows = int(np.ceil(n_metrics / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.6 * nrows), sharex=True)
            axes = np.array(axes).reshape(-1)

            for ax, (metric_key, metric_title) in zip(axes, metric_specs):
                for run_data in dataset_runs:
                    epochs, values = get_train_loss_series(run_data["epoch_events"], metric_key)
                    if np.all(~np.isfinite(values)):
                        continue
                    ax.plot(
                        epochs,
                        values,
                        label=run_data["model_label"],
                        color=run_data["color"],
                        linewidth=2.0,
                    )

                ax.set_title(f"Train {metric_title}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.grid(True, alpha=0.3)
                ax.legend()

            for ax in axes[n_metrics:]:
                ax.axis("off")

            plt.suptitle(f"{dataset_display_name(dataset_name)}: Training Losses", y=1.02, fontsize=13)
            plt.tight_layout()
            plt.show()

    def plot_training_losses_per_model_across_datasets(
        self,
        preferred_aggregate: Optional[Sequence[str]] = None,
    ) -> None:
        preferred_aggregate = list(
            preferred_aggregate or ["total", "ce", "reg", "kl", "level_ce", "gk_loss", "margin", "consistency"]
        )

        for model_name in self.model_keys:
            model_runs = list(self.runs_by_model.get(model_name, []))
            if not model_runs:
                continue

            if len(model_runs) < 2:
                print(
                    f"Skipping {model_display_name(model_name)}: "
                    "need at least 2 datasets for cross-dataset loss comparison."
                )
                continue

            loss_keys_per_run = []
            for run_data in model_runs:
                keys = {
                    key
                    for event in run_data["epoch_events"]
                    for key in event.get("train_losses", {}).keys()
                }
                if keys:
                    loss_keys_per_run.append(keys)

            if not loss_keys_per_run:
                print(f"No training loss keys found for model {model_display_name(model_name)}")
                continue

            shared_loss_keys = set.intersection(*loss_keys_per_run)
            if not shared_loss_keys:
                print(f"No shared train loss keys across datasets for {model_display_name(model_name)}")
                continue

            shared_loss_keys = sorted(shared_loss_keys)
            aggregate_loss_keys = [key for key in preferred_aggregate if key in shared_loss_keys]

            level_loss_ids = sorted(
                {
                    int(match.group(1))
                    for key in shared_loss_keys
                    for match in [_LOSS_LEVEL_PATTERN.match(key)]
                    if match is not None
                }
            )

            metric_specs = [(key, key) for key in aggregate_loss_keys]

            base_for_labels = model_runs[0]
            for level_idx in level_loss_ids:
                level_name = get_level_label(level_idx, base_for_labels)
                metric_specs.append((f"loss_level_{level_idx}", f"loss_{level_name} (L{level_idx})"))

            used_keys = {key for key, _ in metric_specs}
            other_shared_keys = [key for key in shared_loss_keys if key not in used_keys]
            metric_specs.extend((key, key) for key in other_shared_keys)

            if not metric_specs:
                print(f"No comparable train loss keys for model {model_display_name(model_name)}")
                continue

            n_metrics = len(metric_specs)
            ncols = 2
            nrows = int(np.ceil(n_metrics / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.6 * nrows), sharex=True)
            axes = np.array(axes).reshape(-1)

            cmap = plt.get_cmap("tab10")
            dataset_colors = {
                run_data["dataset_name"]: cmap(idx % 10)
                for idx, run_data in enumerate(model_runs)
            }

            for ax, (metric_key, metric_title) in zip(axes, metric_specs):
                for run_data in model_runs:
                    epochs, values = get_train_loss_series(run_data["epoch_events"], metric_key)
                    if np.all(~np.isfinite(values)):
                        continue

                    ax.plot(
                        epochs,
                        values,
                        label=run_data["dataset_label"],
                        color=dataset_colors[run_data["dataset_name"]],
                        linewidth=2.0,
                    )

                ax.set_title(f"Train {metric_title}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.grid(True, alpha=0.3)
                ax.legend()

            for ax in axes[n_metrics:]:
                ax.axis("off")

            plt.suptitle(f"{model_display_name(model_name)}: Training Losses Across Datasets", y=1.02, fontsize=13)
            plt.tight_layout()
            plt.show()

    def plot_per_run_per_level_training_losses(self) -> None:
        for dataset_name in self.dataset_keys:
            dataset_runs = self.runs_by_dataset.get(dataset_name, [])
            for run_data in dataset_runs:
                all_loss_keys = sorted(
                    {
                        key
                        for event in run_data["epoch_events"]
                        for key in event.get("train_losses", {}).keys()
                    }
                )

                level_loss_ids = sorted(
                    {
                        int(match.group(1))
                        for key in all_loss_keys
                        for match in [_LOSS_LEVEL_PATTERN.match(key)]
                        if match is not None
                    }
                )

                if not level_loss_ids:
                    print(
                        f"No per-level train losses for {run_data['model_label']} "
                        f"on {dataset_display_name(dataset_name)}"
                    )
                    continue

                fig, ax = plt.subplots(figsize=(10, 4.8))
                cmap = plt.get_cmap("tab20")
                plotted = 0

                for idx, level_idx in enumerate(level_loss_ids):
                    metric_key = f"loss_level_{level_idx}"
                    epochs, values = get_train_loss_series(run_data["epoch_events"], metric_key)
                    if np.all(~np.isfinite(values)):
                        continue

                    level_name = get_level_label(level_idx, run_data)
                    ax.plot(
                        epochs,
                        values,
                        label=f"{level_name} (L{level_idx})",
                        color=cmap(idx % 20),
                        linewidth=2.0,
                    )
                    plotted += 1

                if plotted == 0:
                    plt.close(fig)
                    print(
                        f"No finite per-level train losses for {run_data['model_label']} "
                        f"on {dataset_display_name(dataset_name)}"
                    )
                    continue

                ax.set_title(
                    f"{run_data['model_label']} on {dataset_display_name(dataset_name)}: "
                    "Per-Level Training Losses"
                )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.grid(True, alpha=0.3)
                ax.legend(ncol=2, fontsize=9)
                plt.tight_layout()
                plt.show()

    def plot_per_level_validation_accuracy(
        self,
        mode_specs: Optional[Sequence[Tuple[str, str, str, str]]] = None,
        show_best_errorbars: bool = False,
        include_topdown: Optional[bool] = None,
    ) -> None:
        include_topdown = self.config.include_topdown_metrics if include_topdown is None else bool(include_topdown)
        mode_specs = (
            _default_mode_specs(include_topdown)
            if mode_specs is None
            else _filter_mode_specs(mode_specs, include_topdown)
        )

        for dataset_name in self.dataset_keys:
            dataset_runs = self.runs_by_dataset[dataset_name]
            if not dataset_runs:
                continue

            all_level_ids = sorted(
                {
                    int(key.rsplit("_", 1)[-1])
                    for run_data in dataset_runs
                    for event in run_data["epoch_events"]
                    for key in event.get("val_metrics_norm", {}).keys()
                    if (
                        _include_topdown_metric(key, include_topdown)
                        and
                        (key.startswith("acc_level_topdown_") or key.startswith("acc_level_independent_"))
                        and key.rsplit("_", 1)[-1].isdigit()
                    )
                }
            )

            if not all_level_ids:
                print(f"No per-level validation accuracy metrics for dataset {dataset_display_name(dataset_name)}")
                continue

            fig, axes = plt.subplots(len(all_level_ids), 1, figsize=(10, 5 * len(all_level_ids)), sharex=True)
            if len(all_level_ids) == 1:
                axes = [axes]

            for ax, level_idx in zip(axes, all_level_ids):
                level_label = get_level_label(level_idx, dataset_runs[0])

                for run_data in dataset_runs:
                    for mode_key, line_style, mode_label, marker in mode_specs:
                        metric_key = f"acc_level_{mode_key}_{level_idx}"
                        epochs, values, stds, counts = metric_series_with_std(
                            run_data["epoch_events"], metric_key
                        )
                        if np.all(~np.isfinite(values)):
                            continue

                        ax.plot(
                            epochs,
                            values * 100.0,
                            label=f"{run_data['model_label']} ({mode_label})",
                            color=run_data["color"],
                            linestyle=line_style,
                            linewidth=2.0,
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
                                    s=42,
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

                ax.set_title(f"Validation Accuracy - {level_label} (L{level_idx})")
                ax.set_ylabel("Accuracy (%)")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)

            axes[-1].set_xlabel("Epoch")
            plt.suptitle(f"{dataset_display_name(dataset_name)}: Per-Level Validation Accuracy", y=1.01, fontsize=13)
            plt.tight_layout()
            plt.show()

    def show_final_test_tables(self, include_topdown: Optional[bool] = None) -> None:
        include_topdown = self.config.include_topdown_metrics if include_topdown is None else bool(include_topdown)
        metric_rows_base = [
            ("fpa_independent", "FPA independent"),
            ("fpa_topdown", "FPA top-down"),
            ("weighted_ap_independent", "wAP independent"),
            ("weighted_ap_topdown", "wAP top-down"),
            ("tice_independent", "TICE independent"),
            ("tice_topdown", "TICE top-down"),
            ("ahd_independent", "AHD independent"),
            ("ahd_topdown", "AHD top-down"),
        ]

        for dataset_name in self.dataset_keys:
            dataset_runs = self.runs_by_dataset[dataset_name]
            if not dataset_runs:
                continue

            base = dataset_runs[0]

            level_ids_from_test = {
                int(key.rsplit("_", 1)[-1])
                for run_data in dataset_runs
                for metric_map in _iter_test_metric_maps(run_data)
                for key in metric_map.keys()
                if (
                    _include_topdown_metric(key, include_topdown)
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
                        for key in event.get("val_metrics_norm", {}).keys()
                        if (
                            _include_topdown_metric(key, include_topdown)
                            and
                            (key.startswith("acc_level_topdown_") or key.startswith("acc_level_independent_"))
                            and key.rsplit("_", 1)[-1].isdigit()
                        )
                    }
                )

            metric_rows = [
                row for row in metric_rows_base
                if _include_topdown_metric(row[0], include_topdown)
            ]
            for level_idx in all_level_ids:
                level_label = get_level_label(level_idx, base)
                metric_rows.append((f"acc_level_independent_{level_idx}", f"Acc independent {level_label} (L{level_idx})"))
                if include_topdown:
                    metric_rows.append((f"acc_level_topdown_{level_idx}", f"Acc top-down {level_label} (L{level_idx})"))

            values_by_metric: Dict[str, List[float]] = {}
            best_by_metric: Dict[str, set[int]] = {}
            second_best_by_metric: Dict[str, set[int]] = {}
            for metric_key, _ in metric_rows:
                values = [_test_metric_value(run_data, metric_key) for run_data in dataset_runs]
                values_by_metric[metric_key] = values
                best_indices, second_best_indices = _best_and_second_best_indices(metric_key, values)
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

                if include_topdown:
                    best_epoch_cells.append(f"{epoch_text(td_section)}/{epoch_text(ind_section)}")
                else:
                    best_epoch_cells.append(epoch_text(ind_section or td_section))

            header_labels = ["Metric"] + [
                f"{run_data['model_label']} (n={run_data.get('num_seeds', 1)})"
                for run_data in dataset_runs
            ]
            table_lines = [
                f"### Dataset: `{dataset_display_name(dataset_name)}`",
                f"Baseline run: **{base['model_label']}**",
                "",
                "| " + " | ".join(header_labels) + " |",
                "|---|" + "|".join(["---:"] * (len(header_labels) - 1)) + "|",
                f"| Best epoch ({'TD/Ind' if include_topdown else 'Ind'}) | "
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

    def print_quick_summary(self) -> None:
        for dataset_name in self.dataset_keys:
            dataset_runs = self.runs_by_dataset[dataset_name]
            if not dataset_runs:
                continue

            print("\n" + dataset_display_name(dataset_name))
            print("~" * len(dataset_display_name(dataset_name)))

            for run_data in dataset_runs:
                best_events = run_data.get("best_epoch_events", {})
                best_topdown = (
                    best_events.get("topdown")
                    if isinstance(best_events, MappingABC)
                    else run_data.get("best_epoch_event")
                )
                best_independent = (
                    best_events.get("independent")
                    if isinstance(best_events, MappingABC)
                    else None
                )
                if best_topdown is None and best_independent is None:
                    print(f"{run_data['model_label']}: no epoch events")
                    continue

                topdown_metrics = best_topdown.get("val_metrics_norm", {}) if best_topdown is not None else {}
                independent_metrics = (
                    best_independent.get("val_metrics_norm", {}) if best_independent is not None else topdown_metrics
                )
                best_fpa_ind = float(independent_metrics.get("fpa_independent", np.nan))
                best_fpa_td = float(topdown_metrics.get("fpa_topdown", np.nan))
                best_wap_ind = float(independent_metrics.get("weighted_ap_independent", np.nan))
                best_wap_td = float(topdown_metrics.get("weighted_ap_topdown", np.nan))
                best_tice_ind = float(independent_metrics.get("tice_independent", np.nan))
                best_tice_td = float(topdown_metrics.get("tice_topdown", np.nan))
                best_ahd_ind = float(independent_metrics.get("ahd_independent", np.nan))
                best_ahd_td = float(topdown_metrics.get("ahd_topdown", np.nan))

                summary = (
                    f"{run_data['model_label']}: "
                    f"best_ind={_fmt_epoch_score(best_independent, _event_score(best_independent, 'independent') if best_independent else None)}, "
                    f"val FPA_ind={_pct_or_na(best_fpa_ind)}, "
                    f"val wAP_ind={_pct_or_na(best_wap_ind)}, "
                    f"val TICE_ind={_pct_or_na(best_tice_ind)}, "
                    f"val AHD_ind={_edges_or_na(best_ahd_ind)}"
                )
                if self.config.include_topdown_metrics:
                    summary = (
                        f"{run_data['model_label']}: "
                        f"best_td={_fmt_epoch_score(best_topdown, _event_score(best_topdown, 'topdown') if best_topdown else None)}, "
                        f"best_ind={_fmt_epoch_score(best_independent, _event_score(best_independent, 'independent') if best_independent else None)}, "
                        f"val FPA_ind={_pct_or_na(best_fpa_ind)}, val FPA_td={_pct_or_na(best_fpa_td)}, "
                        f"val wAP_ind={_pct_or_na(best_wap_ind)}, val wAP_td={_pct_or_na(best_wap_td)}, "
                        f"val TICE_ind={_pct_or_na(best_tice_ind)}, val TICE_td={_pct_or_na(best_tice_td)}, "
                        f"val AHD_ind={_edges_or_na(best_ahd_ind)}, val AHD_td={_edges_or_na(best_ahd_td)}"
                    )
                print(summary)

            if len(dataset_runs) < 2:
                continue

            base = dataset_runs[0]
            for comp in dataset_runs[1:]:
                fpa_ind_delta = _safe_delta(
                    float(_test_metrics_for_mode(comp, "independent").get("fpa_independent", np.nan)),
                    float(_test_metrics_for_mode(base, "independent").get("fpa_independent", np.nan)),
                )
                fpa_td_delta = _safe_delta(
                    float(_test_metrics_for_mode(comp, "topdown").get("fpa_topdown", np.nan)),
                    float(_test_metrics_for_mode(base, "topdown").get("fpa_topdown", np.nan)),
                )
                wap_ind_delta = _safe_delta(
                    float(_test_metrics_for_mode(comp, "independent").get("weighted_ap_independent", np.nan)),
                    float(_test_metrics_for_mode(base, "independent").get("weighted_ap_independent", np.nan)),
                )
                wap_td_delta = _safe_delta(
                    float(_test_metrics_for_mode(comp, "topdown").get("weighted_ap_topdown", np.nan)),
                    float(_test_metrics_for_mode(base, "topdown").get("weighted_ap_topdown", np.nan)),
                )
                tice_ind_delta = _safe_delta(
                    float(_test_metrics_for_mode(comp, "independent").get("tice_independent", np.nan)),
                    float(_test_metrics_for_mode(base, "independent").get("tice_independent", np.nan)),
                )
                tice_td_delta = _safe_delta(
                    float(_test_metrics_for_mode(comp, "topdown").get("tice_topdown", np.nan)),
                    float(_test_metrics_for_mode(base, "topdown").get("tice_topdown", np.nan)),
                )
                ahd_ind_delta = _safe_delta(
                    float(_test_metrics_for_mode(comp, "independent").get("ahd_independent", np.nan)),
                    float(_test_metrics_for_mode(base, "independent").get("ahd_independent", np.nan)),
                )
                ahd_td_delta = _safe_delta(
                    float(_test_metrics_for_mode(comp, "topdown").get("ahd_topdown", np.nan)),
                    float(_test_metrics_for_mode(base, "topdown").get("ahd_topdown", np.nan)),
                )

                delta_summary = (
                    f"Test delta ({comp['model_label']} - {base['model_label']}): "
                    f"FPA_ind={_fmt_delta_pp(fpa_ind_delta)}, "
                    f"wAP_ind={_fmt_delta_pp(wap_ind_delta)}, "
                    f"TICE_ind={_fmt_delta_pp(tice_ind_delta)}, "
                    f"AHD_ind={_fmt_delta_edges(ahd_ind_delta)}"
                )
                if self.config.include_topdown_metrics:
                    delta_summary = (
                        f"Test delta ({comp['model_label']} - {base['model_label']}): "
                        f"FPA_ind={_fmt_delta_pp(fpa_ind_delta)}, "
                        f"FPA_td={_fmt_delta_pp(fpa_td_delta)}, "
                        f"wAP_ind={_fmt_delta_pp(wap_ind_delta)}, "
                        f"wAP_td={_fmt_delta_pp(wap_td_delta)}, "
                        f"TICE_ind={_fmt_delta_pp(tice_ind_delta)}, "
                        f"TICE_td={_fmt_delta_pp(tice_td_delta)}, "
                        f"AHD_ind={_fmt_delta_edges(ahd_ind_delta)}, "
                        f"AHD_td={_fmt_delta_edges(ahd_td_delta)}"
                    )
                print(delta_summary)
