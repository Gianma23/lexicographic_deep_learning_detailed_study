from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import colors as mcolors

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv


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


RunSpec = Union[str, Path, Mapping[str, Any]]
RunData = MutableMapping[str, Any]
BEST_SELECTION_MODES = ("topdown", "independent")

def resolve_project_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd.parent]
    for candidate in candidates:
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


def parse_run(run_dir: Union[str, Path]) -> Dict[str, Any]:
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
    orthonormal_plugin_enabled = False
    orthonormal_plugin_loss: Optional[str] = None
    orthonormal_plugin_weight_mode: Optional[str] = None
    orthonormal_plugin_transform_mode: Optional[str] = None
    orthonormal_plugin_alpha: Optional[float] = None
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
        plugin_cfg = cfg.get("orthonormal_plugin", {})
        if isinstance(plugin_cfg, MappingABC):
            orthonormal_plugin_enabled = bool(plugin_cfg.get("enabled", False))
            raw_plugin_loss = plugin_cfg.get("loss")
            if raw_plugin_loss is not None:
                if not isinstance(raw_plugin_loss, str):
                    raise ValueError("orthonormal_plugin.loss in config_resolved.yaml must be a string.")
                orthonormal_plugin_loss = raw_plugin_loss
            raw_plugin_weight_mode = plugin_cfg.get("weight_mode")
            if raw_plugin_weight_mode is not None:
                if not isinstance(raw_plugin_weight_mode, str):
                    raise ValueError("orthonormal_plugin.weight_mode in config_resolved.yaml must be a string.")
                orthonormal_plugin_weight_mode = raw_plugin_weight_mode
            raw_plugin_transform_mode = plugin_cfg.get("transform_mode")
            if raw_plugin_transform_mode is not None:
                if not isinstance(raw_plugin_transform_mode, str):
                    raise ValueError("orthonormal_plugin.transform_mode in config_resolved.yaml must be a string.")
                orthonormal_plugin_transform_mode = raw_plugin_transform_mode
            raw_plugin_alpha = plugin_cfg.get("alpha")
            if raw_plugin_alpha is not None:
                try:
                    orthonormal_plugin_alpha = float(raw_plugin_alpha)
                except (TypeError, ValueError):
                    orthonormal_plugin_alpha = None

    best_epoch_events = _best_epoch_events_by_mode(epoch_events, test_results)
    best_epoch_event = best_epoch_events.get("topdown")

    return {
        "run_dir": run_path,
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
        "orthonormal_plugin_enabled": orthonormal_plugin_enabled,
        "orthonormal_plugin_loss": orthonormal_plugin_loss,
        "orthonormal_plugin_weight_mode": orthonormal_plugin_weight_mode,
        "orthonormal_plugin_transform_mode": orthonormal_plugin_transform_mode,
        "orthonormal_plugin_alpha": orthonormal_plugin_alpha,
        "best_epoch_events": best_epoch_events,
        "best_epoch_event": best_epoch_event,
    }


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
    names = run_data.get("level_names") or []
    if level_idx < len(names):
        return str(names[level_idx])
    return f"L{level_idx}"


def _normalize_dataset_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("dataset key must be a string.")
    if value in {"cifar-100", "cub-200-2011", "fgvc-aircraft", "inat19"}:
        return value
    raise ValueError(
        f"Unsupported dataset key '{value}'. "
        "Expected one of ['cifar-100', 'cub-200-2011', 'fgvc-aircraft', 'inat19']."
    )


def _resolve_run_dir(path_like: Union[str, Path], outputs_root: Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return outputs_root / path


def _read_hcc_cfg(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config_resolved.yaml"
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
    cfg_path = run_dir / "config_resolved.yaml"
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


def _detect_orthonormal_plugin_family(run_like: Mapping[str, Any], text: str) -> Optional[str]:
    plugin_enabled = bool(run_like.get("orthonormal_plugin_enabled", False))
    looks_like_plugin = plugin_enabled or "orthonormal_plugin" in text or "plugin" in text
    if not looks_like_plugin:
        return None

    loss_raw = run_like.get("orthonormal_plugin_loss", None)
    loss_mode = loss_raw.strip().lower() if isinstance(loss_raw, str) else ""
    transform_raw = run_like.get("orthonormal_plugin_transform_mode", None)
    transform_mode = transform_raw.strip().lower() if isinstance(transform_raw, str) else ""

    if transform_mode == "final_only" or "final_only" in text or "final only" in text:
        return "orthonormal_plugin_final_only"
    if loss_mode == "level_softmax_ce_reg" or "level_softmax_ce_reg" in text:
        return "orthonormal_plugin_loss_level_softmax_ce_reg"
    if loss_mode == "global_softmax_ce_reg" or "global_softmax_ce_reg" in text:
        return "orthonormal_plugin_loss_global_softmax_ce_reg"
    if loss_mode == "kl_reg" or "kl_reg" in text:
        return "orthonormal_plugin_loss_kl_reg"
    return "orthonormal_plugin_loss_global_softmax_ce_reg"


def _detect_color_family(run_like: Mapping[str, Any]) -> str:
    if bool(run_like.get("is_baseline", False)):
        return "baseline"

    label = str(run_like.get("label", "")).lower()
    run_dir_name = ""
    run_dir = run_like.get("run_dir", None)
    if run_dir is not None:
        run_dir_name = Path(run_dir).name.lower()

    text = f"{label} {run_dir_name}"
    plugin_family = _detect_orthonormal_plugin_family(run_like, text)
    if plugin_family is not None:
        return plugin_family

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
        # Non-Hier-COS models using the shared orthonormal plugin.
        "orthonormal_plugin_loss_kl_reg": ["#d1d5db", "#9ca3af", "#4b5563"],
        "orthonormal_plugin_loss_global_softmax_ce_reg": ["#f9a8d4", "#ec4899", "#9d174d"],
        "orthonormal_plugin_loss_level_softmax_ce_reg": ["#67e8f9", "#06b6d4", "#0e7490"],
        "orthonormal_plugin_final_only": ["#ddd6fe", "#a78bfa", "#6d28d9"],
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


@dataclass
class HCastAnalysisConfig:
    outputs_root: Path = Path(os.environ.get("OUTPUTS_ROOT", "/scratch/g.saggini1/outputs"))
    include_baselines: bool = True
    baseline_color: str = "#1f77b4"
    baseline_by_dataset: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "cifar-100": {"run_dir": "hcast_cifar100", "label": "H-CAST"},
            "cub-200-2011": {"run_dir": "hcast_cub200", "label": "H-CAST"},
            "fgvc-aircraft": {"run_dir": "hcast_aircraft", "label": "H-CAST"},
        }
    )
    manual_runs: List[RunSpec] = field(default_factory=list)
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


class HCastAnalysis:
    def __init__(self, config: HCastAnalysisConfig, run_data_list: List[RunData]) -> None:
        self.config = config
        self.run_data_list = run_data_list
        self.run_data_by_dataset: Dict[str, List[RunData]] = {}
        for run_data in run_data_list:
            self.run_data_by_dataset.setdefault(run_data["dataset_key"], []).append(run_data)

        for dataset_runs in self.run_data_by_dataset.values():
            dataset_runs.sort(key=lambda run: (0 if run.get("is_baseline") else 1, run.get("_run_order", 0)))

        if bool(config.auto_recolor_unlocked_runs):
            _apply_semantic_color_gradients(self.run_data_by_dataset)
        self.dataset_keys = list(self.run_data_by_dataset.keys())

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
            parsed = parse_run(run["run_dir"])
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
            has_log = (run_dir / "run_log.jsonl").exists()
            if not run_dir.exists() or not run_dir.is_dir() or not has_log:
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

        if not selected_dataset_keys:
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
            has_log = (baseline_run_dir / "run_log.jsonl").exists()
            if not baseline_run_dir.exists() or not baseline_run_dir.is_dir() or not has_log:
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

    def print_run_summary(self) -> None:
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
                plugin_enabled = bool(run_data.get("orthonormal_plugin_enabled", False))

                temp_txt = "" if temperature is None else f", T={temperature:g}"
                mode_txt = "" if projection_mode is None else f", proj_mode={projection_mode}"
                if strength_max is None or not np.isfinite(strength_max):
                    strength_txt = ""
                else:
                    strength_txt = f", strength_max={strength_max:g}"
                if plugin_enabled:
                    plugin_bits = [
                        f"plugin_loss={run_data.get('orthonormal_plugin_loss', 'unknown')}",
                        f"plugin_weight={run_data.get('orthonormal_plugin_weight_mode', 'unknown')}",
                        f"plugin_transform={run_data.get('orthonormal_plugin_transform_mode', 'unknown')}",
                    ]
                    plugin_alpha = run_data.get("orthonormal_plugin_alpha", None)
                    if plugin_alpha is not None and np.isfinite(plugin_alpha):
                        plugin_bits.append(f"plugin_alpha={plugin_alpha:g}")
                    plugin_txt = ", " + ", ".join(plugin_bits)
                else:
                    plugin_txt = ""
                baseline_txt = ", baseline" if is_baseline else ""

                print(
                    f"[{dataset_name}] {run_data['label']}: epochs={epoch_count}, "
                    f"best_td_epoch={best_td_epoch}, best_ind_epoch={best_ind_epoch}"
                    f"{temp_txt}{mode_txt}{strength_txt}{plugin_txt}{baseline_txt}"
                )

    def plot_validation_curves(
        self,
        metric_families: Optional[Sequence[Tuple[str, str, bool]]] = None,
        mode_specs: Optional[Sequence[Tuple[str, str, str, str]]] = None,
    ) -> None:
        metric_families = metric_families or [
            ("fpa", "Validation FPA (%)", True),
            ("weighted_ap", "Validation wAP (%)", True),
            ("tice", "Validation TICE (%)", True),
            ("ahd", "Validation AHD (edges)", False),
        ]
        mode_specs = mode_specs or [
            ("independent", "--", "independent", "x"),
            ("topdown", "-", "top-down", "o"),
        ]

        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

            fig, axes = plt.subplots(len(metric_families), 1, figsize=(10, 5 * len(metric_families)), sharex=True)
            if len(metric_families) == 1:
                axes = [axes]

            for ax, (metric_prefix, metric_title, is_percent) in zip(axes, metric_families):
                for run_data in dataset_runs:
                    for mode_key, line_style, mode_label, marker in mode_specs:
                        metric_key = f"{metric_prefix}_{mode_key}"
                        epochs, values = get_metric_series(run_data["epoch_events"], metric_key)
                        plot_values = values * 100.0 if is_percent else values
                        ax.plot(
                            epochs,
                            plot_values,
                            label=f"{run_data['label']} ({mode_label})",
                            color=run_data["color"],
                            linestyle=line_style,
                            linewidth=2.0,
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

                ax.set_title(metric_title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Score (%)" if is_percent else "Distance (edges)")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)

            plt.suptitle(
                f"{dataset_key}: Validation Metrics (Top-Down + Independent on Same Plot)",
                y=1.03,
                fontsize=13,
            )
            plt.tight_layout()
            plt.show()

    def plot_projection_diagnostics(
        self,
        base_diag_specs: Optional[Sequence[Union[Tuple[str, str, bool], Tuple[str, str, bool, str]]]] = None,
    ) -> None:
        base_diag_specs = base_diag_specs or [
            ("proj_constraint_alpha", "Constraint alpha", False, "val"),
            ("proj_constraint_strength", "Constraint strength", False, "val"),
            ("proj_temperature", "Projection temperature", False, "val"),
            ("proj_mode_intrinsic_soft", "Intrinsic-soft mode flag (1=true)", False, "val"),
        ]

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
            diag_specs.append((str(metric_key), str(title), bool(is_percent), source_norm))

        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

            active_diag_specs = []
            for metric_key, title, is_percent, source in diag_specs:
                has_finite = False
                for run_data in dataset_runs:
                    if source == "train":
                        _, values = get_train_metric_series(run_data["epoch_events"], metric_key)
                    else:
                        _, values = get_metric_series(run_data["epoch_events"], metric_key)
                    if np.any(np.isfinite(values)):
                        has_finite = True
                        break
                if has_finite:
                    active_diag_specs.append((metric_key, title, is_percent, source))

            if not active_diag_specs:
                print(f"[{dataset_key}] No projection diagnostics found in selected runs.")
                continue

            fig, axes = plt.subplots(len(active_diag_specs), 1, figsize=(10, 4.3 * len(active_diag_specs)), sharex=True)
            if len(active_diag_specs) == 1:
                axes = [axes]

            for ax, (metric_key, title, is_percent, source) in zip(axes, active_diag_specs):
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
                        linewidth=2.0,
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
                                    s=45,
                                    zorder=4,
                                )

                title_prefix = "Validation" if source == "val" else "Training"
                ax.set_title(f"{title_prefix} {title}")
                ax.set_ylabel("Value (%)" if is_percent else "Value")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)

            axes[-1].set_xlabel("Epoch")
            plt.suptitle(f"{dataset_key}: Projection Diagnostics", y=1.01, fontsize=13)
            plt.tight_layout()
            plt.show()

    def plot_training_losses(
        self,
        aggregate_loss_keys: Optional[Sequence[str]] = None,
    ) -> None:
        aggregate_loss_keys = aggregate_loss_keys or ["total", "ce", "reg", "kl", "level_ce", "gk_loss"]

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
            metric_specs = [(key, key) for key in aggregate_loss_keys]
            for level_idx in level_loss_ids:
                level_name = get_level_label(level_idx, dataset_runs[0])
                metric_specs.append((f"loss_level_{level_idx}", f"loss_{level_name} (L{level_idx})"))

            n_metrics = len(metric_specs)
            ncols = 2
            nrows = int(np.ceil(n_metrics / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.6 * nrows), sharex=True)
            axes = np.array(axes).reshape(-1)

            for ax, (metric_key, metric_title) in zip(axes, metric_specs):
                for run_data in dataset_runs:
                    epochs, values = get_train_loss_series(run_data["epoch_events"], metric_key)
                    ax.plot(
                        epochs,
                        values,
                        label=run_data["label"],
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

            plt.suptitle(f"{dataset_key}: Training Losses", y=1.02, fontsize=13)
            plt.tight_layout()
            plt.show()

    def plot_per_run_per_level_training_losses(self) -> None:
        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

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
                        int(key.rsplit("_", 1)[-1])
                        for key in all_loss_keys
                        if key.startswith("loss_level_") and key.rsplit("_", 1)[-1].isdigit()
                    }
                )
                if not level_loss_ids:
                    print(f"[{dataset_key}] No per-level train losses for {run_data['label']}")
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
                    print(f"[{dataset_key}] No finite per-level train losses for {run_data['label']}")
                    continue

                ax.set_title(f"{run_data['label']}: Per-Level Training Losses ({dataset_key})")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.grid(True, alpha=0.3)
                ax.legend(ncol=2, fontsize=9)
                plt.tight_layout()
                plt.show()

    def plot_gradient_diagnostics(
        self,
        smooth_window: int = 5,
        show_raw_overlay: bool = False,
        raw_alpha: float = 0.12,
        n_cols: int = 3,
        log_scale_for_norms: bool = True,
        ax_width: float = 6.0,
        ax_height: float = 3.8,
        legend_gutter_ratio: float = 0.78,
        legend_max_rows_per_col: int = 10,
    ) -> None:
        plot_ncols = max(1, int(n_cols))
        gradient_level_specs = [
            (
                "coarse",
                [
                    ("t1", "coarse"),
                    ("t2", "coarse"),
                    ("t3", "coarse"),
                    ("t2t1", "coarse"),
                    ("t3t2t1", "coarse"),
                ],
            ),
            (
                "mid",
                [
                    ("t1", "mid"),
                    ("t2", "mid"),
                    ("t2t1", "mid"),
                ],
            ),
            (
                "fine",
                [
                    ("t1", "fine"),
                ],
            ),
        ]
        trunk_level_order = [
            trunk_level for _level_name, trunk_levels in gradient_level_specs for trunk_level in trunk_levels
        ]
        param_trunk_order = ["t1", "t2", "t3", "t2t1", "t3t2t1"]

        def _trunk_label(trunk_name: str) -> str:
            return trunk_name.upper()

        metric_titles: Dict[str, str] = {
            "post_projection_applied_t2_mid_coarse": "Post projection applied (T2, mid vs coarse, 1=true)",
            "post_projection_applied_t1_mid_coarse": "Post projection applied (T1, mid vs coarse, 1=true)",
            "post_projection_applied_t1_fine_higher": "Post projection applied (T1, fine vs coarse+mid_proj, 1=true)",
            "cos_t2_mid_coarse": "Cosine (T2, mid vs coarse)",
            "cos_t1_mid_coarse": "Cosine (T1, mid vs coarse)",
            "cos_t2t1_mid_coarse": "Cosine (T2T1, mid vs coarse)",
            "cos_t1_fine_higher": "Cosine (T1, fine vs coarse+mid)",
            "cos_t1_fine_coarse": "Cosine (T1, fine vs coarse)",
            "cos_t1_fine_mid": "Cosine (T1, fine vs mid)",
            "post_cos_t2_mid_proj_coarse": "Post cosine (T2, mid_proj vs coarse)",
            "post_cos_t1_mid_proj_coarse": "Post cosine (T1, mid_proj vs coarse)",
            "post_cos_t2t1_mid_proj_coarse": "Post cosine (T2T1, mid_proj vs coarse)",
            "post_cos_t1_fine_proj_higher": "Post cosine (T1, fine_proj vs coarse+mid_proj)",
            "post_cos_t1_fine_proj_coarse": "Post cosine (T1, fine_proj vs coarse)",
            "post_cos_t1_fine_proj_mid_proj": "Post cosine (T1, fine_proj vs mid_proj)",
        }

        for trunk_name, level_name in trunk_level_order:
            trunk_tag = _trunk_label(trunk_name)
            metric_titles[f"grad_norm_{trunk_name}_{level_name}"] = f"Gradient norm ({trunk_tag}, {level_name})"
            metric_titles[f"post_grad_norm_{trunk_name}_{level_name}"] = f"Post gradient norm ({trunk_tag}, {level_name})"

        for trunk_name in param_trunk_order:
            trunk_tag = _trunk_label(trunk_name)
            metric_titles[f"param_norm_{trunk_name}"] = f"Parameter norm ({trunk_tag})"
            metric_titles[f"delta_param_norm_{trunk_name}"] = f"Parameter delta norm ({trunk_tag}, epoch)"

        pre_grad_groups = [
            (
                level_name,
                [f"grad_norm_{trunk}_{grad_level}" for trunk, grad_level in trunk_levels],
            )
            for level_name, trunk_levels in gradient_level_specs
        ]
        post_grad_groups = [
            (
                level_name,
                [f"post_grad_norm_{trunk}_{grad_level}" for trunk, grad_level in trunk_levels],
            )
            for level_name, trunk_levels in gradient_level_specs
        ]
        pre_grad_keys = [metric_key for _level_name, metric_keys in pre_grad_groups for metric_key in metric_keys]
        post_grad_keys = [metric_key for _level_name, metric_keys in post_grad_groups for metric_key in metric_keys]
        param_norm_keys = [f"param_norm_{trunk}" for trunk in param_trunk_order]
        delta_param_norm_keys = [f"delta_param_norm_{trunk}" for trunk in param_trunk_order]
        pre_cosine_groups = [
            (
                "mid vs coarse",
                [
                    "cos_t2_mid_coarse",
                    "cos_t1_mid_coarse",
                    "cos_t2t1_mid_coarse",
                ],
            ),
            (
                "fine vs coarse+mid and components",
                [
                    "cos_t1_fine_higher",
                    "cos_t1_fine_coarse",
                    "cos_t1_fine_mid",
                ],
            ),
        ]
        post_cosine_groups = [
            (
                "mid_proj vs coarse",
                [
                    "post_cos_t2_mid_proj_coarse",
                    "post_cos_t1_mid_proj_coarse",
                    "post_cos_t2t1_mid_proj_coarse",
                ],
            ),
            (
                "fine_proj vs coarse+mid_proj and components",
                [
                    "post_cos_t1_fine_proj_higher",
                    "post_cos_t1_fine_proj_coarse",
                    "post_cos_t1_fine_proj_mid_proj",
                ],
            ),
        ]
        requested_post_cosine_keys = [
            metric_key for _group_name, metric_keys in post_cosine_groups for metric_key in metric_keys
        ]

        base_metric_groups = [
            ("Parameter Norms (Hierarchy Order: T1->T2->T3 + aggregate trunks)", param_norm_keys, "norm"),
            ("Parameter Delta Norms (Hierarchy Order: T1->T2->T3 + aggregate trunks)", delta_param_norm_keys, "delta"),
            (
                "Lex Projection Applied Flags",
                [
                    "post_projection_applied_t2_mid_coarse",
                    "post_projection_applied_t1_mid_coarse",
                    "post_projection_applied_t1_fine_higher",
                ],
                "flag",
            ),
        ]

        pre_post_pair_groups = [
            (
                level_name,
                list(zip(pre_keys, post_keys)),
            )
            for (level_name, pre_keys), (_post_level_name, post_keys) in zip(pre_grad_groups, post_grad_groups)
        ]
        pre_post_pairs = [pair for _level_name, pairs in pre_post_pair_groups for pair in pairs]
        cosine_pre_post_pair_groups = [
            (
                "mid vs coarse",
                [
                    ("cos_t2_mid_coarse", "post_cos_t2_mid_proj_coarse"),
                    ("cos_t1_mid_coarse", "post_cos_t1_mid_proj_coarse"),
                    ("cos_t2t1_mid_coarse", "post_cos_t2t1_mid_proj_coarse"),
                ],
            ),
            (
                "fine vs coarse+mid and components",
                [
                    ("cos_t1_fine_higher", "post_cos_t1_fine_proj_higher"),
                    ("cos_t1_fine_coarse", "post_cos_t1_fine_proj_coarse"),
                    ("cos_t1_fine_mid", "post_cos_t1_fine_proj_mid_proj"),
                ],
            ),
        ]
        cosine_pre_post_pairs = [
            pair for _group_name, pairs in cosine_pre_post_pair_groups for pair in pairs
        ]

        def _param_curve_style(run_data: RunData) -> Dict[str, Any]:
            """Use baseline solid and comparison runs dashed on parameter plots."""
            label = str(run_data.get("label", "")).lower()
            is_baseline = bool(run_data.get("is_baseline", False)) or label in {"h-cast", "hcast", "baseline"}
            return {
                "linestyle": "-" if is_baseline else "--",
                "marker": None,
                "markevery": None,
                "linewidth": 2.4 if is_baseline else 2.2,
            }

        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

            has_post_requested_cosine = False
            for run_data in dataset_runs:
                for event in run_data["epoch_events"]:
                    for source_key in ("train_metrics_norm", "train_metrics"):
                        metric_source = event.get(source_key, {})
                        if not isinstance(metric_source, dict):
                            continue
                        for post_key in requested_post_cosine_keys:
                            value_float = _metric_value_from_source(metric_source, post_key)
                            if np.isfinite(value_float):
                                has_post_requested_cosine = True
                                break
                        if has_post_requested_cosine:
                            break
                    if has_post_requested_cosine:
                        break
                if has_post_requested_cosine:
                    break

            metric_groups = list(base_metric_groups)
            if not has_post_requested_cosine:
                metric_groups.extend(
                    [
                        (f"Cosines - Pre - {group_name}", metric_keys, "cosine")
                        for group_name, metric_keys in pre_cosine_groups
                    ]
                )

            base_metric_keys = sorted(
                {metric for _, group_metrics, _ in metric_groups for metric in group_metrics}.union(
                    {metric for pair in pre_post_pairs for metric in pair}
                ).union(
                    {metric for pair in cosine_pre_post_pairs for metric in pair}
                )
            )

            run_series_cache = []
            for run_data in dataset_runs:
                metric_map = _build_train_metric_map(run_data["epoch_events"], base_metric_keys)
                run_series_cache.append((run_data, metric_map))

            plotted_any_group = False
            for group_title, metric_keys, metric_kind in metric_groups:
                active_keys = [metric_key for metric_key in metric_keys if has_any_finite(run_series_cache, metric_key)]
                if not active_keys:
                    continue

                plotted_any_group = True
                n_metrics = len(active_keys)
                ncols = plot_ncols
                nrows = int(np.ceil(n_metrics / ncols))
                fig_width = ax_width * ncols
                fig_height = ax_height * nrows
                fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=True)
                axes = np.array(axes).reshape(-1)

                legend_handles = None
                legend_labels = None
                is_parameter_group = group_title.startswith("Parameter Norms") or group_title.startswith(
                    "Parameter Delta Norms"
                )

                for plot_idx, (ax, metric_key) in enumerate(zip(axes, active_keys)):
                    for run_data, metric_map in run_series_cache:
                        epochs, values = metric_map[metric_key]
                        smooth_values = moving_average_ignore_nan(values, smooth_window)
                        line_style = _param_curve_style(run_data) if is_parameter_group else {
                            "linestyle": "-",
                            "marker": None,
                            "markevery": None,
                            "linewidth": 2.2,
                        }

                        ax.plot(
                            epochs,
                            smooth_values,
                            label=run_data["label"],
                            color=run_data["color"],
                            linestyle=line_style["linestyle"],
                            marker=line_style["marker"],
                            markevery=line_style["markevery"],
                            markersize=4.0,
                            linewidth=line_style["linewidth"],
                        )

                        if show_raw_overlay and int(smooth_window) > 1:
                            ax.plot(
                                epochs,
                                values,
                                color=run_data["color"],
                                linewidth=1.0,
                                linestyle=line_style["linestyle"] if is_parameter_group else "-",
                                alpha=float(raw_alpha),
                            )

                    if metric_kind == "cosine":
                        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
                        ax.set_ylim(-1.05, 1.05)
                    elif metric_kind == "flag":
                        ax.set_ylim(-0.05, 1.05)
                    elif metric_kind == "norm" and bool(log_scale_for_norms):
                        robust_range = get_positive_robust_range(run_series_cache, metric_key)
                        if robust_range is not None:
                            ymin, ymax = robust_range
                            ax.set_yscale("log")
                            ax.set_ylim(ymin, ymax)
                    elif metric_kind == "mixed":
                        if "cos_" in metric_key:
                            ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
                            ax.set_ylim(-1.05, 1.05)
                        elif bool(log_scale_for_norms):
                            robust_range = get_positive_robust_range(run_series_cache, metric_key)
                            if robust_range is not None:
                                ymin, ymax = robust_range
                                ax.set_yscale("log")
                                ax.set_ylim(ymin, ymax)

                    ax.set_title(metric_titles.get(metric_key, metric_key))
                    ax.set_ylabel("Value")
                    ax.grid(True, alpha=0.25)

                    if plot_idx >= (nrows - 1) * ncols:
                        ax.set_xlabel("Epoch")

                    if legend_handles is None:
                        handles, labels = ax.get_legend_handles_labels()
                        if handles:
                            legend_handles, legend_labels = handles, labels

                for ax in axes[n_metrics:]:
                    ax.set_visible(False)

                layout_rect = (0.0, 0.0, 1.0, 0.95)
                if legend_handles:
                    legend_ncol = max(1, int(np.ceil(len(legend_labels) / legend_max_rows_per_col)))
                    fig.legend(
                        legend_handles,
                        legend_labels,
                        loc="upper left",
                        bbox_to_anchor=(legend_gutter_ratio + 0.01, 0.975),
                        ncol=legend_ncol,
                        borderaxespad=0.0,
                        frameon=False,
                        title="Runs",
                        fontsize=9,
                    )
                    layout_rect = (0.0, 0.0, legend_gutter_ratio, 0.94)

                fig.suptitle(f"{dataset_key}: {group_title}", fontsize=13, y=0.99)
                plt.tight_layout(rect=layout_rect)
                plt.show()

            for level_name, pair_group in pre_post_pair_groups:
                active_pairs = [
                    pair
                    for pair in pair_group
                    if has_any_finite(run_series_cache, pair[0]) or has_any_finite(run_series_cache, pair[1])
                ]
                if not active_pairs:
                    continue

                plotted_any_group = True
                n_metrics = len(active_pairs)
                ncols = plot_ncols
                nrows = int(np.ceil(n_metrics / ncols))
                fig_width = ax_width * ncols
                fig_height = ax_height * nrows
                fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=True)
                axes = np.array(axes).reshape(-1)

                legend_handles = None
                legend_labels = None

                for plot_idx, (ax, (pre_key, post_key)) in enumerate(zip(axes, active_pairs)):
                    for run_data, metric_map in run_series_cache:
                        epochs, pre_values = metric_map[pre_key]
                        _, post_values = metric_map[post_key]
                        pre_smooth = moving_average_ignore_nan(pre_values, smooth_window)
                        post_smooth = moving_average_ignore_nan(post_values, smooth_window)

                        ax.plot(
                            epochs,
                            pre_smooth,
                            label=f"{run_data['label']} (pre)",
                            color=run_data["color"],
                            linewidth=2.2,
                        )
                        ax.plot(
                            epochs,
                            post_smooth,
                            label=f"{run_data['label']} (post)",
                            color=run_data["color"],
                            linewidth=2.2,
                            linestyle="--",
                        )

                        if show_raw_overlay and int(smooth_window) > 1:
                            ax.plot(
                                epochs,
                                pre_values,
                                color=run_data["color"],
                                linewidth=1.0,
                                alpha=float(raw_alpha),
                            )
                            ax.plot(
                                epochs,
                                post_values,
                                color=run_data["color"],
                                linewidth=1.0,
                                alpha=float(raw_alpha),
                                linestyle="--",
                            )

                    if bool(log_scale_for_norms):
                        robust_range = _positive_robust_range_for_keys(run_series_cache, [pre_key, post_key])
                        if robust_range is not None:
                            ymin, ymax = robust_range
                            ax.set_yscale("log")
                            ax.set_ylim(ymin, ymax)

                    ax.set_title(metric_titles.get(pre_key, pre_key).replace("Gradient norm", "Pre/Post gradient norm"))
                    ax.set_ylabel("Value")
                    ax.grid(True, alpha=0.25)

                    if plot_idx >= (nrows - 1) * ncols:
                        ax.set_xlabel("Epoch")

                    if legend_handles is None:
                        handles, labels = ax.get_legend_handles_labels()
                        if handles:
                            legend_handles, legend_labels = handles, labels

                for ax in axes[n_metrics:]:
                    ax.set_visible(False)

                layout_rect = (0.0, 0.0, 1.0, 0.95)
                if legend_handles:
                    legend_ncol = max(1, int(np.ceil(len(legend_labels) / legend_max_rows_per_col)))
                    fig.legend(
                        legend_handles,
                        legend_labels,
                        loc="upper left",
                        bbox_to_anchor=(legend_gutter_ratio + 0.01, 0.975),
                        ncol=legend_ncol,
                        borderaxespad=0.0,
                        frameon=False,
                        title="Runs (solid=pre, dashed=post)",
                        fontsize=9,
                    )
                    layout_rect = (0.0, 0.0, legend_gutter_ratio, 0.94)

                fig.suptitle(
                    f"{dataset_key}: Lex Pre vs Post Gradient Norms - {level_name.title()}",
                    fontsize=13,
                    y=0.99,
                )
                plt.tight_layout(rect=layout_rect)
                plt.show()

            if has_post_requested_cosine:
                for group_name, pair_group in cosine_pre_post_pair_groups:
                    active_pairs = [
                        pair
                        for pair in pair_group
                        if has_any_finite(run_series_cache, pair[0]) or has_any_finite(run_series_cache, pair[1])
                    ]
                    if not active_pairs:
                        continue

                    plotted_any_group = True
                    n_metrics = len(active_pairs)
                    ncols = plot_ncols
                    nrows = int(np.ceil(n_metrics / ncols))
                    fig_width = ax_width * ncols
                    fig_height = ax_height * nrows
                    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharex=True)
                    axes = np.array(axes).reshape(-1)

                    legend_handles = None
                    legend_labels = None

                    for plot_idx, (ax, (pre_key, post_key)) in enumerate(zip(axes, active_pairs)):
                        for run_data, metric_map in run_series_cache:
                            epochs, pre_values = metric_map[pre_key]
                            _, post_values = metric_map[post_key]
                            pre_smooth = moving_average_ignore_nan(pre_values, smooth_window)
                            post_smooth = moving_average_ignore_nan(post_values, smooth_window)

                            ax.plot(
                                epochs,
                                pre_smooth,
                                label=f"{run_data['label']} (pre)",
                                color=run_data["color"],
                                linewidth=2.2,
                            )
                            ax.plot(
                                epochs,
                                post_smooth,
                                label=f"{run_data['label']} (post)",
                                color=run_data["color"],
                                linewidth=2.2,
                                linestyle="--",
                            )

                            if show_raw_overlay and int(smooth_window) > 1:
                                ax.plot(
                                    epochs,
                                    pre_values,
                                    color=run_data["color"],
                                    linewidth=1.0,
                                    alpha=float(raw_alpha),
                                )
                                ax.plot(
                                    epochs,
                                    post_values,
                                    color=run_data["color"],
                                    linewidth=1.0,
                                    alpha=float(raw_alpha),
                                    linestyle="--",
                                )

                        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
                        ax.set_ylim(-1.05, 1.05)
                        ax.set_title(metric_titles.get(pre_key, pre_key).replace("Cosine", "Pre/Post cosine"))
                        ax.set_ylabel("Value")
                        ax.grid(True, alpha=0.25)

                        if plot_idx >= (nrows - 1) * ncols:
                            ax.set_xlabel("Epoch")

                        if legend_handles is None:
                            handles, labels = ax.get_legend_handles_labels()
                            if handles:
                                legend_handles, legend_labels = handles, labels

                    for ax in axes[n_metrics:]:
                        ax.set_visible(False)

                    layout_rect = (0.0, 0.0, 1.0, 0.95)
                    if legend_handles:
                        legend_ncol = max(1, int(np.ceil(len(legend_labels) / legend_max_rows_per_col)))
                        fig.legend(
                            legend_handles,
                            legend_labels,
                            loc="upper left",
                            bbox_to_anchor=(legend_gutter_ratio + 0.01, 0.975),
                            ncol=legend_ncol,
                            borderaxespad=0.0,
                            frameon=False,
                            title="Runs (solid=pre, dashed=post)",
                            fontsize=9,
                        )
                        layout_rect = (0.0, 0.0, legend_gutter_ratio, 0.94)

                    fig.suptitle(
                        f"{dataset_key}: Lex Pre vs Post Cosines - {group_name}",
                        fontsize=13,
                        y=0.99,
                    )
                    plt.tight_layout(rect=layout_rect)
                    plt.show()

            if not plotted_any_group:
                print(f"[{dataset_key}] No trunk gradient/parameter diagnostics found in selected runs.")

    def plot_per_level_validation_accuracy(
        self, mode_specs: Optional[Sequence[Tuple[str, str, str, str]]] = None
    ) -> None:
        mode_specs = mode_specs or [
            ("independent", "--", "independent", "x"),
            ("topdown", "-", "top-down", "o"),
        ]

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
                        (key.startswith("acc_level_topdown_") or key.startswith("acc_level_independent_"))
                        and key.rsplit("_", 1)[-1].isdigit()
                    )
                }
            )

            if not all_level_ids:
                print(f"[{dataset_key}] No per-level validation accuracy metrics found in selected runs.")
                continue

            fig, axes = plt.subplots(len(all_level_ids), 1, figsize=(10, 5 * len(all_level_ids)), sharex=True)
            if len(all_level_ids) == 1:
                axes = [axes]

            for ax, level_idx in zip(axes, all_level_ids):
                level_label = get_level_label(level_idx, dataset_runs[0])
                for run_data in dataset_runs:
                    for mode_key, line_style, mode_label, marker in mode_specs:
                        metric_key = f"acc_level_{mode_key}_{level_idx}"
                        epochs, values = get_metric_series(run_data["epoch_events"], metric_key)
                        ax.plot(
                            epochs,
                            values * 100.0,
                            label=f"{run_data['label']} ({mode_label})",
                            color=run_data["color"],
                            linestyle=line_style,
                            linewidth=2.0,
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

                ax.set_title(f"Validation Accuracy - {level_label} (L{level_idx})")
                ax.set_ylabel("Accuracy (%)")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)

            axes[-1].set_xlabel("Epoch")
            plt.suptitle(f"{dataset_key}: Per-Level Validation Accuracy", y=1.01, fontsize=13)
            plt.tight_layout()
            plt.show()

    def show_final_test_tables(self) -> None:
        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if len(dataset_runs) < 2:
                print(f"[{dataset_key}] Skipping final test table: need at least two runs (base + comparison).")
                continue

            base = dataset_runs[0]

            level_ids_from_test = {
                int(key.rsplit("_", 1)[-1])
                for run_data in dataset_runs
                for mode in BEST_SELECTION_MODES
                for key in _test_metrics_for_mode(run_data, mode).keys()
                if (
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
                            (key.startswith("acc_level_topdown_") or key.startswith("acc_level_independent_"))
                            and key.rsplit("_", 1)[-1].isdigit()
                        )
                    }
                )

            metric_rows: List[Tuple[str, str]] = [
                ("fpa_independent", "FPA independent"),
                ("fpa_topdown", "FPA top-down"),
                ("weighted_ap_independent", "wAP independent"),
                ("weighted_ap_topdown", "wAP top-down"),
                ("tice_independent", "TICE independent"),
                ("tice_topdown", "TICE top-down"),
                ("ahd_independent", "AHD independent"),
                ("ahd_topdown", "AHD top-down"),
            ]

            for level_idx in all_level_ids:
                level_label = get_level_label(level_idx, base)
                metric_rows.append((f"acc_level_independent_{level_idx}", f"Acc independent {level_label} (L{level_idx})"))
                metric_rows.append((f"acc_level_topdown_{level_idx}", f"Acc top-down {level_label} (L{level_idx})"))

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
                best_indices, second_best_indices = _best_and_second_best_indices(metric_key, metric_values)
                best_by_metric[metric_key] = best_indices
                second_best_by_metric[metric_key] = second_best_indices

            best_epoch_cells: List[str] = []
            for run_data in dataset_runs:
                test_results = run_data.get("test_results", {})
                td_epoch = None
                ind_epoch = None
                if isinstance(test_results, MappingABC):
                    td_section = test_results.get("topdown")
                    if isinstance(td_section, MappingABC):
                        td_epoch = _coerce_int(td_section.get("best_epoch"))
                    ind_section = test_results.get("independent")
                    if isinstance(ind_section, MappingABC):
                        ind_epoch = _coerce_int(ind_section.get("best_epoch"))

                best_events = run_data.get("best_epoch_events", {})
                if td_epoch is None:
                    td_event = (
                        best_events.get("topdown")
                        if isinstance(best_events, MappingABC)
                        else run_data.get("best_epoch_event")
                    )
                    td_epoch = _coerce_int(td_event.get("epoch")) if isinstance(td_event, MappingABC) else None
                if ind_epoch is None:
                    ind_event = (
                        best_events.get("independent")
                        if isinstance(best_events, MappingABC)
                        else run_data.get("best_epoch_event")
                    )
                    ind_epoch = _coerce_int(ind_event.get("epoch")) if isinstance(ind_event, MappingABC) else None
                if ind_epoch is None:
                    ind_epoch = td_epoch

                if td_epoch is None and ind_epoch is None:
                    best_epoch_cells.append("n/a")
                else:
                    td_text = str(td_epoch) if td_epoch is not None else "n/a"
                    ind_text = str(ind_epoch) if ind_epoch is not None else "n/a"
                    best_epoch_cells.append(f"{td_text}/{ind_text}")

            header_labels = ["Metric"] + [run_data["label"] for run_data in dataset_runs]
            table_lines = [
                f"### Dataset: `{dataset_key}`",
                f"Baseline run: **{base['label']}**",
                "",
                "| " + " | ".join(header_labels) + " |",
                "|---|" + "|".join(["---:"] * (len(header_labels) - 1)) + "|",
                "| Best epoch (TD/Ind) | " + " | ".join(best_epoch_cells) + " |",
            ]

            for metric_key, metric_name in metric_rows:
                metric_label = f"{metric_name} {_metric_goal_arrow(metric_key)}"
                row_cells = [metric_label]
                values = values_by_metric[metric_key]
                for run_idx, value in enumerate(values):
                    cell = _fmt_value(metric_key, value) if run_idx == 0 else _merged_comp_cell(metric_key, value, values[0])
                    if run_idx in best_by_metric[metric_key] and cell != "n/a":
                        cell = f"**{cell}**"
                    elif run_idx in second_best_by_metric[metric_key] and cell != "n/a":
                        cell = f"<u>{cell}</u>"
                    row_cells.append(cell)
                table_lines.append("| " + " | ".join(row_cells) + " |")

            show_markdown("\n".join(table_lines))
