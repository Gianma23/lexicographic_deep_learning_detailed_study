from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import colors as mcolors

try:
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover - notebook runtime dependent
    Markdown = None
    display = None


plt.style.use("seaborn-v0_8-whitegrid")


RunSpec = Union[str, Path, Mapping[str, Any]]
RunData = MutableMapping[str, Any]

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


def metric_for_best(metrics: Mapping[str, float]) -> float:
    m = normalize_metrics(metrics)

    has_fpa = "fpa_topdown" in m
    has_tice = "tice_topdown" in m
    has_wap = "weighted_ap_topdown" in m

    if has_fpa or has_tice or has_wap:
        fpa = float(m.get("fpa_topdown", 0.0))
        neg_tice = -float(m.get("tice_topdown", 1.0))
        wap = float(m.get("weighted_ap_topdown", 0.0))
        return float(fpa + 1e-3 * neg_tice + 1e-6 * wap)

    deepest_topdown = [
        key
        for key in m
        if key.startswith("acc_level_topdown_") and key[len("acc_level_topdown_") :].isdigit()
    ]
    deepest_independent = [
        key
        for key in m
        if key.startswith("acc_level_independent_") and key[len("acc_level_independent_") :].isdigit()
    ]
    deepest = deepest_topdown or deepest_independent
    if not deepest:
        return float(m.get("fpa_topdown", 0.0))

    deepest_key = max(deepest, key=lambda key: int(key.rsplit("_", 1)[-1]))
    primary = float(m.get(deepest_key, 0.0))
    tie = float(m.get("fpa_topdown", 0.0))
    return float(primary + 1e-3 * tie)


def parse_run(run_dir: Union[str, Path]) -> Dict[str, Any]:
    run_path = Path(run_dir)
    events = load_jsonl_events(run_path / "run_log.jsonl")
    epoch_events = [event for event in events if event.get("event") == "epoch"]

    for event in epoch_events:
        event["train_metrics_norm"] = normalize_metrics(event.get("train_metrics", {}))
        event["val_metrics_norm"] = normalize_metrics(event.get("val_metrics", {}))
        event["rank_score"] = metric_for_best(event["val_metrics_norm"])

    test_events = [event for event in events if event.get("event") == "test"]
    test_event = test_events[-1] if test_events else {}
    test_metrics = normalize_metrics(test_event.get("test_metrics", {}))

    test_metrics_yaml: Dict[str, Any] = {}
    test_yaml_path = run_path / "test_metrics.yaml"
    if test_yaml_path.exists():
        test_metrics_yaml = load_yaml(test_yaml_path) or {}
        if not test_metrics:
            test_metrics = normalize_metrics(test_metrics_yaml.get("test_metrics", {}))

    level_names: List[str] = []
    cfg_path = run_path / "config_resolved.yaml"
    if cfg_path.exists():
        cfg = load_yaml(cfg_path) or {}
        level_names = list(cfg.get("dataset", {}).get("levels", []))

    best_epoch_event = None
    if epoch_events:
        best_idx = int(np.argmax([event["rank_score"] for event in epoch_events]))
        best_epoch_event = epoch_events[best_idx]

    return {
        "run_dir": run_path,
        "events": events,
        "epoch_events": epoch_events,
        "test_event": test_event,
        "test_metrics": test_metrics,
        "test_metrics_yaml": test_metrics_yaml,
        "level_names": level_names,
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
    normalized = "".join(ch for ch in str(value).strip().lower() if ch.isalnum())
    return normalized or None


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
    return str(raw_name)


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


def _resolve_baseline_spec(
    dataset_key: str, baseline_lookup: Mapping[str, Mapping[str, Any]]
) -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
    spec = baseline_lookup.get(dataset_key)
    if spec is not None:
        return spec, dataset_key

    match = re.match(r"^(.*?)(?:19|20)\d{2}$", str(dataset_key))
    if match:
        stripped_key = match.group(1)
        spec = baseline_lookup.get(stripped_key)
        if spec is not None:
            return spec, stripped_key

    return None, None


def _detect_color_family(run_like: Mapping[str, Any]) -> str:
    if bool(run_like.get("is_baseline", False)):
        return "baseline"

    label = str(run_like.get("label", "")).lower()
    run_dir_name = ""
    run_dir = run_like.get("run_dir", None)
    if run_dir is not None:
        run_dir_name = Path(run_dir).name.lower()

    text = f"{label} {run_dir_name}"
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
        # HCC variants use a perceptually clear warm ramp instead of similar greens.
        "hcc": ["#f6d32d", "#f59e0b", "#ef4444", "#991b1b"],
        # Lexicographic runs use a green/teal ramp, separated from baseline blue and HCC warm colors.
        "lex": ["#006d5b", "#009e73", "#20c997"],
        "other": ["#9ca3af", "#4b5563"],
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


def _best_indices(metric_key: str, values: Sequence[float]) -> set[int]:
    finite_pairs = [(idx, val) for idx, val in enumerate(values) if np.isfinite(val)]
    if not finite_pairs:
        return set()

    best_value = min(val for _, val in finite_pairs) if _metric_goal_is_lower_better(metric_key) else max(
        val for _, val in finite_pairs
    )
    return {idx for idx, val in finite_pairs if np.isclose(val, best_value, rtol=1e-9, atol=1e-12)}


@dataclass
class HCastAnalysisConfig:
    outputs_root: Path = Path("/scratch/g.saggini1/outputs")
    include_baselines: bool = True
    baseline_color: str = "#1f77b4"
    baseline_by_dataset: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "cifar100": {"run_dir": "hcast_cifar100", "label": "H-CAST"},
            "cub200": {"run_dir": "hcast_cub200", "label": "H-CAST"},
            "fgvcaircraft": {"run_dir": "hcast_aircraft", "label": "H-CAST"},
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
            parsed["dataset_key"] = _normalize_dataset_key(parsed["dataset_name"]) or parsed["dataset_name"]
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
        baseline_lookup = {
            _normalize_dataset_key(key): value for key, value in config.baseline_by_dataset.items() if _normalize_dataset_key(key)
        }

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
            baseline_spec, matched_key = _resolve_baseline_spec(dataset_key, baseline_lookup)
            if not baseline_spec:
                available_keys = ", ".join(sorted(baseline_lookup.keys()))
                print(f"No baseline configured for dataset key: {dataset_key}. Available keys: {available_keys}")
                continue
            if matched_key != dataset_key:
                print(f"Using baseline key {matched_key} for dataset key {dataset_key}")

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
                continue

            baseline_meta = _run_meta_from_dir(baseline_run_dir)
            baseline_has_explicit_color = ("color" in baseline_spec) and (baseline_spec.get("color") is not None)
            runs.append(
                {
                    "label": baseline_spec.get("label", "H-CAST"),
                    "run_dir": baseline_run_dir,
                    "color": baseline_spec.get("color", config.baseline_color),
                    "color_locked": bool(baseline_has_explicit_color),
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
                best_epoch = run_data["best_epoch_event"]["epoch"] if run_data["best_epoch_event"] else None
                best_score = run_data["best_epoch_event"]["rank_score"] if run_data["best_epoch_event"] else None
                temperature = run_data.get("temperature", None)
                projection_mode = run_data.get("hcc_projection_mode", None)
                strength_max = run_data.get("hcc_constraint_strength_max", None)
                dataset_name = run_data.get("dataset_name", "unknown")
                is_baseline = bool(run_data.get("is_baseline", False))

                temp_txt = "" if temperature is None else f", T={temperature:g}"
                mode_txt = "" if projection_mode is None else f", proj_mode={projection_mode}"
                if strength_max is None or not np.isfinite(strength_max):
                    strength_txt = ""
                else:
                    strength_txt = f", strength_max={strength_max:g}"
                baseline_txt = ", baseline" if is_baseline else ""

                print(
                    f"[{dataset_name}] {run_data['label']}: epochs={epoch_count}, best_epoch={best_epoch}, "
                    f"best_score={best_score}{temp_txt}{mode_txt}{strength_txt}{baseline_txt}"
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

                        best_event = run_data["best_epoch_event"]
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
        self, base_diag_specs: Optional[Sequence[Tuple[str, str, bool]]] = None
    ) -> None:
        base_diag_specs = base_diag_specs or [
            ("proj_constraint_alpha", "Constraint alpha", False),
            ("proj_constraint_strength", "Constraint strength", False),
            ("proj_temperature", "Projection temperature", False),
            ("proj_mode_intrinsic_soft", "Intrinsic-soft mode flag (1=true)", False),
        ]

        for dataset_key in self.dataset_keys:
            dataset_runs = self.run_data_by_dataset.get(dataset_key, [])
            if not dataset_runs:
                continue

            diag_specs = list(base_diag_specs)
            active_diag_specs = []
            for metric_key, title, is_percent in diag_specs:
                has_finite = False
                for run_data in dataset_runs:
                    _, values = get_metric_series(run_data["epoch_events"], metric_key)
                    if np.any(np.isfinite(values)):
                        has_finite = True
                        break
                if has_finite:
                    active_diag_specs.append((metric_key, title, is_percent))

            if not active_diag_specs:
                print(f"[{dataset_key}] No projection diagnostics found in selected runs.")
                continue

            fig, axes = plt.subplots(len(active_diag_specs), 1, figsize=(10, 4.3 * len(active_diag_specs)), sharex=True)
            if len(active_diag_specs) == 1:
                axes = [axes]

            for ax, (metric_key, title, is_percent) in zip(axes, active_diag_specs):
                for run_data in dataset_runs:
                    epochs, values = get_metric_series(run_data["epoch_events"], metric_key)
                    plot_values = values * 100.0 if is_percent else values
                    ax.plot(
                        epochs,
                        plot_values,
                        label=run_data["label"],
                        color=run_data["color"],
                        linewidth=2.0,
                    )

                    best_event = run_data["best_epoch_event"]
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

                ax.set_title(f"Validation {title}")
                ax.set_ylabel("Value (%)" if is_percent else "Value")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)

            axes[-1].set_xlabel("Epoch")
            plt.suptitle(f"{dataset_key}: Projection Diagnostics", y=1.01, fontsize=13)
            plt.tight_layout()
            plt.show()

    def plot_training_losses(self, aggregate_loss_keys: Optional[Sequence[str]] = None) -> None:
        aggregate_loss_keys = aggregate_loss_keys or ["total", "level_ce", "gk_loss"]

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

                        best_event = run_data["best_epoch_event"]
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
                for key in run_data.get("test_metrics", {}).keys()
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
            for metric_key, _ in metric_rows:
                metric_values = [float(run_data["test_metrics"].get(metric_key, np.nan)) for run_data in dataset_runs]
                values_by_metric[metric_key] = metric_values
                best_by_metric[metric_key] = _best_indices(metric_key, metric_values)

            header_labels = ["Metric"] + [run_data["label"] for run_data in dataset_runs]
            table_lines = [
                f"### Dataset: `{dataset_key}`",
                f"Baseline run: **{base['label']}**",
                "",
                "| " + " | ".join(header_labels) + " |",
                "|---|" + "|".join(["---:"] * (len(header_labels) - 1)) + "|",
            ]

            for metric_key, metric_name in metric_rows:
                metric_label = f"{metric_name} {_metric_goal_arrow(metric_key)}"
                row_cells = [metric_label]
                values = values_by_metric[metric_key]
                for run_idx, value in enumerate(values):
                    cell = _fmt_value(metric_key, value) if run_idx == 0 else _merged_comp_cell(metric_key, value, values[0])
                    if run_idx in best_by_metric[metric_key] and cell != "n/a":
                        cell = f"**{cell}**"
                    row_cells.append(cell)
                table_lines.append("| " + " | ".join(row_cells) + " |")

            show_markdown("\n".join(table_lines))
