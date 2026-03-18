import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import yaml


def save_yaml(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def as_float_dict(metrics: Mapping[str, Any]) -> Dict[str, float]:
    converted: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            converted[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return converted


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _append_jsonl(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_epoch_metrics_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return

    preferred_columns = ["epoch", "lr", "best_metric"]
    all_keys = {key for row in rows for key in row.keys()}
    dynamic_columns = sorted(key for key in all_keys if key not in preferred_columns)
    fieldnames = [key for key in preferred_columns if key in all_keys] + dynamic_columns

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def initialize_epoch_rows(start_epoch: int, jsonl_path: Path, csv_path: Path) -> List[Dict[str, Any]]:
    if start_epoch <= 0:
        for stale_path in (jsonl_path, csv_path):
            if stale_path.exists():
                stale_path.unlink()
        return []
    return _load_jsonl(jsonl_path)


def append_epoch_metrics(
    rows: List[Dict[str, Any]],
    jsonl_path: Path,
    csv_path: Path,
    epoch: int,
    lr: float,
    best_metric: float,
    train_metrics: Mapping[str, Any],
    val_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "epoch": int(epoch),
        "lr": float(lr),
        "best_metric": float(best_metric),
    }
    record.update({f"train_{k}": v for k, v in as_float_dict(train_metrics).items()})
    record.update({f"val_{k}": v for k, v in as_float_dict(val_metrics).items()})
    rows.append(record)
    _append_jsonl(jsonl_path, record)
    _write_epoch_metrics_csv(csv_path, rows)
    return record


def update_level_history(
    history: Dict[int, List[float]],
    epoch_ids: List[int],
    metrics: Mapping[str, Any],
    metric_prefix: str,
):
    observed: Dict[int, float] = {}
    for key, value in metrics.items():
        if not key.startswith(metric_prefix):
            continue
        suffix = key[len(metric_prefix) :]
        if suffix.isdigit():
            observed[int(suffix)] = float(value)

    for level_idx in observed:
        if level_idx not in history:
            history[level_idx] = [float("nan")] * (len(epoch_ids) - 1)

    for level_idx, series in history.items():
        series.append(observed.get(level_idx, float("nan")))


def _has_finite_values(history: Dict[int, List[float]]) -> bool:
    for series in history.values():
        for value in series:
            if np.isfinite(value):
                return True
    return False


def save_train_level_losses_plot(
    out_dir: str,
    epoch_ids: List[int],
    level_loss_history: Dict[int, List[float]],
    level_names: Optional[List[str]] = None,
    model_name: str = "Model",
) -> Optional[str]:
    """Save train loss curves for each hierarchy level into output directory."""
    if not epoch_ids or not level_loss_history or not _has_finite_values(level_loss_history):
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plot_path = Path(out_dir) / "train_losses_per_level.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    levels = sorted(level_loss_history.keys())
    names = level_names or []
    for level in levels:
        label_name = names[level] if level < len(names) else f"level_{level}"
        ax.plot(epoch_ids, level_loss_history[level], label=f"loss {label_name}", linewidth=1.6)

    ax.set_title(f"{str(model_name).upper()} Train Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return str(plot_path)


def save_val_level_accuracies_plot(
    out_dir: str,
    epoch_ids: List[int],
    level_acc_history: Dict[int, List[float]],
    level_names: Optional[List[str]] = None,
    model_name: str = "Model",
) -> Optional[str]:
    """Save validation accuracy curves for each hierarchy level into output directory."""
    if not epoch_ids or not level_acc_history or not _has_finite_values(level_acc_history):
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plot_path = Path(out_dir) / "val_accuracies_per_level.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    levels = sorted(level_acc_history.keys())
    names = level_names or []
    for level in levels:
        label_name = names[level] if level < len(names) else f"level_{level}"
        percent_series = [100.0 * value if np.isfinite(value) else value for value in level_acc_history[level]]
        ax.plot(epoch_ids, percent_series, label=f"val acc {label_name}", linewidth=1.6)

    ax.set_title(f"{str(model_name).upper()} Validation Accuracies")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return str(plot_path)

