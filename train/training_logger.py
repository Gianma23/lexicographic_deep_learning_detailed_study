import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import yaml


LOSS_EXACT_KEYS = {"total", "level_ce", "gk_loss"}
LOSS_SUFFIXES = ("_loss", "_ce")


def _as_float_dict(metrics: Mapping[str, Any]) -> Dict[str, float]:
    converted: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            converted[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return converted


def _is_loss_key(key: str) -> bool:
    if key in LOSS_EXACT_KEYS:
        return True
    if key.startswith("loss_"):
        return True
    return key.endswith(LOSS_SUFFIXES)


def _split_train_outputs(train_outputs: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    losses: Dict[str, float] = {}
    metrics: Dict[str, float] = {}
    for key, value in _as_float_dict(train_outputs).items():
        if _is_loss_key(key):
            losses[key] = value
        else:
            metrics[key] = value
    return losses, metrics


class TrainingLogger:
    def __init__(
        self,
        output_dir: Union[str, Path],
        start_epoch: int = 0,
        level_names: Optional[List[str]] = None,
        model_name: str = "Model",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.output_dir / "config_resolved.yaml"
        self.run_log_path = self.output_dir / "run_log.jsonl"
        self.test_metrics_path = self.output_dir / "test_metrics.yaml"
        self.train_loss_plot_path = self.output_dir / "train_losses_per_level.png"
        self.val_acc_plot_path = self.output_dir / "val_accuracies_per_level.png"
        self.level_names = [str(name) for name in (level_names or [])]
        self.model_name = str(model_name)

        self.epoch_ids: List[int] = []
        self.level_loss_history: Dict[int, List[float]] = {}
        self.level_val_acc_history: Dict[int, List[float]] = {}

        if start_epoch <= 0 and self.run_log_path.exists():
            self.run_log_path.unlink()

    def save_resolved_config(self, cfg_resolved: Mapping[str, Any]) -> Path:
        self._save_yaml(self.config_path, dict(cfg_resolved))
        return self.config_path

    def log_epoch(
        self,
        epoch: int,
        lr: float,
        best_metric: float,
        train_outputs: Mapping[str, Any],
        val_metrics: Mapping[str, Any],
    ) -> Dict[str, Any]:
        train_losses, train_metrics = _split_train_outputs(train_outputs)
        val_metrics_float = _as_float_dict(val_metrics)
        event = {
            "event": "epoch",
            "epoch": int(epoch),
            "lr": float(lr),
            "best_metric": float(best_metric),
            "train_losses": train_losses,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics_float,
        }
        self._append_event(event)
        self.epoch_ids.append(int(epoch))
        self._update_level_history(
            history=self.level_loss_history,
            epoch_ids=self.epoch_ids,
            metrics=train_losses,
            metric_prefix="loss_level_",
        )
        val_acc_prefix = "acc_level_topdown_"
        if not any(key.startswith(val_acc_prefix) for key in val_metrics_float.keys()):
            val_acc_prefix = "acc_level_independent_"
        self._update_level_history(
            history=self.level_val_acc_history,
            epoch_ids=self.epoch_ids,
            metrics=val_metrics_float,
            metric_prefix=val_acc_prefix,
        )
        return event

    def log_test(self, best_checkpoint: str, best_metric: float, test_metrics: Mapping[str, Any]) -> Dict[str, Any]:
        payload = {
            "best_checkpoint": str(best_checkpoint),
            "best_metric": float(best_metric),
            "test_metrics": _as_float_dict(test_metrics),
        }
        self._save_yaml(self.test_metrics_path, payload)
        train_plot, val_plot = self._save_plots()

        event = {"event": "test", **payload, "plots": {"train_losses": train_plot, "val_accuracies": val_plot}}
        self._append_event(event)
        print(f"[LOGGER] saved test metrics: {self.test_metrics_path}")
        if train_plot:
            print(f"saved_train_loss_plot: {train_plot}")
        elif self.level_loss_history:
            print("saved_train_loss_plot: skipped (matplotlib not installed)")
        if val_plot:
            print(f"saved_val_accuracy_plot: {val_plot}")
        elif self.level_val_acc_history:
            print("saved_val_accuracy_plot: skipped (matplotlib not installed)")
        return {**payload, "plots": event["plots"]}

    def _append_event(self, payload: Mapping[str, Any]):
        self.run_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.run_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(payload), sort_keys=True) + "\n")

    def _save_yaml(self, path: Path, payload: Dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)

    @staticmethod
    def _update_level_history(
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

    @staticmethod
    def _has_finite_values(history: Dict[int, List[float]]) -> bool:
        for series in history.values():
            for value in series:
                if math.isfinite(value):
                    return True
        return False

    def _save_plots(self) -> Tuple[Optional[str], Optional[str]]:
        train_plot_path = self._save_train_level_losses_plot()
        val_plot_path = self._save_val_level_accuracies_plot()
        return train_plot_path, val_plot_path

    def _save_train_level_losses_plot(self) -> Optional[str]:
        if not self.epoch_ids or not self.level_loss_history or not self._has_finite_values(self.level_loss_history):
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        levels = sorted(self.level_loss_history.keys())
        for level in levels:
            level_name = self.level_names[level] if level < len(self.level_names) else f"level_{level}"
            ax.plot(self.epoch_ids, self.level_loss_history[level], label=f"loss {level_name}", linewidth=1.6)

        ax.set_title(f"{self.model_name.upper()} Train Losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.train_loss_plot_path, dpi=150)
        plt.close(fig)
        return str(self.train_loss_plot_path)

    def _save_val_level_accuracies_plot(self) -> Optional[str]:
        if not self.epoch_ids or not self.level_val_acc_history or not self._has_finite_values(self.level_val_acc_history):
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        levels = sorted(self.level_val_acc_history.keys())
        for level in levels:
            level_name = self.level_names[level] if level < len(self.level_names) else f"level_{level}"
            percent_series = [
                100.0 * value if math.isfinite(value) else value for value in self.level_val_acc_history[level]
            ]
            ax.plot(self.epoch_ids, percent_series, label=f"val acc {level_name}", linewidth=1.6)

        ax.set_title(f"{self.model_name.upper()} Validation Accuracies")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.val_acc_plot_path, dpi=150)
        plt.close(fig)
        return str(self.val_acc_plot_path)
