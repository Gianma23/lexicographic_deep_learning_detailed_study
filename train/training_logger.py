import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

import yaml


LOSS_KEYS_HINT_FIELD = "__loss_keys__"


def _as_float_dict(metrics: Mapping[str, Any]) -> Dict[str, float]:
    converted: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            converted[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return converted


def _required_loss_keys(train_outputs: Mapping[str, Any]) -> Set[str]:
    if LOSS_KEYS_HINT_FIELD not in train_outputs:
        raise KeyError(f"Missing required '{LOSS_KEYS_HINT_FIELD}' in train outputs.")

    raw_keys = train_outputs[LOSS_KEYS_HINT_FIELD]
    if not isinstance(raw_keys, (list, tuple, set, frozenset)):
        raise TypeError(
            f"'{LOSS_KEYS_HINT_FIELD}' must be a list/tuple/set of metric keys, got {type(raw_keys).__name__}."
        )
    return {str(key) for key in raw_keys}


def _split_train_outputs(train_outputs: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    hinted_loss_keys = _required_loss_keys(train_outputs)
    train_outputs_float = _as_float_dict(train_outputs)

    losses: Dict[str, float] = {}
    metrics: Dict[str, float] = {}
    for key, value in train_outputs_float.items():
        if key in hinted_loss_keys:
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
        self.level_names = [str(name) for name in (level_names or [])]
        self.model_name = str(model_name)

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
        return event

    def log_test(self, best_checkpoint: str, best_metric: float, test_metrics: Mapping[str, Any]) -> Dict[str, Any]:
        payload = {
            "best_checkpoint": str(best_checkpoint),
            "best_metric": float(best_metric),
            "test_metrics": _as_float_dict(test_metrics),
        }
        self._save_yaml(self.test_metrics_path, payload)
        event = {"event": "test", **payload}
        self._append_event(event)
        print(f"[LOGGER] saved test metrics: {self.test_metrics_path}")
        return payload

    def _append_event(self, payload: Mapping[str, Any]):
        self.run_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.run_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(payload), sort_keys=True) + "\n")

    def _save_yaml(self, path: Path, payload: Dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
