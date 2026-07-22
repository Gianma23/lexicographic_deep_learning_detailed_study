import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

import yaml

from .runtime.selection import BEST_SELECTION_MODES

LOSS_KEYS_HINT_FIELD = "__loss_keys__"


def _as_float_dict(metrics: Mapping[str, Any]) -> Dict[str, float]:
    converted: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            converted[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return converted


def _as_selection_key_dict(keys: Mapping[str, Any]) -> Dict[str, List[float]]:
    converted: Dict[str, List[float]] = {}
    for mode, value in keys.items():
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        try:
            converted[str(mode)] = [float(component) for component in value]
        except (TypeError, ValueError):
            continue
    return converted


def _required_loss_keys(outputs: Mapping[str, Any]) -> Set[str]:
    if LOSS_KEYS_HINT_FIELD not in outputs:
        raise KeyError(f"Missing required '{LOSS_KEYS_HINT_FIELD}' in outputs.")

    raw_keys = outputs[LOSS_KEYS_HINT_FIELD]
    if not isinstance(raw_keys, (list, tuple, set, frozenset)):
        raise TypeError(
            f"'{LOSS_KEYS_HINT_FIELD}' must be a list/tuple/set of metric keys, got {type(raw_keys).__name__}."
        )
    return {str(key) for key in raw_keys}


def _split_outputs(outputs: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    hinted_loss_keys = _required_loss_keys(outputs)
    outputs_float = _as_float_dict(outputs)

    losses: Dict[str, float] = {}
    metrics: Dict[str, float] = {}
    for key, value in outputs_float.items():
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
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.output_dir / "config_resolved.yaml"
        self.run_log_path = self.output_dir / "run_log.jsonl"
        self.test_metrics_path = self.output_dir / "test_metrics.yaml"
        self.level_names = [str(name) for name in (level_names or [])]

        if start_epoch <= 0 and self.run_log_path.exists():
            self.run_log_path.unlink()

    def save_resolved_config(self, cfg_resolved: Mapping[str, Any]) -> Path:
        self._save_yaml(self.config_path, dict(cfg_resolved))
        return self.config_path

    def log_epoch(
        self,
        epoch: int,
        lr: float,
        best_metrics: Mapping[str, Any],
        best_selection_keys: Mapping[str, Any],
        candidate_selection: Mapping[str, Any],
        train_outputs: Mapping[str, Any],
        val_outputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        train_losses, train_metrics = _split_outputs(train_outputs)
        val_losses, val_metrics = _split_outputs(val_outputs)
        best_metrics_float = _as_float_dict(best_metrics)
        event = {
            "event": "epoch",
            "epoch": int(epoch),
            "lr": float(lr),
            "best_metrics": best_metrics_float,
            "best_selection_keys": _as_selection_key_dict(best_selection_keys),
            "candidate_selection": dict(candidate_selection),
            "train_losses": train_losses,
            "train_metrics": train_metrics,
            "val_losses": val_losses,
            "val_metrics": val_metrics,
        }
        self._append_event(event)
        return event

    def log_test(self, results_by_mode: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for mode in BEST_SELECTION_MODES:
            result = results_by_mode[mode]
            payload[mode] = {
                "best_checkpoint": str(result.get("best_checkpoint", "")),
                "best_epoch": int(result.get("best_epoch", -1)),
                "best_metric": float(result.get("best_metric", float("nan"))),
                "best_selection_key": [
                    float(component)
                    for component in result.get(
                        "best_selection_key",
                        [float("nan"), float("nan"), float("nan")],
                    )
                ],
                "test_metrics": _as_float_dict(result.get("test_metrics", {})),
            }

        self._save_yaml(self.test_metrics_path, payload)
        event = {"event": "test", "test_results": payload}
        self._append_event(event)
        print(f"[LOGGER] saved test metrics: {self.test_metrics_path}")
        return payload

    def log_resume(self, resume_info: Mapping[str, Any]) -> Dict[str, Any]:
        warnings_raw = resume_info.get("warnings", [])
        warnings_list = [str(item) for item in warnings_raw] if isinstance(warnings_raw, (list, tuple)) else []
        event = {
            "event": "resume",
            "resumed": bool(resume_info.get("resumed", False)),
            "resume_path": str(resume_info.get("resume_path", "")),
            "checkpoint_found": bool(resume_info.get("checkpoint_found", False)),
            "start_epoch": int(resume_info.get("start_epoch", 0)),
            "config_check_passed": bool(resume_info.get("config_check_passed", False)),
            "rng_state_restored": bool(resume_info.get("rng_state_restored", False)),
            "loader_rng_state_restored": bool(resume_info.get("loader_rng_state_restored", False)),
            "full_reproducibility_restored": bool(resume_info.get("full_reproducibility_restored", False)),
            "resume_state_version": resume_info.get("resume_state_version", None),
            "warnings": warnings_list,
        }
        self._append_event(event)
        return event

    def _append_event(self, payload: Mapping[str, Any]):
        self.run_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.run_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(payload), sort_keys=True) + "\n")

    def _save_yaml(self, path: Path, payload: Dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
