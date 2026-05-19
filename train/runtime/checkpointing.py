import os
import random
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from .common import section_to_dict, to_plain_data
from .selection import BEST_SELECTION_MODES

RESUME_STATE_VERSION = 1
_RESUME_ALLOWED_TRAIN_DIFF_KEYS = {"resume", "output_dir", "stop_epoch"}


@dataclass
class ResumeInfo:
    resumed: bool = False
    resume_path: str = ""
    checkpoint_found: bool = False
    start_epoch: int = 0
    config_check_passed: bool = False
    rng_state_restored: bool = False
    loader_rng_state_restored: bool = False
    full_reproducibility_restored: bool = False
    resume_state_version: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalized_cfg_for_resume_compare(cfg_resolved: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize config payload and strip run-control keys allowed to differ."""
    normalized = to_plain_data(cfg_resolved)
    if not isinstance(normalized, dict):
        raise ValueError("Resolved config must be a mapping for strict resume validation.")

    sanitized = deepcopy(normalized)
    train_cfg = sanitized.get("train")
    if isinstance(train_cfg, dict):
        for key in _RESUME_ALLOWED_TRAIN_DIFF_KEYS:
            train_cfg.pop(key, None)
    return sanitized


def _collect_value_differences(left: Any, right: Any, path: str, out: List[str]) -> None:
    location = path if path else "<root>"

    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left.keys()) | set(right.keys()))
        for key in keys:
            child_path = f"{path}.{key}" if path else str(key)
            if key not in left:
                out.append(f"{child_path}: missing in checkpoint cfg, current={right[key]!r}")
                continue
            if key not in right:
                out.append(f"{child_path}: checkpoint={left[key]!r}, missing in current cfg")
                continue
            _collect_value_differences(left[key], right[key], child_path, out)
        return

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append(f"{location}: list length checkpoint={len(left)} current={len(right)}")
            return
        for idx, (left_item, right_item) in enumerate(zip(left, right)):
            _collect_value_differences(left_item, right_item, f"{location}[{idx}]", out)
        return

    if left != right:
        out.append(f"{location}: checkpoint={left!r} current={right!r}")


def _validate_resume_config_or_raise(
    checkpoint_cfg_resolved: Any,
    current_cfg_resolved: Any,
    resume_path: str,
) -> None:
    if checkpoint_cfg_resolved is None:
        raise ValueError(
            "Strict resume check failed: checkpoint is missing `cfg_resolved`, "
            "cannot verify reproducible compatibility."
        )
    if current_cfg_resolved is None:
        raise ValueError(
            "Strict resume check failed: current run is missing `cfg_resolved`, "
            "cannot verify reproducible compatibility."
        )

    checkpoint_cfg = _normalized_cfg_for_resume_compare(section_to_dict(checkpoint_cfg_resolved))
    current_cfg = _normalized_cfg_for_resume_compare(section_to_dict(current_cfg_resolved))

    differences: List[str] = []
    _collect_value_differences(checkpoint_cfg, current_cfg, "", differences)
    if differences:
        max_items = 30
        excerpt = differences[:max_items]
        suffix = ""
        if len(differences) > max_items:
            suffix = f"\n- ... {len(differences) - max_items} more differences"
        raise ValueError(
            "Strict resume check failed: checkpoint config differs from current resolved config.\n"
            f"resume_path={resume_path}\n"
            "Allowed differences: train.resume, train.output_dir, train.stop_epoch\n"
            + "\n".join(f"- {item}" for item in excerpt)
            + suffix
        )


def _capture_global_rng_state() -> Dict[str, Any]:
    """Capture all global RNG states required for deterministic resume."""
    cuda_available = torch.cuda.is_available()
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if cuda_available else None,
        "cuda_available_at_save": bool(cuda_available),
    }


def _capture_loader_generator_states(loaders: Optional[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Capture DataLoader generator states keyed by split name."""
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(loaders, Mapping):
        return out

    for name, loader in loaders.items():
        key = str(name)
        generator = getattr(loader, "generator", None)
        if isinstance(generator, torch.Generator):
            out[key] = {
                "has_generator": True,
                "generator_state": generator.get_state().clone(),
            }
        else:
            out[key] = {
                "has_generator": False,
                "generator_state": None,
            }
    return out


def _record_resume_warning(message: str, warning_log: List[str]) -> None:
    warning_log.append(message)
    warnings.warn(message, RuntimeWarning)


def _restore_global_rng_state(rng_state: Any, warning_log: List[str]) -> bool:
    """Restore global RNG state; returns True only when fully restored."""
    if not isinstance(rng_state, Mapping):
        _record_resume_warning(
            "Resume checkpoint does not contain a valid `rng_state`; deterministic replay is not guaranteed.",
            warning_log,
        )
        return False

    restored = True

    python_state = rng_state.get("python_random_state", None)
    if python_state is None:
        _record_resume_warning("Missing Python RNG state in checkpoint.", warning_log)
        restored = False
    else:
        random.setstate(python_state)

    numpy_state = rng_state.get("numpy_random_state", None)
    if numpy_state is None:
        _record_resume_warning("Missing NumPy RNG state in checkpoint.", warning_log)
        restored = False
    else:
        np.random.set_state(numpy_state)

    torch_cpu_state = rng_state.get("torch_cpu_rng_state", None)
    if not isinstance(torch_cpu_state, torch.Tensor):
        _record_resume_warning("Missing/invalid Torch CPU RNG state in checkpoint.", warning_log)
        restored = False
    else:
        torch.random.set_rng_state(torch_cpu_state)

    cuda_state_all = rng_state.get("torch_cuda_rng_state_all", None)
    if cuda_state_all is not None:
        if not torch.cuda.is_available():
            _record_resume_warning(
                "Checkpoint contains CUDA RNG state, but CUDA is unavailable at resume time.",
                warning_log,
            )
            restored = False
        elif not isinstance(cuda_state_all, (list, tuple)):
            _record_resume_warning("Invalid CUDA RNG state payload in checkpoint.", warning_log)
            restored = False
        else:
            try:
                torch.cuda.set_rng_state_all(list(cuda_state_all))
            except Exception as exc:
                _record_resume_warning(f"Failed to restore CUDA RNG state: {exc}", warning_log)
                restored = False

    return restored


def _restore_loader_generator_states(
    loader_states: Any,
    loaders: Optional[Mapping[str, Any]],
    warning_log: List[str],
) -> bool:
    """Restore DataLoader generator states; returns True on full restoration."""
    if not isinstance(loaders, Mapping):
        _record_resume_warning(
            "Resume called without loader mapping; DataLoader generator state cannot be restored.",
            warning_log,
        )
        return False
    if not isinstance(loader_states, Mapping):
        _record_resume_warning(
            "Resume checkpoint does not contain loader generator states; deterministic replay is not guaranteed.",
            warning_log,
        )
        return False

    restored = True
    for name, loader in loaders.items():
        key = str(name)
        saved_state = loader_states.get(key, None)
        if not isinstance(saved_state, Mapping):
            _record_resume_warning(f"Missing loader RNG state for split '{key}' in checkpoint.", warning_log)
            restored = False
            continue

        if not bool(saved_state.get("has_generator", False)):
            continue

        generator = getattr(loader, "generator", None)
        if not isinstance(generator, torch.Generator):
            _record_resume_warning(
                f"Current loader '{key}' has no generator, cannot restore saved generator state.",
                warning_log,
            )
            restored = False
            continue

        generator_state = saved_state.get("generator_state", None)
        if not isinstance(generator_state, torch.Tensor):
            _record_resume_warning(f"Invalid generator state payload for loader '{key}'.", warning_log)
            restored = False
            continue

        try:
            generator.set_state(generator_state)
        except Exception as exc:
            _record_resume_warning(f"Failed to restore generator state for loader '{key}': {exc}", warning_log)
            restored = False

    return restored


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metrics: Mapping[str, float],
    cfg_resolved: Dict[str, Any],
    loaders: Optional[Mapping[str, Any]] = None,
) -> None:
    """Persist training state so runs can be resumed exactly."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rng_state = _capture_global_rng_state()
    loader_generator_states = _capture_loader_generator_states(loaders)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metrics": normalize_best_metrics(best_metrics),
        "cfg_resolved": cfg_resolved,
        "rng_state": rng_state,
        "loader_generator_states": loader_generator_states,
        "resume_reproducibility": {
            "version": RESUME_STATE_VERSION,
            "has_rng_state": True,
            "has_loader_generator_states": bool(loader_generator_states),
            "loader_keys": sorted(loader_generator_states.keys()),
        },
    }
    torch.save(payload, path)


def initial_best_metrics() -> Dict[str, float]:
    return {mode: float("-inf") for mode in BEST_SELECTION_MODES}


def normalize_best_metrics(best_metrics: Mapping[str, Any]) -> Dict[str, float]:
    return {mode: float(best_metrics[mode]) for mode in BEST_SELECTION_MODES}


def resume_if_available(
    resume_path: Optional[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    cfg_resolved: Optional[Mapping[str, Any]] = None,
    loaders: Optional[Mapping[str, Any]] = None,
    checkpoint_loader=None,
) -> Tuple[int, Dict[str, float], ResumeInfo]:
    """Load checkpoint state if present and return (start_epoch, best_metrics, resume_info)."""
    resume_info = ResumeInfo(resume_path=str(resume_path or ""))

    if checkpoint_loader is None:
        from .finetune import load_trusted_checkpoint as checkpoint_loader

    if not resume_path:
        return 0, initial_best_metrics(), resume_info

    if not os.path.exists(resume_path):
        _record_resume_warning(
            f"Resume checkpoint not found at '{resume_path}'. Starting from scratch.",
            resume_info.warnings,
        )
        return 0, initial_best_metrics(), resume_info

    ckpt = checkpoint_loader(resume_path, map_location="cpu")
    resume_info.checkpoint_found = True
    resume_info.resumed = True
    resume_info.resume_path = str(resume_path)

    resume_meta = ckpt.get("resume_reproducibility", None)
    if isinstance(resume_meta, Mapping):
        resume_info.resume_state_version = int(resume_meta.get("version", RESUME_STATE_VERSION))

    _validate_resume_config_or_raise(
        checkpoint_cfg_resolved=ckpt.get("cfg_resolved", None),
        current_cfg_resolved=cfg_resolved,
        resume_path=str(resume_path),
    )
    resume_info.config_check_passed = True

    best_metrics = normalize_best_metrics(ckpt["best_metrics"])

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    if ckpt.get("rng_state", None) is None:
        _record_resume_warning(
            "Checkpoint has no RNG snapshot (`rng_state`); deterministic replay is not guaranteed.",
            resume_info.warnings,
        )
    else:
        resume_info.rng_state_restored = _restore_global_rng_state(ckpt.get("rng_state"), resume_info.warnings)

    if ckpt.get("loader_generator_states", None) is None:
        _record_resume_warning(
            "Checkpoint has no loader generator snapshot (`loader_generator_states`); deterministic replay is not guaranteed.",
            resume_info.warnings,
        )
    else:
        resume_info.loader_rng_state_restored = _restore_loader_generator_states(
            loader_states=ckpt.get("loader_generator_states"),
            loaders=loaders,
            warning_log=resume_info.warnings,
        )

    # Resume from the next epoch after the one saved in the checkpoint.
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    resume_info.start_epoch = start_epoch
    resume_info.full_reproducibility_restored = bool(
        resume_info.config_check_passed
        and resume_info.rng_state_restored
        and resume_info.loader_rng_state_restored
    )
    return start_epoch, best_metrics, resume_info
