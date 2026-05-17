import inspect
import math
import os
import random
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import torch

try:
    from timm.scheduler import create_scheduler_v2 as timm_create_scheduler_v2
except Exception:  # pragma: no cover
    timm_create_scheduler_v2 = None


BEST_SELECTION_MODES = ("topdown", "independent")
RESUME_STATE_VERSION = 1
_RESUME_ALLOWED_TRAIN_DIFF_KEYS = {"resume", "output_dir", "stop_epoch"}


def seed_everything(seed: int, deterministic: bool = True):
    """Seed Python/NumPy/PyTorch RNGs and configure deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # OpenCV RNG is used by H-CAST SEEDS segmentation when available.
    try:
        import cv2
        cv2.setRNGSeed(int(seed))
    except Exception:
        pass
    # Deterministic mode improves reproducibility, benchmark improves speed.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def _section_to_dict(section: Any) -> Dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "items"):
        return {k: v for k, v in section.items()}
    return {}


def _to_plain_data(value: Any) -> Any:
    """Convert nested mappings/sequences into plain Python containers."""
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _normalized_cfg_for_resume_compare(cfg_resolved: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize config payload and strip run-control keys allowed to differ."""
    normalized = _to_plain_data(cfg_resolved)
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

    checkpoint_cfg = _normalized_cfg_for_resume_compare(_section_to_dict(checkpoint_cfg_resolved))
    current_cfg = _normalized_cfg_for_resume_compare(_section_to_dict(current_cfg_resolved))

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


def _normalize_finetune_url(url: str) -> str:
    # Hugging Face "blob" URLs are HTML pages; convert them to direct file URLs.
    if "huggingface.co" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    return url


def _load_external_checkpoint(path_or_url: str):
    parsed = urlparse(path_or_url)
    is_url = parsed.scheme in {"http", "https"}
    source = _normalize_finetune_url(path_or_url) if is_url else path_or_url
    if is_url:
        return torch.hub.load_state_dict_from_url(source, map_location="cpu", check_hash=False), source
    return torch.load(source, map_location="cpu"), source


def load_trusted_checkpoint(path: Any, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a full local training checkpoint produced by this repository."""
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def _extract_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return dict(checkpoint["model"])
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return dict(checkpoint["state_dict"])

    if isinstance(checkpoint, dict) and checkpoint:
        sample = next(iter(checkpoint.values()))
        if torch.is_tensor(sample):
            return dict(checkpoint)

    raise ValueError("Unsupported checkpoint format. Expected keys `model` or `state_dict`.")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if not all(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def _interpolate_pos_embed_if_needed(model: torch.nn.Module, checkpoint_model: Dict[str, torch.Tensor]):
    if "pos_embed" not in checkpoint_model:
        return
    if not hasattr(model, "pos_embed") or not hasattr(model, "patch_embed"):
        return

    ckpt_pos = checkpoint_model["pos_embed"]
    model_pos = getattr(model, "pos_embed")
    if not torch.is_tensor(ckpt_pos) or not torch.is_tensor(model_pos):
        return
    if ckpt_pos.shape == model_pos.shape:
        return

    try:
        num_patches = int(model.patch_embed.num_patches)
    except Exception:
        return

    num_extra_tokens = int(model_pos.shape[-2] - num_patches)
    if num_extra_tokens < 0:
        return

    orig_tokens = int(ckpt_pos.shape[-2] - num_extra_tokens)
    new_tokens = int(num_patches)
    if orig_tokens <= 0 or new_tokens <= 0:
        return

    orig_size = int(orig_tokens ** 0.5)
    new_size = int(new_tokens ** 0.5)
    if orig_size * orig_size != orig_tokens or new_size * new_size != new_tokens:
        return

    embedding_size = ckpt_pos.shape[-1]
    extra_tokens = ckpt_pos[:, :num_extra_tokens]
    pos_tokens = ckpt_pos[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)


def load_finetune_checkpoint(cfg: Any, model: torch.nn.Module) -> bool:
    """Load a finetune checkpoint using upstream H-CAST adaptation logic."""
    model_cfg = _section_to_dict(getattr(cfg, "model", None))
    train_cfg = _section_to_dict(getattr(cfg, "train", None))
    finetune_path = str(model_cfg.get("finetune", train_cfg.get("finetune", "")) or "").strip()
    if not finetune_path:
        return False

    model_name = str(model_cfg.get("name", "")).lower()
    if model_name == "hcast":
        inner = getattr(model, "model", None)
        if not isinstance(inner, torch.nn.Module):
            print("finetune: skipped (H-CAST timm backend unavailable)")
            return False
        target_model = inner
    else:
        target_model = model

    checkpoint, resolved_source = _load_external_checkpoint(finetune_path)
    checkpoint_model = _strip_module_prefix(_extract_checkpoint_state_dict(checkpoint))
    state_dict = target_model.state_dict()

    # Mirror upstream H-CAST key filtering for checkpoint adaptation.
    for key in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias", "cls_token"]:
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]

    _interpolate_pos_embed_if_needed(target_model, checkpoint_model)
    incompat = target_model.load_state_dict(checkpoint_model, strict=False)
    missing = len(getattr(incompat, "missing_keys", []))
    unexpected = len(getattr(incompat, "unexpected_keys", []))
    print(f"finetune: loaded from {resolved_source} (missing={missing}, unexpected={unexpected})")
    return True


def build_optimizer(cfg: Any, model: torch.nn.Module):
    """Build an optimizer from cfg.optim."""
    name = str(cfg.optim.name).lower()
    lr = float(cfg.optim.lr)
    wd = float(cfg.optim.get("weight_decay", 0.0))
    momentum = float(cfg.optim.get("momentum", 0.0))
    nesterov = bool(cfg.optim.get("nesterov", False))
    raw_opt_eps = cfg.optim.get("opt_eps", None)
    opt_eps = None if raw_opt_eps is None else float(raw_opt_eps)

    raw_opt_betas = cfg.optim.get("opt_betas", None)
    opt_betas = None
    if raw_opt_betas is not None:
        if isinstance(raw_opt_betas, str):
            raise ValueError("optim.opt_betas must be null or a list/tuple with two floats.")
        try:
            betas = list(raw_opt_betas)
        except TypeError as exc:
            raise ValueError("optim.opt_betas must be null or a list/tuple with two floats.") from exc
        if len(betas) != 2:
            raise ValueError("optim.opt_betas must be null or a list/tuple with two floats.")
        opt_betas = (float(betas[0]), float(betas[1]))

    if name == "adam":
        kwargs = {"lr": lr, "weight_decay": wd}
        if opt_eps is not None:
            kwargs["eps"] = opt_eps
        if opt_betas is not None:
            kwargs["betas"] = opt_betas
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        kwargs = {"lr": lr, "weight_decay": wd}
        if opt_eps is not None:
            kwargs["eps"] = opt_eps
        if opt_betas is not None:
            kwargs["betas"] = opt_betas
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "sgd":
        model_cfg = _section_to_dict(getattr(cfg, "model", None))
        model_name = str(model_cfg.get("name", "")).strip().lower()
        if model_name in {"hiercos", "hrn"} and hasattr(model, "parameter_groups"):
            if model_name == "hrn":
                lr_scale = float(model_cfg.get("trunk_lr_scale", 0.1))
                param_groups = model.parameter_groups(base_lr=lr, trunk_lr_scale=lr_scale)
            else:
                lr_scale = float(model_cfg.get("backbone_lr_scale", 0.1))
                param_groups = model.parameter_groups(base_lr=lr, backbone_lr_scale=lr_scale)
            if not param_groups:
                raise ValueError(f"{model_name} optimizer parameter_groups() returned no trainable parameters.")
            return torch.optim.SGD(
                param_groups,
                lr=lr,
                weight_decay=wd,
                momentum=momentum,
                nesterov=nesterov,
            )
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=momentum,
            nesterov=nesterov,
        )

    raise ValueError(f"Unsupported optimizer '{name}'")


def build_scheduler(cfg: Any, optimizer: torch.optim.Optimizer):
    """Build an LR scheduler from cfg.scheduler"""
    sched_cfg = _section_to_dict(cfg.scheduler)
    name = str(sched_cfg.get("name", "none")).lower()
    if name == "none":
        return None
    if name == "hiercos_cosine":
        base_lr = float(sched_cfg.get("base_lr", cfg.optim.get("lr", 0.1)))
        return HierCosCosineScheduler(
            optimizer=optimizer,
            num_epochs=int(cfg.train.epochs),
            base_lr=base_lr,
        )
    model_cfg = _section_to_dict(getattr(cfg, "model", None))
    model_name = str(model_cfg.get("name", "")).strip().lower()
    if model_name == "hrn" and name == "cosine":
        base_lr = float(sched_cfg.get("base_lr", cfg.optim.get("lr", 0.002)))
        return HierCosCosineScheduler(
            optimizer=optimizer,
            num_epochs=int(cfg.train.epochs),
            base_lr=base_lr,
        )

    if timm_create_scheduler_v2 is None:
        raise RuntimeError("timm scheduler is required but timm.scheduler.create_scheduler_v2 is unavailable")

    extra = {
        k: v
        for k, v in sched_cfg.items()
        if k not in {"name", "use_timm", "step_size", "gamma"}
    }
    normalized_extra = {str(k).replace("-", "_"): v for k, v in extra.items()}

    allowed = set(inspect.signature(timm_create_scheduler_v2).parameters.keys())
    filtered = {k: v for k, v in normalized_extra.items() if k in allowed}
    scheduler, _ = timm_create_scheduler_v2(
        optimizer,
        sched=name,
        num_epochs=int(cfg.train.epochs),
        **filtered,
    )
    return scheduler


class HierCosCosineScheduler:
    """Upstream cosine schedule with per-group LR scaling."""

    def __init__(self, optimizer: torch.optim.Optimizer, num_epochs: int, base_lr: float = 0.1):
        self.optimizer = optimizer
        self.num_epochs = max(int(num_epochs), 1)
        self.base_lr = float(base_lr)
        self.group_scales = self._infer_group_scales()
        self.last_t = 0.0

    def _infer_group_scales(self):
        if abs(self.base_lr) <= 1e-12:
            return [1.0 for _ in self.optimizer.param_groups]
        return [float(group.get("lr", self.base_lr)) / self.base_lr for group in self.optimizer.param_groups]

    def _lr_at(self, t: float) -> float:
        return float((self.base_lr / 2.0) * (math.cos(math.pi * (float(t) / float(self.num_epochs))) + 1.0))

    def step(self, epoch: Optional[float] = None, _metric: Optional[float] = None):
        if epoch is None:
            t = float(self.last_t + 1.0)
        else:
            t = float(epoch)
        self.last_t = t
        scheduled_lr = self._lr_at(t)
        for group, scale in zip(self.optimizer.param_groups, self.group_scales):
            group["lr"] = float(scheduled_lr * scale)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "num_epochs": int(self.num_epochs),
            "base_lr": float(self.base_lr),
            "group_scales": [float(scale) for scale in self.group_scales],
            "last_t": float(self.last_t),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return
        self.num_epochs = max(int(state_dict.get("num_epochs", self.num_epochs)), 1)
        self.base_lr = float(state_dict.get("base_lr", self.base_lr))

        saved_scales = state_dict.get("group_scales", None)
        if isinstance(saved_scales, (list, tuple)) and len(saved_scales) == len(self.optimizer.param_groups):
            self.group_scales = [float(scale) for scale in saved_scales]
        else:
            self.group_scales = self._infer_group_scales()

        self.last_t = float(state_dict.get("last_t", 0.0))
        self.step(self.last_t)


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
):
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
) -> Tuple[int, Dict[str, float], Dict[str, Any]]:
    """Load checkpoint state if present and return (start_epoch, best_metrics, resume_info)."""
    resume_info: Dict[str, Any] = {
        "resumed": False,
        "resume_path": str(resume_path or ""),
        "checkpoint_found": False,
        "start_epoch": 0,
        "config_check_passed": False,
        "rng_state_restored": False,
        "loader_rng_state_restored": False,
        "full_reproducibility_restored": False,
        "resume_state_version": None,
        "warnings": [],
    }

    if not resume_path:
        return 0, initial_best_metrics(), resume_info

    if not os.path.exists(resume_path):
        _record_resume_warning(
            f"Resume checkpoint not found at '{resume_path}'. Starting from scratch.",
            resume_info["warnings"],
        )
        return 0, initial_best_metrics(), resume_info

    ckpt = load_trusted_checkpoint(resume_path, map_location="cpu")
    resume_info["checkpoint_found"] = True
    resume_info["resumed"] = True
    resume_info["resume_path"] = str(resume_path)

    resume_meta = ckpt.get("resume_reproducibility", None)
    if isinstance(resume_meta, Mapping):
        resume_info["resume_state_version"] = int(resume_meta.get("version", RESUME_STATE_VERSION))

    _validate_resume_config_or_raise(
        checkpoint_cfg_resolved=ckpt.get("cfg_resolved", None),
        current_cfg_resolved=cfg_resolved,
        resume_path=str(resume_path),
    )
    resume_info["config_check_passed"] = True

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
            resume_info["warnings"],
        )
    else:
        resume_info["rng_state_restored"] = _restore_global_rng_state(ckpt.get("rng_state"), resume_info["warnings"])

    if ckpt.get("loader_generator_states", None) is None:
        _record_resume_warning(
            "Checkpoint has no loader generator snapshot (`loader_generator_states`); deterministic replay is not guaranteed.",
            resume_info["warnings"],
        )
    else:
        resume_info["loader_rng_state_restored"] = _restore_loader_generator_states(
            loader_states=ckpt.get("loader_generator_states"),
            loaders=loaders,
            warning_log=resume_info["warnings"],
        )

    # Resume from the next epoch after the one saved in the checkpoint.
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    resume_info["start_epoch"] = start_epoch
    resume_info["full_reproducibility_restored"] = bool(
        resume_info["config_check_passed"]
        and resume_info["rng_state_restored"]
        and resume_info["loader_rng_state_restored"]
    )
    return start_epoch, best_metrics, resume_info


def metric_for_best(eval_metrics: Mapping[str, float], mode: str) -> float:
    """Select the checkpoint ranking score from validation metrics.

    Lexicographic order:
    1) FPA for the selected mode (higher is better)
    2) TICE for the selected mode (lower is better)
    3) wAP for the selected mode (higher is better)
    Falls back to deepest available level accuracy when H-CAST metrics are absent.
    """
    if mode not in BEST_SELECTION_MODES:
        raise ValueError(f"Unknown selection mode '{mode}'. Expected one of {BEST_SELECTION_MODES}.")

    fpa_key = f"fpa_{mode}"
    tice_key = f"tice_{mode}"
    wap_key = f"weighted_ap_{mode}"
    has_fpa = fpa_key in eval_metrics
    has_tice = tice_key in eval_metrics
    has_wap = wap_key in eval_metrics

    if has_fpa or has_tice or has_wap:
        fpa = float(eval_metrics.get(fpa_key, 0.0))
        # TICE is inconsistency rate, so lower is better: encode as -TICE.
        neg_tice = -float(eval_metrics.get(tice_key, 1.0))
        wap = float(eval_metrics.get(wap_key, 0.0))
        # Base-10 lexicographic packing for bounded metrics in [0, 1].
        return float(fpa + 1e-3 * neg_tice + 1e-6 * wap)

    prefix = f"acc_level_{mode}_"
    deepest = [
        k
        for k in eval_metrics
        if k.startswith(prefix) and k[len(prefix) :].isdigit()
    ]
    if not deepest:
        return float(eval_metrics.get(fpa_key, 0.0))
    deepest_key = max(deepest, key=lambda key: int(key.rsplit("_", 1)[-1]))
    primary = float(eval_metrics.get(deepest_key, 0.0))
    tie = float(eval_metrics.get(fpa_key, 0.0))
    # Tiny path-accuracy term stabilizes ordering when primary scores are tied.
    return float(primary + 1e-3 * tie)
