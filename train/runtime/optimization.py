import inspect
import math
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from .common import section_to_dict

try:
    from timm.scheduler import create_scheduler_v2 as timm_create_scheduler_v2
except Exception:  # pragma: no cover
    timm_create_scheduler_v2 = None


def seed_everything(seed: int, deterministic: bool = True) -> None:
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


class HTCapsNetExponentialScheduler:
    """Official HT-CapsNet epoch-indexed exponential schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        start_epoch: int = 10,
        decay_rate: float = 0.95,
    ):
        self.optimizer = optimizer
        self.initial_lr = float(initial_lr)
        self.start_epoch = int(start_epoch)
        self.decay_rate = float(decay_rate)
        if self.initial_lr <= 0.0:
            raise ValueError("HT-CapsNet scheduler initial_lr must be > 0.")
        if self.start_epoch < 0:
            raise ValueError("HT-CapsNet scheduler start_epoch must be >= 0.")
        if not 0.0 < self.decay_rate <= 1.0:
            raise ValueError("HT-CapsNet scheduler decay_rate must be in (0, 1].")
        self.group_scales = [
            float(group.get("lr", self.initial_lr)) / self.initial_lr
            for group in self.optimizer.param_groups
        ]
        self.last_epoch = 0

    def _lr_at(self, epoch: int) -> float:
        exponent = max(int(epoch) - self.start_epoch, 0)
        return float(self.initial_lr * (self.decay_rate ** exponent))

    def step(self, epoch: Optional[float] = None, _metric: Optional[float] = None) -> None:
        if epoch is None:
            epoch_index = self.last_epoch + 1
        else:
            epoch_index = int(epoch)
        self.last_epoch = epoch_index
        scheduled_lr = self._lr_at(epoch_index)
        for group, scale in zip(self.optimizer.param_groups, self.group_scales):
            group["lr"] = float(scheduled_lr * scale)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "initial_lr": self.initial_lr,
            "start_epoch": self.start_epoch,
            "decay_rate": self.decay_rate,
            "group_scales": list(self.group_scales),
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not isinstance(state_dict, dict):
            return
        self.initial_lr = float(state_dict.get("initial_lr", self.initial_lr))
        self.start_epoch = int(state_dict.get("start_epoch", self.start_epoch))
        self.decay_rate = float(state_dict.get("decay_rate", self.decay_rate))
        saved_scales = state_dict.get("group_scales")
        if isinstance(saved_scales, (list, tuple)) and len(saved_scales) == len(self.optimizer.param_groups):
            self.group_scales = [float(value) for value in saved_scales]
        self.last_epoch = int(state_dict.get("last_epoch", 0))
        self.step(self.last_epoch)


def build_optimizer(cfg: Any, model: torch.nn.Module):
    """Build an optimizer from cfg.optim."""
    name = cfg.optim.name
    if not isinstance(name, str):
        raise ValueError("optim.name must be a string.")
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
        model_cfg = section_to_dict(getattr(cfg, "model", None))
        model_name = model_cfg.get("name", "")
        if not isinstance(model_name, str):
            raise ValueError("model.name must be a string.")
        if hasattr(model, "parameter_groups"):
            if model_name == "hrn":
                lr_scale = float(model_cfg.get("trunk_lr_scale", 0.1))
                param_groups = model.parameter_groups(base_lr=lr, trunk_lr_scale=lr_scale)
            elif model_name == "hiercos":
                lr_scale = float(model_cfg.get("backbone_lr_scale", 0.1))
                transform_lr_scale = float(model_cfg.get("transform_lr_scale", 1.0))
                param_groups = model.parameter_groups(
                    base_lr=lr,
                    backbone_lr_scale=lr_scale,
                    transform_lr_scale=transform_lr_scale,
                )
            else:
                param_groups = model.parameter_groups(base_lr=lr)
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
    """Build an LR scheduler from cfg.scheduler."""
    sched_cfg = section_to_dict(cfg.scheduler)
    name = sched_cfg.get("name", "none")
    if not isinstance(name, str):
        raise ValueError("scheduler.name must be a string.")
    if name == "none":
        return None
    if name == "hiercos_cosine":
        base_lr = float(sched_cfg.get("base_lr", cfg.optim.get("lr", 0.1)))
        return HierCosCosineScheduler(
            optimizer=optimizer,
            num_epochs=int(cfg.train.epochs),
            base_lr=base_lr,
        )
    if name == "ht_capsnet_exponential":
        return HTCapsNetExponentialScheduler(
            optimizer=optimizer,
            initial_lr=float(cfg.optim.get("lr", 0.001)),
            start_epoch=int(sched_cfg.get("start_epoch", 10)),
            decay_rate=float(sched_cfg.get("decay_rate", 0.95)),
        )
    model_cfg = section_to_dict(getattr(cfg, "model", None))
    model_name = model_cfg.get("name", "")
    if not isinstance(model_name, str):
        raise ValueError("model.name must be a string.")
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
        key: value
        for key, value in sched_cfg.items()
        if key not in {"name", "use_timm", "step_size", "gamma"}
    }

    allowed = set(inspect.signature(timm_create_scheduler_v2).parameters.keys())
    filtered = {key: value for key, value in extra.items() if key in allowed}
    scheduler, _ = timm_create_scheduler_v2(
        optimizer,
        sched=name,
        num_epochs=int(cfg.train.epochs),
        **filtered,
    )
    return scheduler
