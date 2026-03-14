import inspect
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from timm.optim import create_optimizer_v2 as timm_create_optimizer_v2
except Exception:  # pragma: no cover
    timm_create_optimizer_v2 = None

try:
    from timm.scheduler import create_scheduler_v2 as timm_create_scheduler_v2
except Exception:  # pragma: no cover
    timm_create_scheduler_v2 = None


def seed_everything(seed: int, deterministic: bool = True):
    """Seed Python/NumPy/PyTorch RNGs and configure cuDNN determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic mode improves reproducibility, benchmark improves speed.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _section_to_dict(section: Any) -> Dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "items"):
        return {k: v for k, v in section.items()}
    return {}


def build_optimizer(cfg: Any, model: torch.nn.Module):
    """Build an optimizer from cfg.optim."""
    name = str(cfg.optim.name).lower()
    lr = float(cfg.optim.lr)
    wd = float(cfg.optim.get("weight_decay", 0.0))
    momentum = float(cfg.optim.get("momentum", 0.0))
    
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

    raise ValueError(f"Unsupported optimizer '{name}'")


def build_scheduler(cfg: Any, optimizer: torch.optim.Optimizer):
    """Build an LR scheduler from cfg.scheduler; returns None if disabled."""
    sched_cfg = _section_to_dict(cfg.scheduler)
    name = str(sched_cfg.get("name", "none")).lower()
    if name == "none":
        return None

    use_timm = bool(sched_cfg.get("use_timm", True))
    if use_timm and timm_create_scheduler_v2 is not None:
        extra = {
            k: v
            for k, v in sched_cfg.items()
            if k not in {"name", "use_timm", "step_size", "gamma"}
        }
        # Backward-compatible aliases from the existing config schema.
        if "step_size" in sched_cfg and "decay_epochs" not in extra:
            extra["decay_epochs"] = int(sched_cfg["step_size"])
        if "gamma" in sched_cfg and "decay_rate" not in extra:
            extra["decay_rate"] = float(sched_cfg["gamma"])

        allowed = set(inspect.signature(timm_create_scheduler_v2).parameters.keys())
        filtered = {k: v for k, v in extra.items() if k in allowed}
        scheduler, _ = timm_create_scheduler_v2(
            optimizer,
            sched=name,
            num_epochs=int(cfg.train.epochs),
            **filtered,
        )
        return scheduler

    if name == "cosine":
        t_max = int(cfg.train.epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    if name == "step":
        step_size = int(sched_cfg.get("step_size", 10))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unsupported scheduler '{name}'")


def step_scheduler(scheduler, epoch: int, metric: Optional[float] = None):
    """Step either timm or torch schedulers with a unified call site."""
    if scheduler is None:
        return

    if metric is not None:
        try:
            scheduler.step(epoch + 1, metric)
            return
        except TypeError:
            pass

    if scheduler.__class__.__name__.lower() == "reducelronplateau" and metric is not None:
        scheduler.step(metric)
        return

    try:
        scheduler.step(epoch + 1)
    except TypeError:
        scheduler.step()


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    cfg_resolved: Dict[str, Any],
):
    """Persist training state so runs can be resumed exactly."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "cfg_resolved": cfg_resolved,
    }
    torch.save(payload, path)


def resume_if_available(
    resume_path: Optional[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
) -> Tuple[int, float]:
    """Load checkpoint state if present and return (start_epoch, best_metric)."""
    if not resume_path:
        return 0, float("-inf")

    if not os.path.exists(resume_path):
        return 0, float("-inf")

    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    # Resume from the next epoch after the one saved in the checkpoint.
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = float(ckpt.get("best_metric", float("-inf")))
    return start_epoch, best_metric


def metric_for_best(eval_metrics: Dict[str, float]) -> float:
    """Select the checkpoint ranking score from validation metrics.

    Prefers the deepest available level accuracy; breaks ties with acc_path.
    """
    deepest = [k for k in eval_metrics if k.startswith("acc_level_")]
    if not deepest:
        return eval_metrics.get("acc_path", 0.0)
    deepest_idx = max(int(k.split("_")[-1]) for k in deepest)
    primary = eval_metrics.get(f"acc_level_{deepest_idx}", 0.0)
    tie = eval_metrics.get("acc_path", 0.0)
    # Tiny path-accuracy term stabilizes ordering when primary scores are tied.
    return float(primary + 1e-3 * tie)


def save_train_level_losses_plot(
    out_dir: str,
    epoch_ids: List[int],
    level_loss_history: Dict[int, List[float]],
    level_names: Optional[List[str]] = None,
    model_name: str = "Model",
) -> Optional[str]:
    """Save train loss curves for each hierarchy level into output directory."""
    if not epoch_ids or not level_loss_history:
        return None

    has_finite = False
    for series in level_loss_history.values():
        for value in series:
            if np.isfinite(value):
                has_finite = True
                break
        if has_finite:
            break
    if not has_finite:
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
