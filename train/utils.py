import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_optimizer(cfg: Any, model: torch.nn.Module):
    name = str(cfg.optim.name).lower()
    lr = float(cfg.optim.lr)
    wd = float(cfg.optim.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        momentum = float(cfg.optim.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

    raise ValueError(f"Unsupported optimizer '{name}'")


def build_scheduler(cfg: Any, optimizer: torch.optim.Optimizer):
    name = str(cfg.scheduler.get("name", "none")).lower()
    if name == "none":
        return None
    if name == "cosine":
        t_max = int(cfg.train.epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    if name == "step":
        step_size = int(cfg.scheduler.get("step_size", 10))
        gamma = float(cfg.scheduler.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    raise ValueError(f"Unsupported scheduler '{name}'")


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

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = float(ckpt.get("best_metric", float("-inf")))
    return start_epoch, best_metric


def metric_for_best(eval_metrics: Dict[str, float]) -> float:
    deepest = [k for k in eval_metrics if k.startswith("acc_level_")]
    if not deepest:
        return eval_metrics.get("acc_path", 0.0)
    deepest_idx = max(int(k.split("_")[-1]) for k in deepest)
    primary = eval_metrics.get(f"acc_level_{deepest_idx}", 0.0)
    tie = eval_metrics.get("acc_path", 0.0)
    return float(primary + 1e-3 * tie)
