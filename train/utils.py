import inspect
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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

    Prefers weighted hierarchy accuracy; breaks ties with FPA.
    Falls back to deepest available level accuracy.
    """
    if "weighted_ap" in eval_metrics or "weighted_acc" in eval_metrics:
        primary = float(eval_metrics.get("weighted_ap", eval_metrics.get("weighted_acc", 0.0)))
        tie = float(eval_metrics.get("fpa", eval_metrics.get("acc_path", 0.0)))
        return float(primary + 1e-3 * tie)

    deepest = [k for k in eval_metrics if k.startswith("acc_level_")]
    if not deepest:
        return float(eval_metrics.get("fpa", eval_metrics.get("acc_path", 0.0)))
    deepest_idx = max(int(k.split("_")[-1]) for k in deepest)
    primary = eval_metrics.get(f"acc_level_{deepest_idx}", 0.0)
    tie = eval_metrics.get("fpa", eval_metrics.get("acc_path", 0.0))
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


def save_val_level_accuracies_plot(
    out_dir: str,
    epoch_ids: List[int],
    level_acc_history: Dict[int, List[float]],
    level_names: Optional[List[str]] = None,
    model_name: str = "Model",
) -> Optional[str]:
    """Save validation accuracy curves for each hierarchy level into output directory."""
    if not epoch_ids or not level_acc_history:
        return None

    has_finite = False
    for series in level_acc_history.values():
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
