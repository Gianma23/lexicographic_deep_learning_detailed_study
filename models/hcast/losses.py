from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


HcastTargets = Union[torch.Tensor, Dict[str, Any]]


def _one_hot_with_smoothing(target: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    smoothing = min(max(float(smoothing), 0.0), 1.0)
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    return torch.full((target.size(0), num_classes), off_value, device=target.device).scatter_(
        1, target.long().view(-1, 1), on_value
    )


def _soft_target_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    target_probs = target_probs.to(dtype=logits.dtype)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def _hard_nll_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    log_probs = torch.log(probs.clamp_min(float(eps)))
    return F.nll_loss(log_probs, target)


def _soft_target_nll_from_probs(probs: torch.Tensor, target_probs: torch.Tensor, eps: float) -> torch.Tensor:
    log_probs = torch.log(probs.clamp_min(float(eps)))
    target_probs = target_probs.to(dtype=log_probs.dtype)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def _hard_targets_from_input(targets: HcastTargets) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        return targets
    labels_a = targets.get("labels_a")
    if not isinstance(labels_a, torch.Tensor):
        raise TypeError("Expected hard targets tensor or mixup target dict with `labels_a` tensor.")
    return labels_a


def _mixup_target_distributions(logits_per_level: List[torch.Tensor], targets: HcastTargets, cfg: Any):
    if not isinstance(targets, dict):
        return None

    labels_a = targets.get("labels_a")
    labels_b = targets.get("labels_b")
    lam = targets.get("lam")
    if not isinstance(labels_a, torch.Tensor) or not isinstance(labels_b, torch.Tensor) or lam is None:
        return None

    lam = float(lam)
    default_smoothing = float(cfg.train.get("smoothing", 0.1))
    smoothing = float(targets.get("label_smoothing", default_smoothing))
    smoothing = min(max(smoothing, 0.0), 1.0)

    mixed_targets = []
    for level, logits in enumerate(logits_per_level):
        num_classes = logits.size(-1)
        y_a = _one_hot_with_smoothing(labels_a[:, level], num_classes, smoothing)
        y_b = _one_hot_with_smoothing(labels_b[:, level], num_classes, smoothing)
        mixed_targets.append(y_a * lam + y_b * (1.0 - lam))
    return mixed_targets


def _global_kl_loss(
    logits_per_level: List[torch.Tensor],
    hard_targets: Optional[torch.Tensor] = None,
    target_probs_per_level: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    # Match upstream H-CAST: concatenate raw logits first, then apply one global log_softmax.
    all_outputs = torch.cat(logits_per_level, dim=1)
    all_outputs = F.log_softmax(all_outputs, dim=1)

    if target_probs_per_level is not None:
        all_targets = torch.cat([target_probs.to(dtype=all_outputs.dtype) for target_probs in target_probs_per_level], dim=1)
    else:
        if hard_targets is None:
            raise ValueError("Either `hard_targets` or `target_probs_per_level` must be provided for global KL.")
        onehots = []
        for level, logits in enumerate(logits_per_level):
            onehots.append(F.one_hot(hard_targets[:, level], num_classes=logits.size(-1)).float())
        all_targets = torch.cat(onehots, dim=1)

    all_targets = F.normalize(all_targets, p=1, dim=1)
    return F.kl_div(all_outputs, all_targets, reduction="batchmean")


def compute_loss(
    output: Dict[str, Any],
    targets: HcastTargets,
    cfg: Any,
    taxonomy: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    _ = taxonomy
    logits_per_level = output["logits_per_level"]
    projected_probs_per_level = output.get("projected_probs_per_level")
    design1_active = bool(output.get("design1_active", False))
    hard_targets = _hard_targets_from_input(targets)
    mixup_target_probs = _mixup_target_distributions(logits_per_level, targets, cfg)
    label_smoothing = min(max(float(cfg.train.get("smoothing", 0.0)), 0.0), 1.0)
    design1_cfg = cfg.model.get("design1", {})
    design1_eps = float(design1_cfg.get("eps", 1e-12))
    if design1_eps <= 0.0:
        design1_eps = 1e-12
    projected_available = isinstance(projected_probs_per_level, list) and (
        len(projected_probs_per_level) == len(logits_per_level)
    )
    # Design1 training path: when projection is active and available, optimize on projected probabilities.
    use_projected_for_loss = projected_available and design1_active

    if mixup_target_probs is not None:
        if use_projected_for_loss:
            level_losses = [
                _soft_target_nll_from_probs(projected_probs_per_level[level], mixup_target_probs[level], eps=design1_eps)
                for level in range(len(projected_probs_per_level))
            ]
        else:
            level_losses = [
                _soft_target_cross_entropy(logits, mixup_target_probs[level])
                for level, logits in enumerate(logits_per_level)
            ]
    else:
        if use_projected_for_loss:
            level_losses = [
                _hard_nll_from_probs(projected_probs_per_level[level], hard_targets[:, level], eps=design1_eps)
                for level in range(len(projected_probs_per_level))
            ]
        else:
            level_losses = [
                F.cross_entropy(logits, hard_targets[:, level], label_smoothing=label_smoothing)
                for level, logits in enumerate(logits_per_level)
            ]
    ce_loss = torch.stack(level_losses).sum()
    total = ce_loss

    if bool(cfg.loss.get("globalkl", False)):
        gk_w = float(cfg.loss.get("gk_weight", 1.0))
        # Keep global KL unchanged: it is always computed from raw logits.
        gk_loss = _global_kl_loss(
            logits_per_level,
            hard_targets=hard_targets,
            target_probs_per_level=mixup_target_probs,
        )
        total = total + gk_w * gk_loss
    else:
        gk_loss = torch.zeros((), device=total.device)

    metrics = {
        "total": float(total.detach().item()),
        "level_ce": float(ce_loss.detach().item()),
        "gk_loss": float(gk_loss.detach().item()),
    }
    for level, level_loss in enumerate(level_losses):
        metrics[f"loss_level_{level}"] = float(level_loss.detach().item())
    return total, metrics
