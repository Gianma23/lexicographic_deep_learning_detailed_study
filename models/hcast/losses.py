from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

try:
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
except Exception:  # pragma: no cover
    LabelSmoothingCrossEntropy = None
    SoftTargetCrossEntropy = None


HcastTargets = Union[torch.Tensor, Dict[str, Any]]


def _soft_target_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    target_probs = target_probs.to(dtype=logits.dtype)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def _hard_criterion_from_cfg(cfg: Any):
    smoothing = min(max(float(cfg.train.get("smoothing", 0.0)), 0.0), 1.0)
    if smoothing > 0.0 and LabelSmoothingCrossEntropy is not None:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    return torch.nn.CrossEntropyLoss()


def _soft_criterion_from_cfg():
    if SoftTargetCrossEntropy is not None:
        return SoftTargetCrossEntropy()
    return None


def _hard_targets_from_input(targets: HcastTargets) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        return targets
    labels_a = targets.get("labels_a")
    if not isinstance(labels_a, torch.Tensor):
        raise TypeError("Expected hard targets tensor or mixup target dict with `labels_a` tensor.")
    return labels_a


def _mixup_target_distributions(logits_per_level: List[torch.Tensor], targets: HcastTargets):
    if not isinstance(targets, dict):
        return None

    soft_targets = targets.get("soft_targets_per_level")
    if isinstance(soft_targets, (list, tuple)) and len(soft_targets) == len(logits_per_level):
        out: List[torch.Tensor] = []
        for level, logits in enumerate(logits_per_level):
            target_level = soft_targets[level]
            if not isinstance(target_level, torch.Tensor):
                return None
            if target_level.ndim != 2 or int(target_level.size(1)) != int(logits.size(-1)):
                return None
            out.append(target_level.to(device=logits.device, dtype=logits.dtype))
        return out
    return None


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
    mixup_target_probs = _mixup_target_distributions(logits_per_level, targets)
    hard_targets = _hard_targets_from_input(targets) if mixup_target_probs is None else None
    loss_cfg = cfg.get("loss", {}) if hasattr(cfg, "get") else cfg.loss
    if loss_cfg is None:
        loss_cfg = {}
    use_bce_loss = bool(loss_cfg.get("bce_loss", False))

    if mixup_target_probs is not None:
        if use_bce_loss:
            criterion = torch.nn.BCEWithLogitsLoss()
            level_losses = [
                criterion(logits, mixup_target_probs[level].to(dtype=logits.dtype))
                for level, logits in enumerate(logits_per_level)
            ]
        else:
            soft_criterion = _soft_criterion_from_cfg()
            if soft_criterion is not None:
                level_losses = [
                    soft_criterion(logits, mixup_target_probs[level].to(dtype=logits.dtype))
                    for level, logits in enumerate(logits_per_level)
                ]
            else:
                level_losses = [
                    _soft_target_cross_entropy(logits, mixup_target_probs[level])
                    for level, logits in enumerate(logits_per_level)
                ]
    else:
        if use_bce_loss:
            criterion = torch.nn.BCEWithLogitsLoss()
            level_losses = []
            for level, logits in enumerate(logits_per_level):
                target_level = F.one_hot(hard_targets[:, level], num_classes=logits.size(-1)).to(dtype=logits.dtype)
                level_losses.append(criterion(logits, target_level))
        else:
            hard_criterion = _hard_criterion_from_cfg(cfg)
            level_losses = [
                hard_criterion(logits, hard_targets[:, level])
                for level, logits in enumerate(logits_per_level)
            ]
    ce_loss = torch.stack(level_losses).sum()
    total = ce_loss

    if bool(loss_cfg.get("globalkl", False)):
        gk_w = float(loss_cfg.get("gk_weight", 1.0))
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
