from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


HcastTargets = Union[torch.Tensor, Dict[str, Any]]


def _hard_criterion_from_cfg(cfg: Any) -> torch.nn.Module:
    smoothing = min(max(float(cfg.train.get("smoothing", 0.0)), 0.0), 1.0)
    if smoothing > 0.0:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    return torch.nn.CrossEntropyLoss()


def _soft_targets_from_input(
    logits_per_level: List[torch.Tensor],
    targets: HcastTargets,
) -> Optional[List[torch.Tensor]]:
    if not isinstance(targets, dict):
        return None
    soft_targets = targets.get("soft_targets_per_level")
    if not isinstance(soft_targets, (list, tuple)) or len(soft_targets) != len(logits_per_level):
        return None

    out: List[torch.Tensor] = []
    for level, (logits, target_level) in enumerate(zip(logits_per_level, soft_targets)):
        if not isinstance(target_level, torch.Tensor):
            return None
        if target_level.ndim != 2 or int(target_level.size(1)) != int(logits.size(-1)):
            return None
        probs = target_level.to(device=logits.device, dtype=logits.dtype)
        out.append(probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return out


def _global_kl_loss(
    logits_per_level: List[torch.Tensor],
    hard_targets: Optional[torch.Tensor] = None,
    target_probs_per_level: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    all_outputs = torch.cat(logits_per_level, dim=1)
    all_outputs = F.log_softmax(all_outputs, dim=1)

    if target_probs_per_level is not None:
        all_targets = torch.cat(
            [target_probs.to(dtype=all_outputs.dtype) for target_probs in target_probs_per_level], dim=1
        )
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
    _taxonomy: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits_per_level = output["logits_per_level"]
    target_probs_per_level = _soft_targets_from_input(logits_per_level, targets)
    hard_targets: Optional[torch.Tensor] = None
    if target_probs_per_level is None:
        if not isinstance(targets, torch.Tensor):
            raise TypeError("Expected hard targets tensor of shape [B, L].")
        if targets.ndim != 2:
            raise ValueError(f"Expected hard targets with shape [B, L], got {tuple(targets.shape)}.")
        hard_targets = targets

    if target_probs_per_level is not None:
        soft_criterion = SoftTargetCrossEntropy()
        level_losses = [
            soft_criterion(logits, target_probs_per_level[level])
            for level, logits in enumerate(logits_per_level)
        ]
    else:
        hard_criterion = _hard_criterion_from_cfg(cfg)
        level_losses = [
            hard_criterion(logits, hard_targets[:, level])
            for level, logits in enumerate(logits_per_level)
        ]

    ce_loss = torch.stack(level_losses).sum()
    total = ce_loss

    if bool(cfg.loss.get("globalkl", False)):
        gk_w = float(cfg.loss.get("gk_weight", 1.0))
        gk_loss = _global_kl_loss(
            logits_per_level,
            hard_targets=hard_targets,
            target_probs_per_level=target_probs_per_level,
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
