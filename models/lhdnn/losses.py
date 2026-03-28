from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


LHDNNTargets = Union[torch.Tensor, Dict[str, Any]]


def _soft_targets_from_input(
    logits_per_level: List[torch.Tensor],
    targets: LHDNNTargets,
) -> Optional[List[torch.Tensor]]:
    if not isinstance(targets, dict):
        return None

    soft_targets = targets.get("soft_targets_per_level")
    if not isinstance(soft_targets, (list, tuple)) or len(soft_targets) != len(logits_per_level):
        return None

    out: List[torch.Tensor] = []
    for level, (logits_level, target_level) in enumerate(zip(logits_per_level, soft_targets)):
        if not isinstance(target_level, torch.Tensor):
            return None
        if target_level.ndim != 2 or int(target_level.size(1)) != int(logits_level.size(-1)):
            return None
        probs = target_level.to(device=logits_level.device, dtype=logits_level.dtype)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        out.append(probs)
    return out


def _hard_targets_from_input(
    targets: LHDNNTargets,
    num_levels: int,
) -> Optional[torch.Tensor]:
    if isinstance(targets, torch.Tensor):
        hard_targets = targets
    elif isinstance(targets, dict):
        hard_targets = targets.get("hard_targets")
        if hard_targets is None:
            hard_targets = targets.get("labels_a")
    else:
        return None

    if not isinstance(hard_targets, torch.Tensor):
        return None
    if hard_targets.ndim != 2:
        return None
    if int(hard_targets.size(1)) != int(num_levels):
        return None
    return hard_targets.long()


def _soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    probs = target_probs.to(device=logits.device, dtype=logits.dtype)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


def compute_loss(
    output: Dict[str, Any],
    targets: LHDNNTargets,
    cfg: Any,
    _taxonomy: Optional[Dict[str, Any]] = None,
    return_aux: bool = False,
) -> Union[
    Tuple[torch.Tensor, Dict[str, float]],
    Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]],
]:
    logits_per_level = output["logits_per_level"]
    if not isinstance(logits_per_level, list) or not logits_per_level:
        raise ValueError("LH-DNN output must contain non-empty `logits_per_level`.")

    soft_targets_per_level = _soft_targets_from_input(logits_per_level, targets)
    hard_targets = _hard_targets_from_input(targets, num_levels=len(logits_per_level))

    if soft_targets_per_level is None and hard_targets is None:
        raise TypeError("Expected hard targets [B, L] or `soft_targets_per_level` list in targets dict.")

    if hard_targets is not None:
        hard_targets = hard_targets.to(device=logits_per_level[0].device, dtype=torch.long)

    label_smoothing = min(max(float(cfg.train.get("smoothing", 0.0)), 0.0), 1.0)

    level_losses: List[torch.Tensor] = []
    for level, logits in enumerate(logits_per_level):
        if soft_targets_per_level is not None:
            level_loss = _soft_cross_entropy(logits, soft_targets_per_level[level])
        else:
            level_loss = F.cross_entropy(
                logits,
                hard_targets[:, level],
                label_smoothing=label_smoothing,
            )
        level_losses.append(level_loss)

    total = torch.stack(level_losses).sum()

    metrics = {
        "total": float(total.detach().item()),
        "level_ce": float(total.detach().item()),
    }
    for level, level_loss in enumerate(level_losses):
        metrics[f"loss_level_{level}"] = float(level_loss.detach().item())
    if not return_aux:
        return total, metrics

    aux_payload: Dict[str, Any] = {
        "level_losses": level_losses,
        "aux_loss": torch.zeros((), device=total.device, dtype=total.dtype),
    }
    return total, metrics, aux_payload
