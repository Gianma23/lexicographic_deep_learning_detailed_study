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
    outputs_per_level: List[torch.Tensor],
    targets: HcastTargets,
) -> Optional[List[torch.Tensor]]:
    if not isinstance(targets, dict):
        return None
    soft_targets = targets.get("soft_targets_per_level")
    if not isinstance(soft_targets, (list, tuple)) or len(soft_targets) != len(outputs_per_level):
        return None

    out: List[torch.Tensor] = []
    for level, (outputs, target_level) in enumerate(zip(outputs_per_level, soft_targets)):
        if not isinstance(target_level, torch.Tensor):
            return None
        if target_level.ndim != 2 or int(target_level.size(1)) != int(outputs.size(-1)):
            return None
        probs = target_level.to(device=outputs.device, dtype=outputs.dtype)
        out.append(probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return out


def _soft_cross_entropy_from_log_probs(log_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    probs = target_probs.to(device=log_probs.device, dtype=log_probs.dtype)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -(probs * log_probs).sum(dim=-1).mean()


def _label_smoothed_target_probs(
    target: torch.Tensor,
    num_classes: int,
    smoothing: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    off_value = smoothing / max(float(num_classes), 1.0)
    on_value = 1.0 - smoothing + off_value
    out = torch.full(
        (target.size(0), num_classes),
        fill_value=off_value,
        dtype=dtype,
        device=target.device,
    )
    return out.scatter_(1, target.unsqueeze(1), on_value)


def _global_kl_loss(
    outputs_per_level: List[torch.Tensor],
    hard_targets: Optional[torch.Tensor] = None,
    target_probs_per_level: Optional[List[torch.Tensor]] = None,
    outputs_are_log_probs: bool = False,
) -> torch.Tensor:
    all_outputs = torch.cat(outputs_per_level, dim=1)
    if not outputs_are_log_probs:
        all_outputs = F.log_softmax(all_outputs, dim=1)

    if target_probs_per_level is not None:
        all_targets = torch.cat(
            [target_probs.to(dtype=all_outputs.dtype) for target_probs in target_probs_per_level], dim=1
        )
    else:
        if hard_targets is None:
            raise ValueError("Either `hard_targets` or `target_probs_per_level` must be provided for global KL.")
        onehots = []
        for level, outputs in enumerate(outputs_per_level):
            onehots.append(F.one_hot(hard_targets[:, level], num_classes=outputs.size(-1)).float())
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
    effective_probs_per_level = output.get("effective_probs_per_level")
    has_effective_probs = (
        isinstance(effective_probs_per_level, list)
        and len(effective_probs_per_level) == len(logits_per_level)
    )
    if effective_probs_per_level is not None and not has_effective_probs:
        raise ValueError(
            "`effective_probs_per_level` must be either None or a list aligned with `logits_per_level`."
        )

    outputs_for_targets = effective_probs_per_level if has_effective_probs else logits_per_level

    target_probs_per_level = _soft_targets_from_input(outputs_for_targets, targets)
    hard_targets: Optional[torch.Tensor] = None
    if target_probs_per_level is None:
        if not isinstance(targets, torch.Tensor):
            raise TypeError("Expected hard targets tensor of shape [B, L].")
        if targets.ndim != 2:
            raise ValueError(f"Expected hard targets with shape [B, L], got {tuple(targets.shape)}.")
        hard_targets = targets.long()

    if has_effective_probs:
        raw_eps = cfg.model.get("design1", {}).get("eps", 1e-12)
        eps = float(raw_eps) if raw_eps is not None else 1e-12
        eps = eps if eps > 0.0 else 1e-12
        log_probs_per_level = [
            torch.log(probs.clamp_min(eps))
            for probs in effective_probs_per_level
        ]

        if target_probs_per_level is not None:
            level_losses = [
                _soft_cross_entropy_from_log_probs(log_probs, target_probs_per_level[level])
                for level, log_probs in enumerate(log_probs_per_level)
            ]
        else:
            smoothing = min(max(float(cfg.train.get("smoothing", 0.0)), 0.0), 1.0)
            if smoothing > 0.0:
                level_losses = []
                for level, log_probs in enumerate(log_probs_per_level):
                    smooth_targets = _label_smoothed_target_probs(
                        hard_targets[:, level],
                        num_classes=log_probs.size(-1),
                        smoothing=smoothing,
                        dtype=log_probs.dtype,
                    )
                    level_losses.append(_soft_cross_entropy_from_log_probs(log_probs, smooth_targets))
            else:
                level_losses = [
                    F.nll_loss(log_probs, hard_targets[:, level])
                    for level, log_probs in enumerate(log_probs_per_level)
                ]
    else:
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
        if has_effective_probs:
            gk_loss = _global_kl_loss(
                log_probs_per_level,
                hard_targets=hard_targets,
                target_probs_per_level=target_probs_per_level,
                outputs_are_log_probs=True,
            )
        else:
            gk_loss = _global_kl_loss(
                logits_per_level,
                hard_targets=hard_targets,
                target_probs_per_level=target_probs_per_level,
                outputs_are_log_probs=False,
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
