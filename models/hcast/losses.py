from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


HcastTargets = Union[torch.Tensor, Dict[str, Any]]
_EPS = 1e-12


def _section_to_dict(section: Any) -> Dict[str, Any]:
    """Convert a config section into a plain dictionary when possible."""
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "items"):
        return {k: v for k, v in section.items()}
    return {}


def _normalize_rows(probs: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Normalize each row to sum to one with epsilon protection."""
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)


def _label_smoothing_from_cfg(cfg: Any) -> float:
    """Extract label smoothing from the training config and clamp it to [0, 1]."""
    train_cfg = _section_to_dict(getattr(cfg, "train", None))
    if not train_cfg and hasattr(cfg, "get"):
        train_cfg = _section_to_dict(cfg.get("train", None))
    return min(max(float(train_cfg.get("smoothing", 0.0)), 0.0), 1.0)


def _soft_targets_from_input(
    scores_per_level: List[torch.Tensor],
    targets: HcastTargets,
) -> Optional[List[torch.Tensor]]:
    """Validate and normalize per-level soft targets, typically from MixUp/CutMix."""
    if not isinstance(targets, dict):
        return None

    soft_targets = targets.get("soft_targets_per_level")
    if not isinstance(soft_targets, (list, tuple)) or len(soft_targets) != len(scores_per_level):
        return None

    out: List[torch.Tensor] = []
    for scores, target_level in zip(scores_per_level, soft_targets):
        if not isinstance(target_level, torch.Tensor) or target_level.ndim != 2:
            return None
        if int(target_level.size(1)) != int(scores.size(-1)):
            return None
        # MixUp/CutMix provide dense targets per level. We move them onto the
        # same device/dtype as the selected score source and re-normalize in
        # case interpolation introduced small row-sum drift.
        probs = target_level.to(device=scores.device, dtype=scores.dtype)
        out.append(_normalize_rows(probs))
    return out


def _hard_targets_from_input(targets: HcastTargets, num_levels: int) -> Optional[torch.Tensor]:
    """Extract hard class indices with shape [batch, num_levels] from the input targets."""
    hard_targets = (
        targets
        if isinstance(targets, torch.Tensor)
        else targets.get("hard_targets")
        if isinstance(targets, dict)
        else None
    )
    if not isinstance(hard_targets, torch.Tensor):
        return None
    if hard_targets.ndim != 2 or int(hard_targets.size(1)) != int(num_levels):
        return None
    return hard_targets.long()


def _soft_cross_entropy_from_log_probs(log_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy when both predictions and targets are distributions."""
    probs = target_probs.to(device=log_probs.device, dtype=log_probs.dtype)
    probs = _normalize_rows(probs)
    return -(probs * log_probs).sum(dim=-1).mean()


def _dynamic_level_weights(
    scores_per_level: List[torch.Tensor],
    hard_targets: torch.Tensor,
    gamma: float,
    eps: float,
) -> List[float]:
    """Estimate per-level loss weights from current accuracy and class-count priors."""
    num_levels = len(scores_per_level)
    if num_levels == 0:
        return []

    num_classes_per_level = [max(int(scores.size(-1)), 0) for scores in scores_per_level]
    total_classes = float(sum(num_classes_per_level))
    if total_classes <= eps:
        omega_init = [1.0 / num_levels for _ in range(num_levels)]
    else:
        omega_init = [1.0 - (float(num_classes) / total_classes) for num_classes in num_classes_per_level]
        omega_sum = float(sum(omega_init))
        omega_init = [value / omega_sum for value in omega_init] if omega_sum > eps else [1.0 / num_levels] * num_levels

    rho: List[float] = []
    for level, scores in enumerate(scores_per_level):
        acc = (scores.argmax(dim=-1) == hard_targets[:, level]).float().mean().item()
        rho.append((1.0 - float(acc)) * omega_init[level])

    rho_sum = float(sum(rho))
    base = omega_init if rho_sum <= eps else [value / rho_sum for value in rho]
    scale = 1.0 - min(max(float(gamma), 0.0), 1.0)
    return [scale * value for value in base]


def _global_kl_loss(
    scores_per_level: List[torch.Tensor],
    hard_targets: Optional[torch.Tensor],
    target_probs_per_level: Optional[List[torch.Tensor]],
) -> torch.Tensor:
    """Compute the optional global KL regularizer over concatenated hierarchy outputs."""
    # Global KL treats the whole hierarchy as one concatenated categorical
    # vector. This regularizer therefore depends on the same score source
    # chosen for the per-level losses below.
    all_outputs = F.log_softmax(torch.cat(scores_per_level, dim=1), dim=1)

    if target_probs_per_level is not None:
        all_targets = torch.cat(
            [target_probs.to(dtype=all_outputs.dtype) for target_probs in target_probs_per_level],
            dim=1,
        )
    else:
        if hard_targets is None:
            raise ValueError("Either `hard_targets` or `target_probs_per_level` must be provided for global KL.")
        onehots = [
            F.one_hot(hard_targets[:, level], num_classes=scores.size(-1)).float()
            for level, scores in enumerate(scores_per_level)
        ]
        all_targets = torch.cat(onehots, dim=1)

    all_targets = all_targets.to(device=all_outputs.device, dtype=all_outputs.dtype)
    all_targets = _normalize_rows(all_targets)
    return F.kl_div(all_outputs, all_targets, reduction="batchmean")


def _level_losses_from_scores(
    scores_per_level: List[torch.Tensor],
    hard_targets: Optional[torch.Tensor],
    target_probs_per_level: Optional[List[torch.Tensor]],
    smoothing: float,
) -> List[torch.Tensor]:
    """Compute one loss value per hierarchy level from logits."""
    if target_probs_per_level is not None:
        # Soft-target path used for MixUp/CutMix.
        log_probs_per_level = [F.log_softmax(scores, dim=-1) for scores in scores_per_level]
        return [
            _soft_cross_entropy_from_log_probs(log_probs, target_probs_per_level[level])
            for level, log_probs in enumerate(log_probs_per_level)
        ]

    if hard_targets is None:
        raise TypeError("Expected hard targets tensor of shape [B, L].")

    # Raw or HCC-projected logits go straight into cross_entropy.
    return [
        F.cross_entropy(scores, hard_targets[:, level], label_smoothing=smoothing)
        for level, scores in enumerate(scores_per_level)
    ]


def _resolve_score_source(output: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Choose raw logits or HCC effective logits as the loss score source."""
    logits_per_level = output["logits_per_level"]
    if not isinstance(logits_per_level, list) or not logits_per_level:
        raise ValueError("H-CAST output must contain non-empty `logits_per_level`.")

    effective_logits = output.get("effective_logits_per_level")
    if effective_logits is not None:
        if not isinstance(effective_logits, list) or len(effective_logits) != len(logits_per_level):
            raise ValueError("`effective_logits_per_level` must be None or a list aligned with `logits_per_level`.")
        return logits_per_level, effective_logits

    # Before HCC activates, or when HCC is disabled, downstream loss stays on raw logits.
    return logits_per_level, logits_per_level


def _resolve_level_weights(
    loss_cfg: Dict[str, Any],
    score_source_per_level: List[torch.Tensor],
    hard_targets: Optional[torch.Tensor],
    num_levels: int,
) -> List[float]:
    """Return static or dynamic per-level weights according to the loss config."""
    level_weight_cfg = _section_to_dict(loss_cfg.get("level_weighting", {}))
    mode = level_weight_cfg.get("mode", "static")
    if not isinstance(mode, str):
        raise ValueError("model.loss.level_weighting.mode must be a string.")
    if mode == "static":
        return [1.0 for _ in range(num_levels)]
    if mode != "dynamic":
        raise ValueError("model.loss.level_weighting.mode must be one of ['static', 'dynamic'].")

    if hard_targets is None:
        raise ValueError(
            "Dynamic level weighting requires hard targets. "
            "Pass `hard_targets` in target dict when using soft targets."
        )

    gamma = float(level_weight_cfg.get("gamma", 0.0))
    eps = max(float(level_weight_cfg.get("eps", _EPS)), _EPS)
    # Dynamic weights are computed from the same logit source used by the loss.
    return _dynamic_level_weights(
        scores_per_level=score_source_per_level,
        hard_targets=hard_targets,
        gamma=gamma,
        eps=eps,
    )


def compute_loss(
    output: Dict[str, Any],
    targets: HcastTargets,
    cfg: Any,
    _taxonomy: Optional[Dict[str, Any]] = None,
    return_aux: bool = False,
) -> Union[
    Tuple[torch.Tensor, Dict[str, float]],
    Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]],
]:
    loss_cfg = _section_to_dict(getattr(cfg.model, "loss", {}))

    # If effective_logits_per_level is present, training uses HCC logits;
    # otherwise training uses raw logits.
    logits_per_level, score_source_per_level = _resolve_score_source(output)
    target_probs_per_level = _soft_targets_from_input(score_source_per_level, targets)
    hard_targets = _hard_targets_from_input(targets, num_levels=len(logits_per_level))

    if target_probs_per_level is None and hard_targets is None:
        raise TypeError("Expected hard targets tensor of shape [B, L].")
    if hard_targets is not None:
        hard_targets = hard_targets.to(device=score_source_per_level[0].device, dtype=torch.long)

    smoothing = _label_smoothing_from_cfg(cfg)
    loss_scores_per_level = score_source_per_level
    gk_scores_per_level = score_source_per_level

    level_losses = _level_losses_from_scores(
        scores_per_level=loss_scores_per_level,
        hard_targets=hard_targets,
        target_probs_per_level=target_probs_per_level,
        smoothing=smoothing,
    )

    level_weights = _resolve_level_weights(
        loss_cfg=loss_cfg,
        score_source_per_level=score_source_per_level,
        hard_targets=hard_targets,
        num_levels=len(level_losses),
    )

    # Main task loss is the weighted sum of per-level losses.
    weighted_level_losses = [
        level_loss * float(level_weights[level])
        for level, level_loss in enumerate(level_losses)
    ]
    weighted_level_ce = torch.stack(weighted_level_losses).sum()
    total = weighted_level_ce

    if bool(loss_cfg.get("globalkl", False)):
        global_kl_weight = float(loss_cfg.get("gk_weight", 1.0))
        gk_loss = _global_kl_loss(
            scores_per_level=gk_scores_per_level,
            hard_targets=hard_targets,
            target_probs_per_level=target_probs_per_level,
        )
        total = total + global_kl_weight * gk_loss
    else:
        gk_loss = torch.zeros((), device=total.device, dtype=total.dtype)

    metrics: Dict[str, float] = {
        "total": float(total.detach().item()),
        "level_ce": float(weighted_level_ce.detach().item()),
        "gk_loss": float(gk_loss.detach().item()),
    }
    for level, level_loss in enumerate(level_losses):
        metrics[f"loss_level_{level}"] = float(level_loss.detach().item())
        metrics[f"loss_weight_level_{level}"] = float(level_weights[level])

    if not return_aux:
        return total, metrics

    aux_payload: Dict[str, Any] = {"level_losses": list(level_losses)}
    return total, metrics, aux_payload
