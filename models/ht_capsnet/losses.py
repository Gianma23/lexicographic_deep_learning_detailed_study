from typing import Any, Dict, List, Optional, Tuple, Union

import torch


CapsTargets = Union[torch.Tensor, Dict[str, Any]]


def _target_probs_from_input(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 1:
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1.0)
    if target.ndim == 2 and int(target.size(1)) == int(logits.size(1)):
        probs = target.to(device=logits.device, dtype=logits.dtype)
        return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    raise ValueError(
        f"Invalid target shape for margin loss: expected [B] or [B, {logits.size(1)}], got {tuple(target.shape)}."
    )


def _margin_loss(logits: torch.Tensor, target: torch.Tensor, m_pos: float, m_neg: float, down_weight: float) -> torch.Tensor:
    target_probs = _target_probs_from_input(logits, target)
    pos = target_probs * torch.relu(m_pos - logits).pow(2)
    neg = (1.0 - target_probs) * torch.relu(logits - m_neg).pow(2)
    return (pos + down_weight * neg).sum(dim=1).mean()


def _hard_targets_from_input(targets: CapsTargets) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        return targets
    hard_targets = targets.get("hard_targets")
    if isinstance(hard_targets, torch.Tensor):
        return hard_targets
    labels_a = targets.get("labels_a")
    if isinstance(labels_a, torch.Tensor):
        return labels_a
    raise TypeError("Expected hard targets tensor or mixup target dict with `hard_targets` tensor.")


def _mixup_target_distributions(logits_per_level: List[torch.Tensor], targets: CapsTargets) -> Optional[List[torch.Tensor]]:
    if not isinstance(targets, dict):
        return None
    soft_targets = targets.get("soft_targets_per_level")
    if not isinstance(soft_targets, (list, tuple)) or len(soft_targets) != len(logits_per_level):
        return None

    out: List[torch.Tensor] = []
    for level, logits in enumerate(logits_per_level):
        target_level = soft_targets[level]
        if not isinstance(target_level, torch.Tensor):
            return None
        if target_level.ndim != 2 or int(target_level.size(1)) != int(logits.size(-1)):
            return None
        probs = target_level.to(device=logits.device, dtype=logits.dtype)
        out.append(probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12))
    return out


def _normalize_parent_of(taxonomy: Optional[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
    if not taxonomy or "parent_of" not in taxonomy:
        return {}
    out: Dict[int, Dict[int, int]] = {}
    for k, v in taxonomy["parent_of"].items():
        out[int(k)] = {int(ck): int(pk) for ck, pk in v.items()}
    return out


def _initial_caps_loss_weights(num_classes_per_level: List[int]) -> List[float]:
    if not num_classes_per_level:
        return []
    total = float(sum(float(v) for v in num_classes_per_level))
    if total <= 0.0:
        return [1.0 for _ in num_classes_per_level]

    q_vals = [1.0 - (float(v) / total) for v in num_classes_per_level]
    q_sum = float(sum(q_vals))
    if q_sum <= 0.0:
        return [1.0 for _ in num_classes_per_level]
    return [float(v / q_sum) for v in q_vals]


def _dynamic_caps_level_weights(
    logits_per_level: List[torch.Tensor],
    hard_targets: Optional[torch.Tensor],
    decay: float,
) -> List[float]:
    num_classes_per_level = [int(logits.size(-1)) for logits in logits_per_level]
    initial = _initial_caps_loss_weights(num_classes_per_level)
    num_levels = len(logits_per_level)
    if not initial or len(initial) != num_levels:
        return [1.0 for _ in range(num_levels)]

    if hard_targets is None or hard_targets.ndim != 2 or int(hard_targets.size(1)) != num_levels:
        return initial

    acc_per_level: List[float] = []
    for level, logits in enumerate(logits_per_level):
        pred = logits.argmax(dim=-1)
        acc = (pred == hard_targets[:, level]).float().mean().detach().item()
        acc_per_level.append(float(acc))

    taus = [1.0 - (acc_per_level[level] * initial[level]) for level in range(num_levels)]
    tau_sum = float(sum(taus))
    if tau_sum <= 0.0:
        return initial

    scale = 1.0 - min(max(float(decay), 0.0), 1.0)
    return [max(0.0, scale * (tau / tau_sum)) for tau in taus]


def _level_weights(
    logits_per_level: List[torch.Tensor],
    cfg: Any,
    targets: CapsTargets,
    hard_targets: Optional[torch.Tensor],
) -> List[float]:
    num_levels = len(logits_per_level)
    if isinstance(targets, dict):
        override = targets.get("level_weights")
        if isinstance(override, torch.Tensor):
            vals = [float(v) for v in override.detach().cpu().tolist()]
            if len(vals) == num_levels:
                return vals
        if isinstance(override, (list, tuple)):
            vals = [float(v) for v in override]
            if len(vals) == num_levels:
                return vals

    loss_cfg = cfg.model.loss
    mode = str(loss_cfg.get("weight_mode", "dynamic")).strip().lower()
    if mode == "static":
        static = _initial_caps_loss_weights([int(logits.size(-1)) for logits in logits_per_level])
        if len(static) == num_levels:
            return static
    elif mode == "dynamic":
        dynamic = _dynamic_caps_level_weights(
            logits_per_level=logits_per_level,
            hard_targets=hard_targets,
            decay=float(loss_cfg.get("dynamic_weight", 0.0)),
        )
        if len(dynamic) == num_levels:
            return dynamic

    values = loss_cfg.get("level_weights")
    if values is None:
        return [1.0 for _ in range(num_levels)]

    values = [float(v) for v in values]
    if len(values) != num_levels:
        return [1.0 for _ in range(num_levels)]
    return values


def compute_loss(
    output: Dict[str, Any],
    targets: CapsTargets,
    cfg: Any,
    taxonomy: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    _ = taxonomy
    logits_per_level = output["logits_per_level"]
    mixup_target_probs = _mixup_target_distributions(logits_per_level, targets)
    try:
        hard_targets = _hard_targets_from_input(targets)
    except TypeError:
        hard_targets = None
    if mixup_target_probs is None and hard_targets is None:
        raise TypeError("Expected hard targets tensor for non-mixup HT-CapsNet loss.")

    loss_cfg = cfg.model.loss
    m_pos = float(loss_cfg.get("margin_m_pos", 0.9))
    m_neg = float(loss_cfg.get("margin_m_neg", 0.1))
    down_w = float(loss_cfg.get("lambda_downweight", 0.5))

    weights = _level_weights(logits_per_level, cfg, targets, hard_targets)
    level_losses = []
    weighted_level_losses = []
    for level, logits in enumerate(logits_per_level):
        level_target = mixup_target_probs[level] if mixup_target_probs is not None else hard_targets[:, level]
        level_loss = _margin_loss(logits, level_target, m_pos, m_neg, down_w)
        level_losses.append(level_loss)
        weighted_level_losses.append(weights[level] * level_loss)

    margin = torch.stack(weighted_level_losses).sum()
    cons = torch.zeros((), device=margin.device)
    total = margin

    metrics = {
        "total": float(total.detach().item()),
        "margin": float(margin.detach().item()),
        "consistency": 0.0,
    }
    for level, level_loss in enumerate(level_losses):
        metrics[f"loss_level_{level}"] = float(level_loss.detach().item())
    return total, metrics
