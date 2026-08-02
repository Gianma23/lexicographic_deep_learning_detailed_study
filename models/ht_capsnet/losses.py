from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..common.hcc import select_effective_logits

CapsTargets = Union[torch.Tensor, Dict[str, Any]]


def _target_probs_from_input(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Convert hard/soft targets to normalized class probabilities."""
    if target.ndim == 1:
        target = target.to(device=logits.device, dtype=torch.long)
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1.0)

    if target.ndim == 2 and int(target.size(1)) == int(logits.size(1)):
        probs = target.to(device=logits.device, dtype=logits.dtype)
        return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    raise ValueError(
        f"Invalid target shape for margin loss: expected [B] or [B, {logits.size(1)}], got {tuple(target.shape)}."
    )


def _margin_loss(logits: torch.Tensor, target: torch.Tensor, m_pos: float, m_neg: float, lambda_down: float) -> torch.Tensor:
    """Capsule margin loss used in the upstream HT-CapsNet codebase."""
    target_probs = _target_probs_from_input(logits, target)
    present = target_probs * torch.relu(m_pos - logits).pow(2)
    absent = (1.0 - target_probs) * torch.relu(logits - m_neg).pow(2)
    return (present + lambda_down * absent).sum(dim=1).mean()


def _soft_targets_from_input(
    logits_per_level: List[torch.Tensor],
    targets: CapsTargets,
) -> Optional[List[torch.Tensor]]:
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


def _hard_targets_from_input(
    targets: CapsTargets,
    num_levels: int,
) -> Optional[torch.Tensor]:
    if isinstance(targets, torch.Tensor):
        hard_targets = targets
    elif isinstance(targets, dict):
        hard_targets = targets.get("hard_targets")
    else:
        return None

    if not isinstance(hard_targets, torch.Tensor):
        return None
    if hard_targets.ndim != 2:
        return None
    if int(hard_targets.size(1)) != int(num_levels):
        return None
    return hard_targets.long()


def _resolve_level_targets(
    logits_per_level: List[torch.Tensor],
    targets: CapsTargets,
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    soft_targets = _soft_targets_from_input(logits_per_level, targets)
    hard_targets = _hard_targets_from_input(targets, num_levels=len(logits_per_level))

    if soft_targets is not None:
        return soft_targets, hard_targets

    if hard_targets is None:
        raise TypeError("Expected hard targets [B, L] or `soft_targets_per_level` list in targets dict.")

    hard_targets = hard_targets.to(device=logits_per_level[0].device, dtype=torch.long)
    level_targets = [hard_targets[:, level] for level in range(len(logits_per_level))]
    return level_targets, hard_targets


def _target_indices_for_accuracy(level_targets: List[torch.Tensor]) -> List[torch.Tensor]:
    indices: List[torch.Tensor] = []
    for target in level_targets:
        if target.ndim == 1:
            indices.append(target.long())
            continue
        if target.ndim == 2:
            indices.append(target.argmax(dim=-1).long())
            continue
        raise ValueError(f"Invalid target rank {target.ndim}; expected 1D or 2D per-level targets.")
    return indices


def _initial_level_weights(num_classes_per_level: List[int]) -> List[float]:
    """Upstream `initial_lw`: inverse class-share weights normalized to sum to 1."""
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


def _dynamic_level_weights(
    logits_per_level: List[torch.Tensor],
    target_indices_per_level: List[torch.Tensor],
    decay: float,
) -> List[float]:
    """Upstream dynamic LW update: (1-w) * tau_i / sum_j tau_j, tau_i = 1 - acc_i * init_i."""
    num_levels = len(logits_per_level)
    initial = _initial_level_weights([int(logits.size(-1)) for logits in logits_per_level])
    if not initial or len(initial) != num_levels:
        return [1.0 for _ in range(num_levels)]
    if len(target_indices_per_level) != num_levels:
        return initial

    acc_per_level: List[float] = []
    for level, logits in enumerate(logits_per_level):
        target = target_indices_per_level[level].to(device=logits.device, dtype=torch.long)
        if target.ndim != 1 or int(target.size(0)) != int(logits.size(0)):
            return initial
        pred = logits.argmax(dim=-1)
        acc = (pred == target).float().mean().detach().item()
        acc_per_level.append(float(acc))

    taus = [1.0 - (acc_per_level[level] * initial[level]) for level in range(num_levels)]
    tau_sum = float(sum(taus))
    if tau_sum <= 0.0:
        return initial

    scale = 1.0 - min(max(float(decay), 0.0), 1.0)
    return [scale * (tau / tau_sum) for tau in taus]


def _level_weights(
    output: Dict[str, Any],
    logits_per_level: List[torch.Tensor],
    cfg: Any,
) -> List[float]:
    num_levels = len(logits_per_level)
    loss_cfg = cfg.model.loss
    mode = loss_cfg.get("weight_mode", "dynamic")
    if not isinstance(mode, str):
        raise ValueError("HT-CapsNet model.loss.weight_mode must be a string.")

    if mode == "dynamic":
        current = output.get("level_loss_weights")
        if isinstance(current, torch.Tensor) and current.numel() == num_levels:
            return [float(value) for value in current.detach().view(-1).cpu().tolist()]
        initial = _initial_level_weights([int(logits.size(-1)) for logits in logits_per_level])
        return initial if len(initial) == num_levels else [1.0 for _ in range(num_levels)]
    if mode == "static":
        static_weights = _initial_level_weights([int(logits.size(-1)) for logits in logits_per_level])
        return static_weights if len(static_weights) == num_levels else [1.0 for _ in range(num_levels)]
    if mode == "none":
        return [1.0 for _ in range(num_levels)]

    raise ValueError(f"Unsupported HT-CapsNet loss weight_mode '{mode}'. Supported values: dynamic, static, none.")


def compute_loss(
    output: Dict[str, Any],
    targets: CapsTargets,
    cfg: Any,
    taxonomy: Optional[Dict[str, Any]] = None,
    return_aux: bool = False,
) -> Union[
    Tuple[torch.Tensor, Dict[str, float]],
    Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]],
]:
    """Compute HT-CapsNet hierarchical margin loss and per-level metrics."""
    _ = taxonomy
    # If effective_logits_per_level is present, training uses HCC-corrected
    # logits; otherwise training uses raw logits.
    logits_per_level, score_source_per_level = select_effective_logits(output)

    level_targets, _hard_targets = _resolve_level_targets(logits_per_level, targets)
    target_indices = _target_indices_for_accuracy(level_targets)

    loss_cfg = cfg.model.loss
    m_pos = float(loss_cfg.get("margin_m_pos", 0.9))
    m_neg = float(loss_cfg.get("margin_m_neg", 0.1))
    lambda_down = float(loss_cfg.get("lambda_downweight", 0.5))

    weights = _level_weights(output, score_source_per_level, cfg)

    level_losses: List[torch.Tensor] = []
    weighted_level_losses: List[torch.Tensor] = []
    for level, logits in enumerate(score_source_per_level):
        level_loss = _margin_loss(logits, level_targets[level], m_pos, m_neg, lambda_down)
        level_losses.append(level_loss)
        weighted_level_losses.append(level_loss * float(weights[level]))

    margin = torch.stack(weighted_level_losses).sum()
    total = margin

    metrics = {
        "total": float(total.detach().item()),
        "margin": float(margin.detach().item()),
    }
    for level, level_loss in enumerate(level_losses):
        metrics[f"loss_level_{level}"] = float(level_loss.detach().item())
        metrics[f"loss_weight_level_{level}"] = float(weights[level])
    if not return_aux:
        return total, metrics

    aux_payload: Dict[str, Any] = {"level_losses": list(level_losses)}
    if str(loss_cfg.get("weight_mode", "dynamic")) == "dynamic":
        next_weights = _dynamic_level_weights(
            logits_per_level=score_source_per_level,
            target_indices_per_level=target_indices,
            decay=float(loss_cfg.get("dynamic_weight", 0.0)),
        )
        aux_payload["next_level_loss_weights"] = torch.tensor(
            next_weights,
            device=score_source_per_level[0].device,
            dtype=score_source_per_level[0].dtype,
        )
    return total, metrics, aux_payload
