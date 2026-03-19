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
    labels_a = targets.get("labels_a")
    if not isinstance(labels_a, torch.Tensor):
        raise TypeError("Expected hard targets tensor or mixup target dict with `labels_a` tensor.")
    return labels_a


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


def _hier_consistency_penalty(logits_per_level: List[torch.Tensor], parent_of: Dict[int, Dict[int, int]]) -> torch.Tensor:
    if not parent_of:
        return torch.zeros((), device=logits_per_level[0].device)

    penalties = []
    for level in range(1, len(logits_per_level)):
        mapping = parent_of.get(level)
        if not mapping:
            continue
        child_pred = logits_per_level[level].argmax(dim=-1)
        parent_pred = logits_per_level[level - 1].argmax(dim=-1)

        mismatch = []
        for c, p in zip(child_pred.tolist(), parent_pred.tolist()):
            mismatch.append(1.0 if mapping.get(int(c), int(p)) != int(p) else 0.0)
        penalties.append(torch.tensor(mismatch, device=logits_per_level[0].device).mean())

    if not penalties:
        return torch.zeros((), device=logits_per_level[0].device)
    return torch.stack(penalties).mean()


def _level_weights(num_levels: int, cfg: Any) -> List[float]:
    values = cfg.loss.get("level_weights")
    if values is None:
        return [1.0 / num_levels for _ in range(num_levels)]

    values = [float(v) for v in values]
    if len(values) != num_levels:
        return [1.0 / num_levels for _ in range(num_levels)]

    s = sum(values)
    if s <= 0:
        return [1.0 / num_levels for _ in range(num_levels)]
    return [v / s for v in values]


def compute_loss(
    output: Dict[str, Any],
    targets: CapsTargets,
    cfg: Any,
    taxonomy: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits_per_level = output["logits_per_level"]
    mixup_target_probs = _mixup_target_distributions(logits_per_level, targets)
    hard_targets = _hard_targets_from_input(targets) if mixup_target_probs is None else None

    m_pos = float(cfg.loss.get("margin_m_pos", 0.9))
    m_neg = float(cfg.loss.get("margin_m_neg", 0.1))
    down_w = float(cfg.loss.get("lambda_downweight", 0.5))
    hier_w = float(cfg.loss.get("hier_weight", 0.2))

    weights = _level_weights(len(logits_per_level), cfg)
    level_losses = []
    weighted_level_losses = []
    for level, logits in enumerate(logits_per_level):
        level_target = mixup_target_probs[level] if mixup_target_probs is not None else hard_targets[:, level]
        level_loss = _margin_loss(logits, level_target, m_pos, m_neg, down_w)
        level_losses.append(level_loss)
        weighted_level_losses.append(weights[level] * level_loss)

    margin = torch.stack(weighted_level_losses).sum()
    cons = _hier_consistency_penalty(logits_per_level, _normalize_parent_of(taxonomy))
    total = margin + hier_w * cons

    metrics = {
        "total": float(total.detach().item()),
        "margin": float(margin.detach().item()),
        "consistency": float(cons.detach().item()),
    }
    for level, level_loss in enumerate(level_losses):
        metrics[f"loss_level_{level}"] = float(level_loss.detach().item())
    return total, metrics
