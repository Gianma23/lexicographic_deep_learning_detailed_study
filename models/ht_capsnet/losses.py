from typing import Any, Dict, List, Optional, Tuple

import torch


def _margin_loss(logits: torch.Tensor, target: torch.Tensor, m_pos: float, m_neg: float, down_weight: float) -> torch.Tensor:
    one_hot = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1.0)
    pos = one_hot * torch.relu(m_pos - logits).pow(2)
    neg = (1.0 - one_hot) * torch.relu(logits - m_neg).pow(2)
    return (pos + down_weight * neg).sum(dim=1).mean()


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
    targets: torch.Tensor,
    cfg: Any,
    taxonomy: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits_per_level = output["logits_per_level"]

    m_pos = float(cfg.loss.get("margin_m_pos", 0.9))
    m_neg = float(cfg.loss.get("margin_m_neg", 0.1))
    down_w = float(cfg.loss.get("lambda_downweight", 0.5))
    hier_w = float(cfg.loss.get("hier_weight", 0.2))

    weights = _level_weights(len(logits_per_level), cfg)
    level_losses = []
    weighted_level_losses = []
    for level, logits in enumerate(logits_per_level):
        level_loss = _margin_loss(logits, targets[:, level], m_pos, m_neg, down_w)
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
