from typing import Any, Dict, List, Optional, Tuple

import torch


def _margin_loss(logits: torch.Tensor, target: torch.Tensor, m_pos: float, m_neg: float, down_weight: float) -> torch.Tensor:
    one_hot = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1.0)
    pos = one_hot * torch.relu(m_pos - logits).pow(2)
    neg = (1.0 - one_hot) * torch.relu(logits - m_neg).pow(2)
    return (pos + down_weight * neg).sum(dim=1).mean()


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
    _ = taxonomy
    logits_per_level = output["logits_per_level"]

    m_pos = float(cfg.loss.get("margin_m_pos", 0.9))
    m_neg = float(cfg.loss.get("margin_m_neg", 0.1))
    down_w = float(cfg.loss.get("lambda_downweight", 0.5))

    weights = _level_weights(len(logits_per_level), cfg)
    level_losses = []
    for level, logits in enumerate(logits_per_level):
        level_losses.append(weights[level] * _margin_loss(logits, targets[:, level], m_pos, m_neg, down_w))

    margin = torch.stack(level_losses).sum()
    total = margin

    return total, {
        "loss_total": float(total.detach().item()),
        "total": float(total.detach().item()),
        "margin": float(margin.detach().item()),
    }
