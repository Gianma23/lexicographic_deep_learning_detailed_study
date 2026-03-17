from typing import Any, Dict, List, Optional

import torch


def per_level_top1(logits_per_level: List[torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for level, logits in enumerate(logits_per_level):
        pred = logits.argmax(dim=-1)
        acc = (pred == targets[:, level]).float().mean().item()
        out[f"acc_level_{level}"] = float(acc)
    return out


def weighted_average_precision(logits_per_level: List[torch.Tensor], targets: torch.Tensor) -> float:
    """H-CAST wAP: class-count-weighted Top-1 accuracy across levels."""
    if not logits_per_level:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    for level, logits in enumerate(logits_per_level):
        pred = logits.argmax(dim=-1)
        acc = float((pred == targets[:, level]).float().mean().item())
        weight = float(logits.size(-1))
        weighted_sum += weight * acc
        total_weight += weight

    if total_weight <= 0.0:
        return 0.0
    return float(weighted_sum / total_weight)


def weighted_accuracy(logits_per_level: List[torch.Tensor], targets: torch.Tensor) -> float:
    # Backward-compatible alias.
    return weighted_average_precision(logits_per_level, targets)


def full_path_accuracy(logits_per_level: List[torch.Tensor], targets: torch.Tensor) -> float:
    preds = [logits.argmax(dim=-1) for logits in logits_per_level]
    pred_path = torch.stack(preds, dim=1)
    return float((pred_path == targets).all(dim=1).float().mean().item())


def consistency_rate(logits_per_level: List[torch.Tensor], taxonomy: Optional[Dict[str, Any]]) -> Optional[float]:
    if not taxonomy or "parent_of" not in taxonomy:
        return None

    parent_of = taxonomy["parent_of"]
    preds = [logits.argmax(dim=-1) for logits in logits_per_level]
    if not preds:
        return None

    valid = torch.ones_like(preds[0], dtype=torch.bool)
    for level in range(1, len(preds)):
        mapping = parent_of.get(str(level)) if isinstance(parent_of, dict) else None
        if mapping is None and isinstance(parent_of, dict):
            mapping = parent_of.get(level)
        if mapping is None:
            return None

        mapping = {int(k): int(v) for k, v in mapping.items()}
        child = preds[level]
        parent = preds[level - 1]

        # Build child->parent lookup to check whole-path validity per sample.
        lookup = torch.full((logits_per_level[level].size(-1),), -1, device=child.device, dtype=parent.dtype)
        for child_id, parent_id in mapping.items():
            if 0 <= int(child_id) < lookup.numel():
                lookup[int(child_id)] = int(parent_id)

        mapped_parent = lookup[child]
        valid = valid & (mapped_parent == parent)

    if valid.numel() == 0:
        return None
    return float(valid.float().mean().item())


def tice_score(logits_per_level: List[torch.Tensor], taxonomy: Optional[Dict[str, Any]]) -> Optional[float]:
    """H-CAST convention: TICE is inconsistency rate (lower is better)."""
    consistency = consistency_rate(logits_per_level, taxonomy)
    if consistency is None:
        return None
    return float(1.0 - consistency)


def inconsistency_rate(logits_per_level: List[torch.Tensor], taxonomy: Optional[Dict[str, Any]]) -> Optional[float]:
    # Backward-compatible alias used by existing callers/docs.
    return tice_score(logits_per_level, taxonomy)


def tice_like_score(logits_per_level: List[torch.Tensor], taxonomy: Optional[Dict[str, Any]]) -> Optional[float]:
    # Backward-compatible alias from previous implementation.
    tice = tice_score(logits_per_level, taxonomy)
    if tice is None:
        return None
    return float(1.0 - tice)


def merge_metric_batches(metric_batches: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_batches:
        return {}

    keys = set().union(*[m.keys() for m in metric_batches])
    out: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in metric_batches if k in m]
        if vals:
            out[k] = float(sum(vals) / len(vals))
    return out
