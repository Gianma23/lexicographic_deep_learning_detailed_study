from typing import Any, Dict, List, Optional

import torch


def per_level_top1(logits_per_level: List[torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for level, logits in enumerate(logits_per_level):
        pred = logits.argmax(dim=-1)
        acc = (pred == targets[:, level]).float().mean().item()
        out[f"acc_level_{level}"] = float(acc)
    return out


def full_path_accuracy(logits_per_level: List[torch.Tensor], targets: torch.Tensor) -> float:
    preds = [logits.argmax(dim=-1) for logits in logits_per_level]
    pred_path = torch.stack(preds, dim=1)
    return float((pred_path == targets).all(dim=1).float().mean().item())


def inconsistency_rate(logits_per_level: List[torch.Tensor], taxonomy: Optional[Dict[str, Any]]) -> Optional[float]:
    if not taxonomy or "parent_of" not in taxonomy:
        return None

    parent_of = taxonomy["parent_of"]
    preds = [logits.argmax(dim=-1) for logits in logits_per_level]

    total_checks = 0
    total_violations = 0
    for level in range(1, len(preds)):
        mapping = parent_of.get(str(level)) if isinstance(parent_of, dict) else None
        if mapping is None and isinstance(parent_of, dict):
            mapping = parent_of.get(level)
        if not mapping:
            continue

        mapping = {int(k): int(v) for k, v in mapping.items()}
        child = preds[level].tolist()
        parent = preds[level - 1].tolist()
        for c, p in zip(child, parent):
            total_checks += 1
            if mapping.get(int(c), int(p)) != int(p):
                total_violations += 1

    if total_checks == 0:
        return None
    return float(total_violations / total_checks)


def tice_like_score(logits_per_level: List[torch.Tensor], taxonomy: Optional[Dict[str, Any]]) -> Optional[float]:
    inc = inconsistency_rate(logits_per_level, taxonomy)
    if inc is None:
        return None
    return float(1.0 - inc)


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
