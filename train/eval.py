from typing import Any, Dict, Optional

import torch

from .metrics import full_path_accuracy, inconsistency_rate, per_level_top1, tice_like_score


def evaluate_batch(output: Dict[str, Any], targets: torch.Tensor, taxonomy: Optional[Dict] = None) -> Dict[str, float]:
    logits_per_level = output["logits_per_level"]

    metrics = per_level_top1(logits_per_level, targets)
    metrics["acc_path"] = full_path_accuracy(logits_per_level, targets)

    inc = inconsistency_rate(logits_per_level, taxonomy)
    tice = tice_like_score(logits_per_level, taxonomy)
    if inc is not None:
        metrics["inconsistency_rate"] = inc
    if tice is not None:
        metrics["tice_like"] = tice

    return metrics


def pretty_metrics(metrics: Dict[str, float]) -> str:
    parts = []
    for key in sorted(metrics.keys()):
        parts.append(f"{key}={metrics[key]:.4f}")
    return " | ".join(parts)
