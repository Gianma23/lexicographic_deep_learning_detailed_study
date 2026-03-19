from typing import Any, Dict, List, Optional

import torch

from .metrics import full_path_accuracy, per_level_top1, tice_score, weighted_average_precision


def evaluate_batch(output: Dict[str, Any], targets: torch.Tensor, taxonomy: Optional[Dict] = None) -> Dict[str, float]:
    logits_per_level = output["logits_per_level"]
    projected_probs_per_level = output.get("projected_probs_per_level")

    metrics = per_level_top1(logits_per_level, targets)
    metrics["weighted_ap"] = weighted_average_precision(logits_per_level, targets)
    metrics["weighted_acc"] = metrics["weighted_ap"]  # backward-compatible alias

    # H-CAST: FPA is full-path exact-match accuracy.
    fpa = full_path_accuracy(logits_per_level, targets)
    metrics["fpa"] = fpa
    metrics["acc_path"] = fpa  # backward-compatible alias

    # H-CAST: TICE is inconsistency rate (lower is better).
    tice = tice_score(logits_per_level, taxonomy)
    if tice is not None:
        metrics["tice"] = tice
        metrics["inconsistency_rate"] = tice  # backward-compatible alias
        metrics["tice_like"] = float(1.0 - tice)

    if isinstance(projected_probs_per_level, list) and projected_probs_per_level:
        tice_projected = tice_score(projected_probs_per_level, taxonomy)
        if tice_projected is not None:
            metrics["tice_projected"] = tice_projected

    return metrics


def _level_acc_keys(metrics: Dict[str, float]) -> List[str]:
    keys = [k for k in metrics.keys() if k.startswith("acc_level_") and k[len("acc_level_") :].isdigit()]
    return sorted(keys, key=lambda k: int(k.split("_")[-1]))


def _format_ratio(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def pretty_metrics(metrics: Dict[str, float], level_names: Optional[List[str]] = None) -> str:
    level_names = level_names or []
    sections: List[str] = []

    level_parts: List[str] = []
    for key in _level_acc_keys(metrics):
        idx = int(key.split("_")[-1])
        name = level_names[idx] if idx < len(level_names) else f"L{idx}"
        level_parts.append(f"{name}={_format_ratio(metrics[key])}")
    if level_parts:
        sections.append("\n\t\tAccuracies[" + ", ".join(level_parts) + "]")

    summary_parts: List[str] = []
    if "weighted_ap" in metrics:
        summary_parts.append(f"wAP={_format_ratio(metrics['weighted_ap'])}")
    elif "weighted_acc" in metrics:
        summary_parts.append(f"weighted={_format_ratio(metrics['weighted_acc'])}")
    if "fpa" in metrics:
        summary_parts.append(f"FPA={_format_ratio(metrics['fpa'])}")
    if "tice" in metrics:
        summary_parts.append(f"TICE={_format_ratio(metrics['tice'])}")
    if "tice_projected" in metrics:
        summary_parts.append(f"TICE_proj={_format_ratio(metrics['tice_projected'])}")
    if summary_parts:
        sections.append("Summary[" + ", ".join(summary_parts) + "]")

    loss_parts: List[str] = []
    if "total" in metrics:
        loss_parts.append(f"total={metrics['total']:.4f}")
    if "level_ce" in metrics:
        loss_parts.append(f"level_ce={metrics['level_ce']:.4f}")
    if "gk_loss" in metrics:
        loss_parts.append(f"gk_loss={metrics['gk_loss']:.4f}")

    loss_level_keys = [k for k in metrics.keys() if k.startswith("loss_level_") and k[len("loss_level_") :].isdigit()]
    for key in sorted(loss_level_keys, key=lambda k: int(k.split("_")[-1])):
        idx = int(key.split("_")[-1])
        name = level_names[idx] if idx < len(level_names) else f"L{idx}"
        loss_parts.append(f"loss_{name}={metrics[key]:.4f}")
    if loss_parts:
        sections.append("\n\t\tLoss[" + ", ".join(loss_parts) + "]")

    known_keys = set(_level_acc_keys(metrics))
    known_keys.update(
        {
            "weighted_acc",
            "weighted_ap",
            "fpa",
            "tice",
            "tice_projected",
            "acc_path",
            "inconsistency_rate",
            "tice_like",
            "total",
            "level_ce",
            "gk_loss",
        }
    )
    known_keys.update(loss_level_keys)

    """ other_keys = sorted([k for k in metrics.keys() if k not in known_keys])
    if other_keys:
        other_parts: List[str] = []
        for key in other_keys:
            value = float(metrics[key])
            if key in {"mixup_applied"}:
                other_parts.append(f"{key}={_format_ratio(value)}")
            else:
                other_parts.append(f"{key}={value:.4f}")
        sections.append("\n\t\tOther[" + ", ".join(other_parts) + "]") """

    if not sections:
        return "(no metrics)"
    return " | ".join(sections)
