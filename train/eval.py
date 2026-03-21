from typing import Any, Dict, List, Optional

import torch

from .metrics import full_path_accuracy, per_level_top1, tice_score, weighted_average_precision


def evaluate_batch(output: Dict[str, Any], targets: torch.Tensor, taxonomy: Optional[Dict] = None) -> Dict[str, float]:
    logits_per_level = output["logits_per_level"]
    effective_probs_per_level = output.get("effective_probs_per_level")
    has_effective_probs = (
        isinstance(effective_probs_per_level, list)
        and len(effective_probs_per_level) == len(logits_per_level)
    )
    if effective_probs_per_level is not None and not has_effective_probs:
        raise ValueError("`effective_probs_per_level` must be None or a list aligned with logits levels.")

    score_source = effective_probs_per_level if has_effective_probs else logits_per_level
    enforce_hierarchy_decode = bool(has_effective_probs and taxonomy is not None)

    metrics = per_level_top1(score_source, targets)
    metrics["weighted_ap"] = weighted_average_precision(score_source, targets)

    # H-CAST: FPA is full-path exact-match accuracy.
    fpa = full_path_accuracy(
        score_source,
        targets,
        taxonomy=taxonomy,
        enforce_hierarchy=enforce_hierarchy_decode,
    )
    metrics["fpa"] = fpa

    # H-CAST: TICE is inconsistency rate (lower is better).
    tice = tice_score(
        score_source,
        taxonomy,
        enforce_hierarchy=enforce_hierarchy_decode,
    )
    if tice is not None:
        metrics["tice"] = tice

    if has_effective_probs:
        # Keep independent-argmax projected path metrics for diagnostics.
        fpa_projected = full_path_accuracy(effective_probs_per_level, targets, enforce_hierarchy=False)
        metrics["fpa_projected"] = fpa_projected
        tice_projected = tice_score(effective_probs_per_level, taxonomy, enforce_hierarchy=False)
        if tice_projected is not None:
            metrics["tice_projected"] = tice_projected

    design1_diagnostics = output.get("design1_diagnostics")
    if isinstance(design1_diagnostics, dict):
        for key, value in design1_diagnostics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    metrics[key] = float(value.detach().item())
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)

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
    if "fpa" in metrics:
        summary_parts.append(f"FPA={_format_ratio(metrics['fpa'])}")
    if "tice" in metrics:
        summary_parts.append(f"TICE={_format_ratio(metrics['tice'])}")
    if "fpa_projected" in metrics:
        summary_parts.append(f"FPA_proj={_format_ratio(metrics['fpa_projected'])}")
    if "tice_projected" in metrics:
        summary_parts.append(f"TICE_proj={_format_ratio(metrics['tice_projected'])}")
    if summary_parts:
        sections.append("Summary[" + ", ".join(summary_parts) + "]")

    proj_diag_parts: List[str] = []
    if "proj_has_negative" in metrics:
        proj_diag_parts.append(f"AnyNegBatch={_format_ratio(metrics['proj_has_negative'])}")

    neg_frac_keys = [
        k
        for k in metrics.keys()
        if k.startswith("proj_neg_frac_level_") and k[len("proj_neg_frac_level_") :].isdigit()
    ]
    if neg_frac_keys:
        neg_parts: List[str] = []
        for key in sorted(neg_frac_keys, key=lambda k: int(k.split("_")[-1])):
            idx = int(key.split("_")[-1])
            name = level_names[idx] if idx < len(level_names) else f"L{idx}"
            neg_parts.append(f"{name}={_format_ratio(metrics[key])}")
        proj_diag_parts.append("NegFrac[" + ", ".join(neg_parts) + "]")

    min_keys = [
        k
        for k in metrics.keys()
        if k.startswith("proj_min_level_") and k[len("proj_min_level_") :].isdigit()
    ]
    if min_keys:
        min_parts: List[str] = []
        for key in sorted(min_keys, key=lambda k: int(k.split("_")[-1])):
            idx = int(key.split("_")[-1])
            name = level_names[idx] if idx < len(level_names) else f"L{idx}"
            min_parts.append(f"{name}={metrics[key]:.3e}")
        proj_diag_parts.append("Min[" + ", ".join(min_parts) + "]")

    if proj_diag_parts:
        sections.append("ProjDiag[" + ", ".join(proj_diag_parts) + "]")

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
            "weighted_ap",
            "fpa",
            "fpa_projected",
            "tice",
            "tice_projected",
            "total",
            "level_ce",
            "gk_loss",
            "proj_has_negative",
        }
    )
    known_keys.update(neg_frac_keys)
    known_keys.update(min_keys)
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
