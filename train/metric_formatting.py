from typing import Dict, List, Optional


def _level_acc_keys(metrics: Dict[str, float], prefix: str) -> List[str]:
    keys = [k for k in metrics.keys() if k.startswith(prefix) and k[len(prefix) :].isdigit()]
    return sorted(keys, key=lambda k: int(k[len(prefix) :]))


def _format_ratio(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _format_scalar(value: float) -> str:
    return f"{float(value):.3f}"


def pretty_metrics(metrics: Dict[str, float], level_names: Optional[List[str]] = None) -> str:
    level_names = level_names or []
    sections: List[str] = []

    acc_ind_parts: List[str] = []
    for key in _level_acc_keys(metrics, "acc_level_independent_"):
        idx = int(key[len("acc_level_independent_") :])
        name = level_names[idx] if idx < len(level_names) else f"L{idx}"
        acc_ind_parts.append(f"{name}={_format_ratio(metrics[key])}")
    if acc_ind_parts:
        sections.append("\n\t\tAcc_ind[" + ", ".join(acc_ind_parts) + "]")

    acc_td_parts: List[str] = []
    for key in _level_acc_keys(metrics, "acc_level_topdown_"):
        idx = int(key[len("acc_level_topdown_") :])
        name = level_names[idx] if idx < len(level_names) else f"L{idx}"
        acc_td_parts.append(f"{name}={_format_ratio(metrics[key])}")
    if acc_td_parts:
        sections.append("Acc_td[" + ", ".join(acc_td_parts) + "]")

    summary_parts: List[str] = []
    if "weighted_ap_independent" in metrics:
        summary_parts.append(f"wAP_ind={_format_ratio(metrics['weighted_ap_independent'])}")
    if "weighted_ap_topdown" in metrics:
        summary_parts.append(f"wAP_td={_format_ratio(metrics['weighted_ap_topdown'])}")
    if "fpa_independent" in metrics:
        summary_parts.append(f"FPA_ind={_format_ratio(metrics['fpa_independent'])}")
    if "fpa_topdown" in metrics:
        summary_parts.append(f"FPA_td={_format_ratio(metrics['fpa_topdown'])}")
    if "tice_independent" in metrics:
        summary_parts.append(f"TICE_ind={_format_ratio(metrics['tice_independent'])}")
    if "tice_topdown" in metrics:
        summary_parts.append(f"TICE_td={_format_ratio(metrics['tice_topdown'])}")
    if "ahd_independent" in metrics:
        summary_parts.append(f"AHD_ind={_format_scalar(metrics['ahd_independent'])}")
    if "ahd_topdown" in metrics:
        summary_parts.append(f"AHD_td={_format_scalar(metrics['ahd_topdown'])}")
    if summary_parts:
        sections.append("Summary[" + ", ".join(summary_parts) + "]")

    loss_parts: List[str] = []
    if "total" in metrics:
        loss_parts.append(f"total={metrics['total']:.4f}")
    if "ce" in metrics:
        loss_parts.append(f"ce={metrics['ce']:.4f}")
    if "reg" in metrics:
        loss_parts.append(f"reg={metrics['reg']:.4f}")
    if "kl" in metrics:
        loss_parts.append(f"kl={metrics['kl']:.4f}")
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

    hcc_diag_keys = [
        "proj_constraint_alpha",
        "proj_logit_residual_before_l1",
        "proj_logit_residual_after_l1",
        "proj_logit_residual_reduction",
        "proj_logit_delta_l1_level_2",
        "proj_gt_logit_delta_level_2",
        "proj_delta_l1_level_2",
        "proj_flip_rate_level_2",
        "proj_gt_prob_delta_level_2",
        "acc_l2_ind_given_l1_correct",
        "acc_l2_td_given_l1_correct",
        "support_l1_ind_correct",
        "support_l1_td_correct",
        "gt_parent_mass_pre_l2",
        "gt_parent_mass_post_l2",
        "gt_child_rank_within_parent_pre_l2",
        "gt_child_rank_within_parent_post_l2",
    ]
    hcc_parts: List[str] = []
    for key in hcc_diag_keys:
        if key not in metrics:
            continue
        value = float(metrics[key])
        if key.startswith("acc_") or key.endswith("_rate_level_2") or key.startswith("support_"):
            hcc_parts.append(f"{key}={_format_ratio(value)}")
        else:
            hcc_parts.append(f"{key}={value:.4f}")
    if hcc_parts:
        sections.append("HCC[" + ", ".join(hcc_parts) + "]")

    if not sections:
        return "(no metrics)"
    return " | ".join(sections)
