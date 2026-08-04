from typing import Any, Dict, List, Optional

import torch

from .metrics import (
    _child_parent_tensors,
    average_hierarchical_distance,
    decoded_preds,
    full_path_accuracy,
    per_level_top1,
    tice_score,
    weighted_average_precision,
)


def _parent_mapping_for_level(
    taxonomy: Optional[Dict[str, Any]],
    level: int,
    num_children: int,
    num_parents: int,
) -> Optional[Dict[int, int]]:
    if not taxonomy or "parent_of" not in taxonomy:
        return None

    parent_of = taxonomy["parent_of"]
    if not isinstance(parent_of, dict):
        return None

    raw_mapping = parent_of.get(level, parent_of.get(str(level)))
    if not isinstance(raw_mapping, dict):
        return None

    mapping: Dict[int, int] = {}
    for child_raw, parent_raw in raw_mapping.items():
        try:
            child = int(child_raw)
            parent = int(parent_raw)
        except (TypeError, ValueError):
            continue
        if child < 0 or child >= num_children:
            continue
        if parent < 0 or parent >= num_parents:
            continue
        mapping[child] = parent

    if len(mapping) != num_children:
        return None
    return mapping


def _conditional_fine_accuracy(
    parent_pred: torch.Tensor,
    fine_pred: torch.Tensor,
    parent_target: torch.Tensor,
    fine_target: torch.Tensor,
) -> Dict[str, float]:
    parent_correct_mask = parent_pred.eq(parent_target)
    support = float(parent_correct_mask.float().mean().item())
    if bool(parent_correct_mask.any()):
        fine_acc = float(fine_pred[parent_correct_mask].eq(fine_target[parent_correct_mask]).float().mean().item())
    else:
        fine_acc = 0.0
    return {"acc": fine_acc, "support": support}


def _mean_gt_rank_within_parent(
    probs: torch.Tensor,
    gt_parent: torch.Tensor,
    gt_child: torch.Tensor,
    sibling_mask: torch.Tensor,
) -> Optional[float]:
    """Mean 1-based rank of the ground-truth child among its own siblings.

    `sibling_mask[parent, child]` marks the taxonomy's parent-child pairs. A
    sample is skipped when its ground-truth parent is out of range, has no
    children, or does not own the ground-truth child. Returns `None` when no
    sample qualifies.
    """
    num_parents, num_children = sibling_mask.shape
    if probs.numel() == 0:
        return None

    # Clamp only to keep the gathers in range; `usable` decides what counts.
    safe_parent = gt_parent.clamp(min=0, max=max(num_parents - 1, 0))
    safe_child = gt_child.clamp(min=0, max=max(num_children - 1, 0))
    siblings = sibling_mask[safe_parent]

    in_range = (
        gt_parent.ge(0)
        & gt_parent.lt(num_parents)
        & gt_child.ge(0)
        & gt_child.lt(num_children)
    )
    # A childless parent yields an all-false row, so this also drops those.
    owns_child = siblings.gather(1, safe_child.unsqueeze(1)).squeeze(1)
    usable = in_range & owns_child
    if not bool(usable.any()):
        return None

    gt_score = probs.gather(1, safe_child.unsqueeze(1))
    ranks = 1 + (probs.gt(gt_score) & siblings).sum(dim=1)
    # Sum then divide in Python. Ranks are small integers so the float64 sum is
    # exact, and dividing here matches the previous mean bit for bit, whereas
    # `Tensor.mean` multiplies by a reciprocal and can differ in the last ulp.
    total = float(ranks[usable].to(torch.float64).sum().item())
    return total / float(int(usable.sum().item()))


def evaluate_batch(
    output: Dict[str, Any],
    targets: torch.Tensor,
    taxonomy: Optional[Dict] = None,
    include_diagnostics: bool = True,
) -> Dict[str, float]:
    """Compute this batch's hierarchical metrics.

    `include_diagnostics=False` drops the level-3 probability-mass and rank
    diagnostics. They are logged during training but discarded by callers that
    only need the headline metrics, and they are a large share of the cost.
    """
    logits_per_level = output["logits_per_level"]
    effective_logits_per_level = output.get("effective_logits_per_level")
    has_effective_logits = (
        isinstance(effective_logits_per_level, list)
        and len(effective_logits_per_level) == len(logits_per_level)
    )
    if effective_logits_per_level is not None and not has_effective_logits:
        raise ValueError("`effective_logits_per_level` must be None or a list aligned with logits levels.")

    effective_probs_per_level = output.get("effective_probs_per_level")
    has_effective_probs = (
        isinstance(effective_probs_per_level, list)
        and len(effective_probs_per_level) == len(logits_per_level)
    )
    if effective_probs_per_level is not None and not has_effective_probs:
        raise ValueError("`effective_probs_per_level` must be None or a list aligned with logits levels.")

    # Metrics are computed from final model scores:
    # projected logits when available, legacy projected probabilities when
    # available, otherwise raw logits.
    if has_effective_logits:
        score_source = effective_logits_per_level
    elif has_effective_probs:
        score_source = effective_probs_per_level
    else:
        score_source = logits_per_level

    # Decode once per decoder. Every metric below is a function of these
    # predictions, so decoding inside each one repeated identical work.
    preds_ind = decoded_preds(score_source, taxonomy=taxonomy, enforce_hierarchy=False)
    preds_td = decoded_preds(score_source, taxonomy=taxonomy, enforce_hierarchy=True)

    metrics: Dict[str, float] = {}
    metrics.update(
        per_level_top1(
            score_source,
            targets,
            taxonomy=taxonomy,
            enforce_hierarchy=False,
            key_prefix="acc_level_independent_",
            preds=preds_ind,
        )
    )
    metrics.update(
        per_level_top1(
            score_source,
            targets,
            taxonomy=taxonomy,
            enforce_hierarchy=True,
            key_prefix="acc_level_topdown_",
            preds=preds_td,
        )
    )
    metrics["weighted_ap_independent"] = weighted_average_precision(
        score_source,
        targets,
        taxonomy=taxonomy,
        enforce_hierarchy=False,
        preds=preds_ind,
    )
    metrics["weighted_ap_topdown"] = weighted_average_precision(
        score_source,
        targets,
        taxonomy=taxonomy,
        enforce_hierarchy=True,
        preds=preds_td,
    )

    # H-CAST: FPA is full-path exact-match accuracy.
    # Save both decoding variants on final scores:
    # 1) independent argmax per level, 2) top-down hierarchical argmax.
    metrics["fpa_independent"] = full_path_accuracy(
        score_source,
        targets,
        taxonomy=taxonomy,
        enforce_hierarchy=False,
        preds=preds_ind,
    )
    metrics["fpa_topdown"] = full_path_accuracy(
        score_source,
        targets,
        taxonomy=taxonomy,
        enforce_hierarchy=True,
        preds=preds_td,
    )
    metrics["ahd_independent"] = average_hierarchical_distance(
        score_source,
        targets,
        taxonomy=taxonomy,
        enforce_hierarchy=False,
        preds=preds_ind,
    )
    metrics["ahd_topdown"] = average_hierarchical_distance(
        score_source,
        targets,
        taxonomy=taxonomy,
        enforce_hierarchy=True,
        preds=preds_td,
    )

    # H-CAST: TICE is inconsistency rate (lower is better).
    tice_independent = tice_score(
        score_source,
        taxonomy,
        enforce_hierarchy=False,
        preds=preds_ind,
    )
    tice_topdown = tice_score(
        score_source,
        taxonomy,
        enforce_hierarchy=True,
        preds=preds_td,
    )
    if tice_independent is not None:
        metrics["tice_independent"] = tice_independent
    if tice_topdown is not None:
        metrics["tice_topdown"] = tice_topdown

    if include_diagnostics and len(logits_per_level) >= 3:
        # Always define diagnostics on probability scores:
        # pre = raw softmax(logits), post = softmax(final logits) or final probabilities.
        pre_probs = [torch.softmax(logits, dim=-1) for logits in logits_per_level]
        if has_effective_logits:
            post_probs = [torch.softmax(logits, dim=-1) for logits in effective_logits_per_level]
        elif has_effective_probs:
            post_probs = effective_probs_per_level
        else:
            post_probs = pre_probs

        level_middle = 1
        level_fine = 2
        fine_pre_logits = logits_per_level[level_fine]
        fine_post_logits = (
            effective_logits_per_level[level_fine]
            if has_effective_logits
            else logits_per_level[level_fine]
        )
        fine_pre = pre_probs[level_fine]
        fine_post = post_probs[level_fine]

        gt_middle = targets[:, level_middle].long()
        gt_fine = targets[:, level_fine].long()
        row_ids = torch.arange(targets.size(0), device=targets.device)

        if has_effective_logits:
            metrics["proj_logit_delta_l1_level_2"] = float(
                (fine_post_logits - fine_pre_logits).abs().sum(dim=-1).mean().item()
            )
            metrics["proj_gt_logit_delta_level_2"] = float(
                (fine_post_logits[row_ids, gt_fine] - fine_pre_logits[row_ids, gt_fine]).mean().item()
            )
        metrics["proj_delta_l1_level_2"] = float((fine_post - fine_pre).abs().sum(dim=-1).mean().item())
        metrics["proj_flip_rate_level_2"] = float(
            fine_pre.argmax(dim=-1).ne(fine_post.argmax(dim=-1)).float().mean().item()
        )
        metrics["proj_gt_prob_delta_level_2"] = float(
            (fine_post[row_ids, gt_fine] - fine_pre[row_ids, gt_fine]).mean().item()
        )

        if len(preds_ind) >= 3:
            cond_ind = _conditional_fine_accuracy(
                parent_pred=preds_ind[level_middle],
                fine_pred=preds_ind[level_fine],
                parent_target=gt_middle,
                fine_target=gt_fine,
            )
            metrics["acc_l2_ind_given_l1_correct"] = cond_ind["acc"]
            metrics["support_l1_ind_correct"] = cond_ind["support"]
        if len(preds_td) >= 3:
            cond_td = _conditional_fine_accuracy(
                parent_pred=preds_td[level_middle],
                fine_pred=preds_td[level_fine],
                parent_target=gt_middle,
                fine_target=gt_fine,
            )
            metrics["acc_l2_td_given_l1_correct"] = cond_td["acc"]
            metrics["support_l1_td_correct"] = cond_td["support"]

        num_middle = int(post_probs[level_middle].size(-1))
        num_fine = int(fine_post.size(-1))
        parent_of_fine = _parent_mapping_for_level(
            taxonomy=taxonomy,
            level=2,
            num_children=num_fine,
            num_parents=num_middle,
        )
        if parent_of_fine is not None:
            # Cached, so the mask is not rebuilt element by element every batch.
            sibling_mask, _ = _child_parent_tensors(
                parent_of_fine,
                num_parents=num_middle,
                num_children=num_fine,
                device=fine_post.device,
            )

            gt_parent_mask = sibling_mask[gt_middle].to(dtype=fine_post.dtype)
            metrics["gt_parent_mass_pre_l2"] = float((fine_pre * gt_parent_mask).sum(dim=-1).mean().item())
            metrics["gt_parent_mass_post_l2"] = float((fine_post * gt_parent_mask).sum(dim=-1).mean().item())

            rank_pre = _mean_gt_rank_within_parent(
                probs=fine_pre,
                gt_parent=gt_middle,
                gt_child=gt_fine,
                sibling_mask=sibling_mask,
            )
            rank_post = _mean_gt_rank_within_parent(
                probs=fine_post,
                gt_parent=gt_middle,
                gt_child=gt_fine,
                sibling_mask=sibling_mask,
            )
            if rank_pre is not None:
                metrics["gt_child_rank_within_parent_pre_l2"] = rank_pre
            if rank_post is not None:
                metrics["gt_child_rank_within_parent_post_l2"] = rank_post

    hcc_diagnostics = output.get("hcc_diagnostics")
    if isinstance(hcc_diagnostics, dict):
        for key, value in hcc_diagnostics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    metrics[key] = float(value.detach().item())
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)

    return metrics
