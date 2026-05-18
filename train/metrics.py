from typing import Any, Dict, List, Optional

import torch


def _argmax_preds(logits_per_level: List[torch.Tensor]) -> List[torch.Tensor]:
    return [logits.argmax(dim=-1) for logits in logits_per_level]


def _normalize_parent_of(taxonomy: Optional[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
    if not taxonomy or "parent_of" not in taxonomy:
        return {}

    parent_of = taxonomy["parent_of"]
    if not isinstance(parent_of, dict):
        return {}

    out: Dict[int, Dict[int, int]] = {}
    for level_key, mapping in parent_of.items():
        if not isinstance(mapping, dict):
            continue
        level = int(level_key)
        out[level] = {int(child): int(parent) for child, parent in mapping.items()}
    return out


def _hierarchical_argmax_preds(
    logits_per_level: List[torch.Tensor],
    taxonomy: Optional[Dict[str, Any]],
) -> List[torch.Tensor]:
    if not logits_per_level:
        return []
    parent_of = _normalize_parent_of(taxonomy)
    if not parent_of:
        return _argmax_preds(logits_per_level)

    preds: List[torch.Tensor] = [logits_per_level[0].argmax(dim=-1)]
    for level in range(1, len(logits_per_level)):
        mapping = parent_of.get(level)
        if not mapping:
            return _argmax_preds(logits_per_level)

        scores = logits_per_level[level]
        parent_pred = preds[level - 1]
        num_children = int(scores.size(-1))
        num_parents = int(logits_per_level[level - 1].size(-1))

        allowed = torch.zeros((num_parents, num_children), dtype=torch.bool, device=scores.device)
        for child_id, parent_id in mapping.items():
            if 0 <= child_id < num_children and 0 <= parent_id < num_parents:
                allowed[parent_id, child_id] = True

        allowed_batch = allowed[parent_pred]
        masked_scores = scores.masked_fill(~allowed_batch, float("-inf"))
        child_pred = masked_scores.argmax(dim=-1)

        has_allowed = allowed_batch.any(dim=-1)
        if not bool(has_allowed.all()):
            fallback = scores.argmax(dim=-1)
            child_pred = torch.where(has_allowed, child_pred, fallback)

        preds.append(child_pred)

    return preds


def _decoded_preds(
    logits_per_level: List[torch.Tensor],
    taxonomy: Optional[Dict[str, Any]],
    enforce_hierarchy: bool,
) -> List[torch.Tensor]:
    if enforce_hierarchy:
        return _hierarchical_argmax_preds(logits_per_level, taxonomy)
    return _argmax_preds(logits_per_level)


def decoded_preds(
    logits_per_level: List[torch.Tensor],
    taxonomy: Optional[Dict[str, Any]] = None,
    enforce_hierarchy: bool = False,
) -> List[torch.Tensor]:
    """Public wrapper around internal decoding utility."""
    return _decoded_preds(logits_per_level, taxonomy, enforce_hierarchy)


def per_level_top1(
    logits_per_level: List[torch.Tensor],
    targets: torch.Tensor,
    taxonomy: Optional[Dict[str, Any]] = None,
    enforce_hierarchy: bool = False,
    key_prefix: str = "acc_level_",
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    preds = _decoded_preds(logits_per_level, taxonomy, enforce_hierarchy)
    for level, pred in enumerate(preds):
        acc = (pred == targets[:, level]).float().mean().item()
        out[f"{key_prefix}{level}"] = float(acc)
    return out


def weighted_average_precision(
    logits_per_level: List[torch.Tensor],
    targets: torch.Tensor,
    taxonomy: Optional[Dict[str, Any]] = None,
    enforce_hierarchy: bool = False,
) -> float:
    """H-CAST wAP: class-count-weighted Top-1 accuracy across levels."""
    if not logits_per_level:
        return 0.0

    preds = _decoded_preds(logits_per_level, taxonomy, enforce_hierarchy)
    weighted_sum = 0.0
    total_weight = 0.0
    for level, logits in enumerate(logits_per_level):
        pred = preds[level]
        acc = float((pred == targets[:, level]).float().mean().item())
        weight = float(logits.size(-1))
        weighted_sum += weight * acc
        total_weight += weight

    if total_weight <= 0.0:
        return 0.0
    return float(weighted_sum / total_weight)


def weighted_accuracy(logits_per_level: List[torch.Tensor], targets: torch.Tensor) -> float:
    # Alias.
    return weighted_average_precision(logits_per_level, targets)


def full_path_accuracy(
    logits_per_level: List[torch.Tensor],
    targets: torch.Tensor,
    taxonomy: Optional[Dict[str, Any]] = None,
    enforce_hierarchy: bool = False,
) -> float:
    preds = _decoded_preds(logits_per_level, taxonomy, enforce_hierarchy)
    pred_path = torch.stack(preds, dim=1)
    return float((pred_path == targets).all(dim=1).float().mean().item())


def average_hierarchical_distance(
    logits_per_level: List[torch.Tensor],
    targets: torch.Tensor,
    taxonomy: Optional[Dict[str, Any]] = None,
    enforce_hierarchy: bool = False,
) -> float:
    """Average LCA-equivalent hierarchical distance between predicted and GT paths.

    For each sample i with hierarchy depth L:
    1) decode predicted full path,
    2) compute longest shared prefix length s_i with target path,
    3) distance d_H(i) = L - s_i.

    This is equivalent to the paper-style LCA-height distance for fixed-depth
    rooted hierarchies and preserves lower-is-better semantics.
    """
    if not logits_per_level:
        return 0.0

    preds = _decoded_preds(logits_per_level, taxonomy, enforce_hierarchy)
    if not preds:
        return 0.0

    pred_path = torch.stack(preds, dim=1)
    if pred_path.numel() == 0:
        return 0.0

    depth = int(pred_path.size(1))
    # Prefix match mask per sample/level, then count initial contiguous matches.
    prefix_matches = pred_path.eq(targets).to(torch.int64).cumprod(dim=1)
    shared_prefix_len = prefix_matches.sum(dim=1).to(dtype=torch.float32)
    distances = float(depth) - shared_prefix_len
    return float(distances.mean().item())


def consistency_rate(
    logits_per_level: List[torch.Tensor],
    taxonomy: Optional[Dict[str, Any]],
    enforce_hierarchy: bool = False,
) -> Optional[float]:
    if not taxonomy or "parent_of" not in taxonomy:
        return None

    parent_of = _normalize_parent_of(taxonomy)
    preds = _decoded_preds(logits_per_level, taxonomy, enforce_hierarchy)
    if not preds:
        return None

    valid = torch.ones_like(preds[0], dtype=torch.bool)
    for level in range(1, len(preds)):
        mapping = parent_of.get(level)
        if mapping is None:
            return None

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


def tice_score(
    logits_per_level: List[torch.Tensor],
    taxonomy: Optional[Dict[str, Any]],
    enforce_hierarchy: bool = False,
) -> Optional[float]:
    """H-CAST convention: TICE is inconsistency rate (lower is better)."""
    consistency = consistency_rate(logits_per_level, taxonomy, enforce_hierarchy=enforce_hierarchy)
    if consistency is None:
        return None
    return float(1.0 - consistency)


def inconsistency_rate(
    logits_per_level: List[torch.Tensor],
    taxonomy: Optional[Dict[str, Any]],
    enforce_hierarchy: bool = False,
) -> Optional[float]:
    # Backward-compatible alias used by existing callers/docs.
    return tice_score(logits_per_level, taxonomy, enforce_hierarchy=enforce_hierarchy)


def tice_like_score(
    logits_per_level: List[torch.Tensor],
    taxonomy: Optional[Dict[str, Any]],
    enforce_hierarchy: bool = False,
) -> Optional[float]:
    # Backward-compatible alias from previous implementation.
    tice = tice_score(logits_per_level, taxonomy, enforce_hierarchy=enforce_hierarchy)
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
