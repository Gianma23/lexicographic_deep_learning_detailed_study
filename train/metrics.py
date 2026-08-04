from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


# Per-level-transition tensors derived only from the taxonomy. Rebuilding these
# element by element on the accelerator once per batch dominated evaluation
# time, so they are cached per (mapping, shape, device). The contents are a pure
# function of the taxonomy, so caching cannot change any metric value.
_TAXONOMY_TENSOR_CACHE: Dict[Any, Dict[str, torch.Tensor]] = {}


def _argmax_preds(logits_per_level: List[torch.Tensor]) -> List[torch.Tensor]:
    return [logits.argmax(dim=-1) for logits in logits_per_level]


def _child_parent_tensors(
    mapping: Dict[int, int],
    num_parents: int,
    num_children: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(allowed[num_parents, num_children], parent_lookup[num_children])`.

    `allowed` marks the valid parent-child pairs for top-down decoding, and
    `parent_lookup` maps each child to its parent for consistency checks, with
    `-1` where the taxonomy defines none.
    """
    key = (
        tuple(sorted(mapping.items())),
        int(num_parents),
        int(num_children),
        str(device),
    )
    cached = _TAXONOMY_TENSOR_CACHE.get(key)
    if cached is None:
        allowed = torch.zeros((num_parents, num_children), dtype=torch.bool)
        parent_lookup = torch.full((num_children,), -1, dtype=torch.long)
        in_range_children: List[int] = []
        in_range_parents: List[int] = []
        lookup_children: List[int] = []
        lookup_parents: List[int] = []
        for child_id, parent_id in mapping.items():
            child_id = int(child_id)
            parent_id = int(parent_id)
            if 0 <= child_id < num_children and 0 <= parent_id < num_parents:
                in_range_children.append(child_id)
                in_range_parents.append(parent_id)
            # The lookup keeps out-of-range parents so that such a child can
            # never match a decoded parent, exactly as the per-element build did.
            if 0 <= child_id < num_children:
                lookup_children.append(child_id)
                lookup_parents.append(parent_id)
        if in_range_children:
            allowed[
                torch.tensor(in_range_parents, dtype=torch.long),
                torch.tensor(in_range_children, dtype=torch.long),
            ] = True
        if lookup_children:
            parent_lookup[torch.tensor(lookup_children, dtype=torch.long)] = torch.tensor(
                lookup_parents, dtype=torch.long
            )
        cached = {
            "allowed": allowed.to(device=device),
            "parent_lookup": parent_lookup.to(device=device),
        }
        _TAXONOMY_TENSOR_CACHE[key] = cached
    return cached["allowed"], cached["parent_lookup"]


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

        allowed, _ = _child_parent_tensors(
            mapping,
            num_parents=num_parents,
            num_children=num_children,
            device=scores.device,
        )

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
    preds: Optional[List[torch.Tensor]] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if preds is None:
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
    preds: Optional[List[torch.Tensor]] = None,
) -> float:
    """H-CAST wAP: class-count-weighted Top-1 accuracy across levels."""
    if not logits_per_level:
        return 0.0

    if preds is None:
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


def full_path_accuracy(
    logits_per_level: List[torch.Tensor],
    targets: torch.Tensor,
    taxonomy: Optional[Dict[str, Any]] = None,
    enforce_hierarchy: bool = False,
    preds: Optional[List[torch.Tensor]] = None,
) -> float:
    if preds is None:
        preds = _decoded_preds(logits_per_level, taxonomy, enforce_hierarchy)
    pred_path = torch.stack(preds, dim=1)
    return float((pred_path == targets).all(dim=1).float().mean().item())


def average_hierarchical_distance(
    logits_per_level: List[torch.Tensor],
    targets: torch.Tensor,
    taxonomy: Optional[Dict[str, Any]] = None,
    enforce_hierarchy: bool = False,
    preds: Optional[List[torch.Tensor]] = None,
) -> float:
    """Average LCA-equivalent hierarchical distance between predicted and GT paths."""
    if not logits_per_level:
        return 0.0

    if preds is None:
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
    preds: Optional[List[torch.Tensor]] = None,
) -> Optional[float]:
    if not taxonomy or "parent_of" not in taxonomy:
        return None

    parent_of = _normalize_parent_of(taxonomy)
    if preds is None:
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

        # Cached child->parent lookup, used to check whole-path validity.
        _, lookup = _child_parent_tensors(
            mapping,
            num_parents=int(logits_per_level[level - 1].size(-1)),
            num_children=int(logits_per_level[level].size(-1)),
            device=child.device,
        )

        mapped_parent = lookup[child].to(dtype=parent.dtype)
        valid = valid & (mapped_parent == parent)

    if valid.numel() == 0:
        return None
    return float(valid.float().mean().item())


def tice_score(
    logits_per_level: List[torch.Tensor],
    taxonomy: Optional[Dict[str, Any]],
    enforce_hierarchy: bool = False,
    preds: Optional[List[torch.Tensor]] = None,
) -> Optional[float]:
    """H-CAST convention: TICE is inconsistency rate (lower is better)."""
    consistency = consistency_rate(
        logits_per_level, taxonomy, enforce_hierarchy=enforce_hierarchy, preds=preds
    )
    if consistency is None:
        return None
    return float(1.0 - consistency)


def merge_metric_batches(
    metric_batches: List[Dict[str, float]],
    batch_weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    if not metric_batches:
        return {}
    if batch_weights is None:
        weights = [1.0] * len(metric_batches)
    else:
        if len(batch_weights) != len(metric_batches):
            raise ValueError("batch_weights must align with metric_batches.")
        weights = [float(weight) for weight in batch_weights]
        if any(weight < 0.0 for weight in weights):
            raise ValueError("batch_weights must be non-negative.")

    keys = set().union(*[metrics.keys() for metrics in metric_batches])
    out: Dict[str, float] = {}
    for key in keys:
        weighted_sum = 0.0
        total_weight = 0.0
        for metrics, batch_weight in zip(metric_batches, weights):
            if key not in metrics:
                continue

            metric_weight = batch_weight
            if key == "acc_l2_ind_given_l1_correct":
                metric_weight *= float(metrics.get("support_l1_ind_correct", 0.0))
            elif key == "acc_l2_td_given_l1_correct":
                metric_weight *= float(metrics.get("support_l1_td_correct", 0.0))

            weighted_sum += float(metrics[key]) * metric_weight
            total_weight += metric_weight
        if total_weight > 0.0:
            out[key] = float(weighted_sum / total_weight)
        elif key in {"acc_l2_ind_given_l1_correct", "acc_l2_td_given_l1_correct"}:
            out[key] = 0.0
    return out
