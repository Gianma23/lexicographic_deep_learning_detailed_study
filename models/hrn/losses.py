from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


HrnTargets = Union[torch.Tensor, Dict[str, Any]]


def _hard_targets_from_input(targets: HrnTargets) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        return targets
    if not isinstance(targets, dict):
        raise TypeError("HRN targets must be a tensor or dict.")

    hard_targets = targets.get("hard_targets")
    if isinstance(hard_targets, torch.Tensor):
        return hard_targets

    labels_a = targets.get("labels_a")
    if isinstance(labels_a, torch.Tensor):
        return labels_a

    raise TypeError("HRN expected `hard_targets` (or legacy `labels_a`) tensor in target dict.")


def _normalize_parent_of(taxonomy: Optional[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
    if not taxonomy or "parent_of" not in taxonomy:
        return {}
    normalized: Dict[int, Dict[int, int]] = {}
    for level_key, mapping in taxonomy["parent_of"].items():
        level = int(level_key)
        normalized[level] = {int(child): int(parent) for child, parent in mapping.items()}
    return normalized


def _level_offsets(num_classes_per_level: List[int]) -> List[int]:
    offsets = [0]
    for n in num_classes_per_level[:-1]:
        offsets.append(offsets[-1] + int(n))
    return offsets


def _build_state_space(
    num_classes_per_level: List[int],
    parent_of: Dict[int, Dict[int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if len(num_classes_per_level) != 3:
        return None

    n0, n1, n2 = [int(n) for n in num_classes_per_level]
    map_1 = parent_of.get(1, {})
    map_2 = parent_of.get(2, {})

    required_l1 = set(range(n1))
    required_l2 = set(range(n2))
    if set(map_1.keys()) != required_l1 or set(map_2.keys()) != required_l2:
        return None
    if any(parent not in required_l1 for parent in map_2.values()):
        return None
    if any(parent not in set(range(n0)) for parent in map_1.values()):
        return None

    total_nodes = n0 + n1 + n2
    off0, off1, off2 = _level_offsets(num_classes_per_level)

    # Upstream `TreeLoss` includes a leading all-zero row.
    rows: List[torch.Tensor] = [torch.zeros(total_nodes, device=device, dtype=dtype)]

    recorded_l0 = [False] * n0
    recorded_l1 = [False] * n1
    for leaf in range(n2):
        l1 = int(map_2[leaf])
        l0 = int(map_1[l1])

        if not recorded_l0[l0]:
            row0 = torch.zeros(total_nodes, device=device, dtype=dtype)
            row0[off0 + l0] = 1.0
            rows.append(row0)
            recorded_l0[l0] = True

        if not recorded_l1[l1]:
            row1 = torch.zeros(total_nodes, device=device, dtype=dtype)
            row1[off0 + l0] = 1.0
            row1[off1 + l1] = 1.0
            rows.append(row1)
            recorded_l1[l1] = True

        row2 = torch.zeros(total_nodes, device=device, dtype=dtype)
        row2[off0 + l0] = 1.0
        row2[off1 + l1] = 1.0
        row2[off2 + leaf] = 1.0
        rows.append(row2)

    return torch.stack(rows, dim=0)


def _resolve_observed_indices(
    hard_targets: torch.Tensor,
    num_classes_per_level: List[int],
    targets: HrnTargets,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(hard_targets.size(0))
    device = hard_targets.device
    offsets = _level_offsets(num_classes_per_level)

    if isinstance(targets, dict):
        observed_global = targets.get("observed_global_index")
        if isinstance(observed_global, torch.Tensor):
            observed_global = observed_global.to(device=device, dtype=torch.long).view(-1)
            if observed_global.numel() != batch_size:
                raise ValueError("`observed_global_index` must have one entry per sample.")
            observed_level = targets.get("observed_level")
            if isinstance(observed_level, torch.Tensor):
                observed_level = observed_level.to(device=device, dtype=torch.long).view(-1)
                if observed_level.numel() != batch_size:
                    raise ValueError("`observed_level` must have one entry per sample.")
            else:
                # Backward-compatible default for current full-path targets.
                observed_level = torch.full((batch_size,), 2, device=device, dtype=torch.long)
            return observed_global, observed_level

        observed_level = targets.get("observed_level")
        if isinstance(observed_level, torch.Tensor):
            observed_level = observed_level.to(device=device, dtype=torch.long).view(-1)
            if observed_level.numel() != batch_size:
                raise ValueError("`observed_level` must have one entry per sample.")
            gathered = hard_targets.gather(1, observed_level.unsqueeze(1)).squeeze(1).long()
            observed_global = gathered + torch.tensor(offsets, device=device, dtype=torch.long)[observed_level]
            return observed_global, observed_level

    # Default behavior: treat current labels as leaf-observed samples.
    observed_level = torch.full((batch_size,), 2, device=device, dtype=torch.long)
    observed_global = hard_targets[:, 2].long() + int(offsets[2])
    return observed_global, observed_level


def _hierarchical_loss(tree_scores_per_level: List[torch.Tensor], observed_global: torch.Tensor, state_space: torch.Tensor) -> torch.Tensor:
    fs = torch.cat(tree_scores_per_level, dim=1).to(dtype=state_space.dtype)  # [B, N]
    joint = torch.exp(torch.matmul(state_space, fs.T))  # [S, B]
    z = joint.sum(dim=0).clamp_min(1e-12)  # [B]
    eligible = state_space[:, observed_global.long()] > 0  # [S, B]
    marginal = (joint * eligible.to(dtype=joint.dtype)).sum(dim=0).clamp_min(1e-12)
    return (-torch.log(marginal / z)).to(dtype=torch.float64).mean()


def compute_loss(
    output: Dict[str, Any],
    targets: HrnTargets,
    cfg: Any,
    taxonomy: Optional[Dict[str, Any]] = None,
    return_aux: bool = False,
) -> Union[
    Tuple[torch.Tensor, Dict[str, float]],
    Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]],
]:
    if isinstance(targets, dict):
        raise ValueError(
            "HRN paper-reproduction mode supports full leaf-labeled samples only; "
            "partial/relabeling targets and mixup/cutmix target dictionaries are unsupported."
        )

    hard_targets = _hard_targets_from_input(targets)
    if hard_targets.ndim != 2 or int(hard_targets.size(1)) != 3:
        raise ValueError(f"HRN expects hard targets of shape [B, 3], got {tuple(hard_targets.shape)}.")

    logits_per_level = output.get("logits_per_level")
    has_logits_per_level = isinstance(logits_per_level, list) and len(logits_per_level) == 3
    if logits_per_level is not None and not has_logits_per_level:
        raise ValueError("HRN output `logits_per_level` must be a 3-level list when provided.")

    tree_scores_per_level = output.get("tree_scores_per_level")
    if not isinstance(tree_scores_per_level, list) or len(tree_scores_per_level) != 3:
        raise ValueError("HRN output must provide `tree_scores_per_level` with 3 tensors.")

    species_ce_logits = output.get("species_ce_logits")
    if not isinstance(species_ce_logits, torch.Tensor):
        if not has_logits_per_level:
            raise ValueError("HRN output must provide `species_ce_logits` or a 3-level `logits_per_level`.")
        species_ce_logits = logits_per_level[2]

    num_classes_per_level = [int(x.size(-1)) for x in tree_scores_per_level]
    parent_of = _normalize_parent_of(taxonomy)
    state_space = _build_state_space(
        num_classes_per_level=num_classes_per_level,
        parent_of=parent_of,
        device=tree_scores_per_level[0].device,
        dtype=torch.float32,
    )
    if state_space is None:
        raise ValueError("HRN combinatorial loss requires a complete 3-level taxonomy (`parent_of` maps for levels 1 and 2).")

    observed_global, observed_level = _resolve_observed_indices(hard_targets=hard_targets, num_classes_per_level=num_classes_per_level, targets=targets)
    hier_loss = _hierarchical_loss(tree_scores_per_level=tree_scores_per_level, observed_global=observed_global, state_space=state_space)

    leaf_mask = observed_level == 2
    if bool(leaf_mask.any()):
        ce_loss = F.cross_entropy(
            species_ce_logits[leaf_mask].to(dtype=torch.float64),
            hard_targets[leaf_mask, 2].long(),
        )
    else:
        ce_loss = hier_loss.new_zeros(())
    total = hier_loss + ce_loss

    metrics = {
        "total": float(total.detach().item()),
        "hier_loss": float(hier_loss.detach().item()),
        "ce_loss_leaf": float(ce_loss.detach().item()),
        # Backward-compatible aliases used by existing logs.
        "tree_loss": float(hier_loss.detach().item()),
        "fine_ce": float(ce_loss.detach().item()),
    }
    if not return_aux:
        return total, metrics

    aux_payload: Dict[str, Any] = {
        "observed_global_index": observed_global,
        "observed_level": observed_level,
        "leaf_mask": leaf_mask,
    }
    return total, metrics, aux_payload
